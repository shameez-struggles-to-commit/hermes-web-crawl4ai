"""Crawl4AI extraction provider and bounded crawl tool for Hermes Agent."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from agent.web_search_provider import WebSearchProvider, get_provider_env
from tools.url_safety import async_is_safe_url, normalize_url_for_request
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

PLUGIN_ID = "hermes-web-crawl4ai"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_CHARS = 20_000
MAX_CRAWL_PAGES = 20


# ---------------------------------------------------------------------------
# Content-quality gate + failure classification
# ---------------------------------------------------------------------------
# Crawl4AI's /md endpoint returns `success: true` for a JS-rendered bot
# challenge (Cloudflare "checking your browser", etc.) and collapses every
# non-success (404, 403, 429, PDF download) into a bare HTTP 500 with the real
# reason only in container logs. To give the agent a usable signal, classify
# results ourselves: a content gate rejects walls/empty shells, and a typed
# taxonomy (not_found / blocked / rate_limited / challenge / empty /
# unsupported_media / pdf_error / upstream_error) replaces the raw httpx string.

_MIN_CONTENT_CHARS = 60

_CHALLENGE_PATTERNS = (
    (re.compile(r"checking\s+your\s+browser", re.IGNORECASE), "challenge"),
    (re.compile(r"just\s+a\s+moment", re.IGNORECASE), "challenge"),
    (re.compile(r"verify\s+you\s+are\s+(a\s+)?human", re.IGNORECASE), "challenge"),
    (re.compile(r"enable\s+javascript\s+and\s+cookies", re.IGNORECASE), "challenge"),
    (re.compile(r"are\s+you\s+a\s+robot", re.IGNORECASE), "challenge"),
    (re.compile(r"pardon\s+our\s+interruption", re.IGNORECASE), "challenge"),
    (re.compile(r"complete\s+the\s+security\s+check", re.IGNORECASE), "challenge"),
    (re.compile(r"one\s+more\s+step", re.IGNORECASE), "challenge"),
)

_BLOCK_PATTERNS = (
    (re.compile(r"access\s+denied", re.IGNORECASE), "blocked"),
    (re.compile(r"403\s+forbidden", re.IGNORECASE), "blocked"),
    (re.compile(r"request\s+unsuccessful", re.IGNORECASE), "blocked"),
    (re.compile(r"rate\s+limit(ed)?\s+(exceeded|reached)", re.IGNORECASE), "rate_limited"),
    (re.compile(r"too\s+many\s+requests", re.IGNORECASE), "rate_limited"),
)

# 'captcha' is a strong signal but also appears in legitimate articles; only
# flag it on short pages (a challenge wall) to avoid false positives.
_CAPTCHA_RE = re.compile(r"captcha", re.IGNORECASE)
_CAPTCHA_MAX_LEN = 500

_GATE_MESSAGES = {
    "challenge": "Page returned a bot/challenge wall instead of content",
    "blocked": "Page blocked automated access",
    "rate_limited": "Page rate-limited the request",
    "empty": "Page returned no extractable content",
}

# Exception-path failures a real browser can plausibly rescue: the target
# blocked/challenged us (401/403 → blocked; 5xx incl. Cloudflare's 503
# challenge → upstream_error). Combined with ``probe_status is None`` at the
# call site this covers unclassifiable targets too. Deliberately excludes
# rate_limited (browser shares the IP/limit), not_found (404/410 return
# earlier), and unmapped statuses (200/204/400/405 … → "crawl4ai_error"):
# a mapped-healthy probe with Crawl4AI erroring means OUR infrastructure is
# the suspect, and escalation would only mask the outage.
_STEALTHABLE_ERROR_TYPES = frozenset({"blocked", "upstream_error"})


def _classify_content(content: str) -> Optional[str]:
    """Return a failure classification when ``content`` looks like a block /
    challenge wall or an empty shell rather than real page content; else None."""
    text = (content or "").strip()
    if not text:
        return "empty"
    for pattern, kind in _CHALLENGE_PATTERNS + _BLOCK_PATTERNS:
        if pattern.search(text):
            return kind
    if len(text) < _CAPTCHA_MAX_LEN and _CAPTCHA_RE.search(text):
        return "challenge"
    if len(text) < _MIN_CONTENT_CHARS:
        return "empty"
    return None


def _status_to_error_type(status: int) -> Optional[str]:
    if status in (404, 410):
        return "not_found"
    if status in (401, 403):
        return "blocked"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "upstream_error"
    return None


def _typed_error_document(url: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "metadata": {},
        "error": message,
        "error_type": error_type,
    }


_PROBE_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


async def _probe_status(url: str) -> tuple[Optional[int], Optional[str]]:
    """Best-effort HEAD probe to recover the target's real HTTP status +
    content-type (Crawl4AI's /md hides both). Never raises; a failed probe
    yields (None, None). Follows redirects so a redirect-to-PDF is detected."""
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            resp = await client.head(url, headers={"User-Agent": _PROBE_UA})
            return resp.status_code, resp.headers.get("content-type", "")
    except Exception:  # noqa: BLE001
        pass
    # Some servers reject HEAD (405) or hang; retry with a streamed GET that
    # reads no body.
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            async with client.stream(
                "GET", url, headers={"User-Agent": _PROBE_UA}
            ) as resp:
                return resp.status_code, resp.headers.get("content-type", "")
    except Exception:  # noqa: BLE001
        return None, None


def _url_is_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


async def _extract_pdf(url: str) -> dict[str, Any]:
    """Extract a PDF via Hermes' own document extractor (tools.read_extract →
    firecrawl-anydoc, lazily installed on first use). Reuses core's scanned-PDF
    coverage detection. Never raises — failures become a typed error document."""
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Accept": "application/pdf,*/*"})
            resp.raise_for_status()
            data = resp.content
        if not data:
            return _typed_error_document(url, "empty", "PDF returned no bytes")
        from tools.read_extract import extract_document_bytes

        text = extract_document_bytes(data, url)
    except Exception as exc:  # noqa: BLE001
        return _typed_error_document(url, "pdf_error", f"PDF extraction failed: {exc}")
    if not text or not text.strip():
        return _typed_error_document(url, "empty", "PDF contained no extractable text")
    return {
        "url": url,
        "title": "",
        "content": text,
        "raw_content": text,
        "metadata": {"resolved_by": "read_extract"},
    }


_SCRAPLING_SPEC = "scrapling[fetchers]==0.4.14"
_SCRAPLING_INSTALL_ENV = "HERMES_WEB_CRAWL4AI_NO_INSTALL"
_SCRAPLING_INSTALL_TIMEOUT_SECONDS = 600
# Waiters give up slightly after the pip deadline so the install outcome is
# normally known; the background install itself is never aborted by a waiter.
_SCRAPLING_RESOLVE_DEADLINE_SECONDS = 620

# Single-flight resolution: the first caller installs; everyone else reuses
# the outcome. Failed installs are cached for the process lifetime (retrying
# pip on every challenge would hammer a broken index).
_scrapling_lock = threading.Lock()
_scrapling_state: dict[str, Any] = {"resolved": False, "fetcher": None}
# One shared in-flight resolution per event loop; waiters await the SAME
# future instead of each blocking an executor thread on the lock.
_scrapling_resolve_task: Any = None
_scrapling_async_guard = asyncio.Lock()


def _no_install_requested() -> bool:
    value = os.environ.get(_SCRAPLING_INSTALL_ENV, "").strip().casefold()
    return value not in ("", "0", "false", "no")


# Deliberate allowlist: the pip child sees ONLY these ambient variables plus
# the two PIP_* knobs set below. Proxy and TLS-override variables
# (HTTPS_PROXY/HTTP_PROXY/ALL_PROXY/SSL_CERT_FILE/REQUESTS_CA_BUNDLE/…) are
# deliberately excluded — an injected proxy or CA bundle could redirect or
# MITM the pinned install, and pip's --isolated does not disable proxy
# handling (verified empirically against pip 26.x). Hosts that genuinely
# need a corporate proxy should install eagerly: pip install -e ".[stealth]".
# Windows requires its system variables (Python documents subprocess failure
# without SystemRoot), so the allowlist is platform-aware.
_PIP_ENV_ALLOWLIST_POSIX = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TMP")
_PIP_ENV_ALLOWLIST_WINDOWS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "WINDIR",
    "USERPROFILE",
)


def _pip_env() -> dict[str, str]:
    allow = _PIP_ENV_ALLOWLIST_WINDOWS if sys.platform == "win32" else _PIP_ENV_ALLOWLIST_POSIX
    env = {key: os.environ[key] for key in allow if key in os.environ}
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _try_import_scrapling() -> Any:
    """Return StealthyFetcher, or None when missing OR broken.

    A partially broken install can raise things other than ImportError while
    resolving the attribute (RuntimeError, OSError, …) — treat any ordinary
    import-time failure as unavailable rather than letting it escape.
    """
    try:
        from scrapling import StealthyFetcher

        return StealthyFetcher
    except Exception:  # noqa: BLE001 — missing or broken both mean unavailable
        return None


def _interprocess_lock():
    """Acquire a best-effort inter-process lock around the pip run.

    Returns an object with a ``release()`` method, or None when locking is
    unavailable (non-fcntl platforms, unwritable temp dir). Hardened: the
    lock file is created with ``O_NOFOLLOW|O_CREAT`` and private perms so a
    pre-planted symlink/FIFO in a world-writable tmp cannot redirect or
    stall us, and a failed flock never leaks the open handle.
    """
    try:
        import fcntl
        import tempfile

        if not hasattr(os, "O_NOFOLLOW"):
            # Without O_NOFOLLOW a pre-planted symlink in a shared tmp would
            # be silently followed — degrade to the in-process lock instead.
            return None

        lock_path = os.path.join(
            tempfile.gettempdir(), f"hermes-web-crawl4ai-scrapling-{os.getuid()}.lock"
        )
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001
            os.close(fd)
            return None

        class _Released:
            def release(self) -> None:
                with contextlib.suppress(Exception):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                with contextlib.suppress(Exception):
                    os.close(fd)

        return _Released()
    except Exception:  # noqa: BLE001 — locking is best-effort
        return None


def _pip_install_scrapling() -> bool:
    """Run the one-time pinned install. Returns True on pip success."""
    import subprocess
    import sys

    argv = [
        sys.executable,
        "-I",
        "-m",
        "pip",
        "install",
        "--isolated",
        "--no-input",
        "--quiet",
        _SCRAPLING_SPEC,
    ]
    logger.info("Installing %s for stealth escalation (one-time)…", _SCRAPLING_SPEC)
    lock = _interprocess_lock()
    try:
        if _try_import_scrapling() is not None:
            # Another process completed the install while we waited on the
            # inter-process lock — skip the redundant pip run.
            return True
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SCRAPLING_INSTALL_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
            env=_pip_env(),
        )
    except FileNotFoundError:
        logger.warning("scrapling self-install failed: pip is not available in this environment")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(
            "scrapling self-install timed out after %ss", _SCRAPLING_INSTALL_TIMEOUT_SECONDS
        )
        return False
    except Exception as exc:  # noqa: BLE001 — any failure just disables the leg
        logger.warning("scrapling self-install failed: %s", exc)
        return False
    finally:
        if lock is not None:
            lock.release()
    if proc.returncode != 0:
        # Fixed classification only — pip stderr can embed index URLs and
        # credentials (even via a proxied connection) and is never logged.
        logger.warning("scrapling self-install failed (pip exit %d)", proc.returncode)
        return False
    return True


def _resolve_scrapling_fetcher():
    """Import StealthyFetcher, self-installing the pinned spec on first miss.

    The plugin owns its stealth-escalation dependency (core's lazy-deps
    allowlist is reserved for in-core backends — see core's
    ``tools/lazy_deps.py`` contract and the plugin developer guide).

    Contract: single-flight per process (thread lock; best-effort inter-
    process lock around pip itself); the outcome — including a FAILED
    install — is cached for the process lifetime; any import-time failure
    (missing or broken package) counts as unavailable. Call this from a
    worker thread (``asyncio.to_thread``) — the pip run may take minutes on
    a cold host. ``HERMES_WEB_CRAWL4AI_NO_INSTALL`` (any casing of a
    truthy value) skips the runtime install; if scrapling is already
    importable, stealth still works — the switch only forbids installing.
    """
    with _scrapling_lock:
        if _scrapling_state["resolved"]:
            return _scrapling_state["fetcher"]

        fetcher = _try_import_scrapling()
        if fetcher is None and not _no_install_requested() and _pip_install_scrapling():
            import importlib

            importlib.invalidate_caches()
            fetcher = _try_import_scrapling()
            if fetcher is None:
                logger.warning("scrapling still unimportable after a successful self-install")

        _scrapling_state["resolved"] = True
        _scrapling_state["fetcher"] = fetcher
        return fetcher


async def _resolve_scrapling_fetcher_async():
    """Awaitable single-flight resolution for the event-loop world.

    The FIRST caller runs the (possibly minutes-long) resolution in one
    worker thread; concurrent callers await the SAME future — no per-waiter
    executor threads parked on the lock. Waiters impose a deadline slightly
    above the pip timeout: if resolution somehow exceeds it, waiters get
    None (stealth unavailable this call) while the background resolution
    finishes and caches for later calls.
    """
    global _scrapling_resolve_task
    loop = asyncio.get_running_loop()
    async with _scrapling_async_guard:
        if _scrapling_state["resolved"]:
            return _scrapling_state["fetcher"]
        if _scrapling_resolve_task is None or _scrapling_resolve_task.done():
            def _resolve_and_pin() -> Any:
                try:
                    return _resolve_scrapling_fetcher()
                except Exception:  # noqa: BLE001 — never let an unexpected
                    # executor failure leave the state non-terminal: the sync
                    # resolver only fails to pin state via a bug, and a
                    # non-terminal state would re-submit resolutions forever.
                    with _scrapling_lock:
                        _scrapling_state["resolved"] = True
                        _scrapling_state["fetcher"] = None
                    return None

            try:
                _scrapling_resolve_task = loop.run_in_executor(None, _resolve_and_pin)
            except Exception:  # noqa: BLE001 — executor submission itself failed
                # (rare: loop shutting down / executor broken). Pin terminal
                # state so later calls don't re-submit forever.
                with _scrapling_lock:
                    _scrapling_state["resolved"] = True
                    _scrapling_state["fetcher"] = None
                return None
    try:
        return await asyncio.wait_for(
            asyncio.shield(_scrapling_resolve_task),
            timeout=_SCRAPLING_RESOLVE_DEADLINE_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "scrapling resolution exceeded %ss — stealth deferred this call",
            _SCRAPLING_RESOLVE_DEADLINE_SECONDS,
        )
        return None
    except Exception:  # noqa: BLE001 — resolution never breaks the caller
        return None


async def _stealth_fetch(url: str) -> dict[str, Any] | None:
    """Last-resort stealth fetch via scrapling when the content gate flags a
    challenge/block, or the exception path sees the target block/challenge us.

    StealthyFetcher is synchronous, so it runs in a worker thread; cancelling
    the caller does NOT stop the browser run (best-effort cancellation). The
    result is policy-checked against the browser's FINAL url (redirects can
    land on private networks), truncated to ``max_chars_per_page``, and logged
    host-only (query strings never enter logs). Returns None on any failure.

    Outcome vocabulary (log token ``scrapling_stealth``, grep-able):
    ``unavailable | failed | blocked_final_url | thin_content | rescued``.

    Scope note: this runs when a fetched document is classified as a
    challenge/block/empty shell by the content gate, and — from the exception
    path — when the target itself blocked/challenged us (401/403/5xx probe)
    or the probe could not classify the target at all. It never runs to mask
    a Crawl4AI outage (mapped-healthy probe + exception stays a typed error).

    Browser engines are provisioned separately — after the first install, run
    ``scrapling install`` once (see scrapling's docs) — see README."""
    started = time.monotonic()
    host = urlparse(url).hostname or "unknown"

    fetcher_cls = await _resolve_scrapling_fetcher_async()
    if fetcher_cls is None:
        logger.info(
            "scrapling_stealth host=%s outcome=unavailable duration=%.1fs",
            host,
            time.monotonic() - started,
        )
        return None

    def _run() -> tuple[str, str]:
        fetcher = fetcher_cls()
        page = fetcher.fetch(
            url,
            headless=True,
            solve_cloudflare=True,
            network_idle=True,
            wait_selector="body",
            timeout=30000,
        )
        if page is None:
            return ("", "")
        text = (page.get_all_text() if page else "") or ""
        return (text, str(getattr(page, "url", "") or url))

    try:
        text, final_url = await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001
        logger.info(
            "scrapling_stealth host=%s outcome=failed duration=%.1fs",
            host,
            time.monotonic() - started,
        )
        return None
    final_url = final_url or url
    if await _target_error(final_url):
        logger.info(
            "scrapling_stealth host=%s outcome=blocked_final_url duration=%.1fs",
            host,
            time.monotonic() - started,
        )
        return None
    text = text.strip()
    if not text or _classify_content(text):
        logger.info(
            "scrapling_stealth host=%s outcome=thin_content duration=%.1fs",
            host,
            time.monotonic() - started,
        )
        return None
    logger.info(
        "scrapling_stealth host=%s outcome=rescued duration=%.1fs",
        host,
        time.monotonic() - started,
    )
    return _truncate_document(
        {
            "url": final_url,
            "title": "",
            "content": text,
            "raw_content": text,
            "metadata": {"resolved_by": "scrapling_stealth"},
        }
    )


@lru_cache(maxsize=1)
def _plugin_config() -> dict[str, Any]:
    """Read this plugin's non-secret settings from config.yaml."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:  # noqa: BLE001 - stripped/early host contexts fail closed
        return {}
    entries = (config.get("plugins") or {}).get("entries") or {}
    entry = entries.get(PLUGIN_ID) or {}
    settings = entry.get("config") or {}
    return settings if isinstance(settings, dict) else {}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _plugin_config().get(name, default)
    try:
        return max(minimum, min(int(value), maximum))
    except (TypeError, ValueError):
        return default


def _api_url() -> str:
    """Return the Crawl4AI base URL from the plugin's config.yaml block."""
    configured = str(_plugin_config().get("api_url") or "").strip()
    return configured.rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = get_provider_env("CRAWL4AI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return {key: item for key, item in value.__dict__.items() if not key.startswith("_")}
        except Exception:  # noqa: BLE001
            pass
    return value


def _normalize_document(item: Any, fallback_url: str = "") -> dict[str, Any]:
    """Normalize one Crawl4AI response object to Hermes' extract contract."""
    plain = _to_plain(item)
    if not isinstance(plain, dict):
        return {}

    metadata = _to_plain(plain.get("metadata"))
    if not isinstance(metadata, dict):
        metadata = {}

    markdown = _to_plain(plain.get("markdown"))
    if isinstance(markdown, dict):
        content = (
            markdown.get("fit_markdown")
            or markdown.get("raw_markdown")
            or markdown.get("markdown_with_citations")
            or markdown.get("references_markdown")
            or ""
        )
        raw_content = (
            markdown.get("raw_markdown")
            or markdown.get("markdown_with_citations")
            or markdown.get("fit_markdown")
            or ""
        )
    else:
        content = markdown if isinstance(markdown, str) else ""
        raw_content = content
    content = (
        content
        or plain.get("fit_markdown")
        or plain.get("raw_markdown")
        or plain.get("extracted_content")
        or plain.get("content")
        or plain.get("cleaned_html")
        or plain.get("html")
        or ""
    )
    raw_content = (
        raw_content
        or plain.get("raw_markdown")
        or plain.get("raw_content")
        or plain.get("html")
        or content
        or ""
    )
    url = (
        plain.get("redirected_url")
        or plain.get("url")
        or metadata.get("sourceURL")
        or metadata.get("url")
        or fallback_url
    )
    title = plain.get("title") or metadata.get("title") or ""

    result: dict[str, Any] = {
        "url": str(url or ""),
        "title": str(title or ""),
        "content": str(content or ""),
        "raw_content": str(raw_content or ""),
        "metadata": metadata,
    }
    if plain.get("success") is False:
        result["content"] = ""
        result["raw_content"] = ""
        result["error"] = str(
            plain.get("error")
            or plain.get("error_message")
            or plain.get("message")
            or "Crawl4AI request failed"
        )
    elif plain.get("error") and not content:
        result["error"] = str(plain["error"])
    return result


def _normalize_documents(response: Any, fallback_url: str = "") -> list[dict[str, Any]]:
    plain = _to_plain(response)
    if not isinstance(plain, dict):
        return []

    candidates = plain.get("results")
    if not isinstance(candidates, list):
        candidates = plain.get("result")
    if not isinstance(candidates, list):
        candidates = []

    documents = [
        normalized
        for item in candidates
        if (normalized := _normalize_document(item, fallback_url=fallback_url))
    ]
    if not documents and any(
        key in plain for key in ("markdown", "html", "content", "raw_markdown")
    ):
        normalized = _normalize_document(plain, fallback_url=fallback_url)
        if normalized:
            documents.append(normalized)
    return documents


async def _post_json(path: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    api_url = _api_url()
    if not api_url:
        raise ValueError(
            "Crawl4AI is not configured. Set "
            f"plugins.entries.{PLUGIN_ID}.config.api_url in config.yaml."
        )
    parsed_url = urlparse(api_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Crawl4AI api_url must be an HTTP(S) URL with a hostname")
    if parsed_url.username or parsed_url.password:
        raise ValueError(
            "Crawl4AI api_url must not contain credentials; use CRAWL4AI_API_TOKEN"
        )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        response = await client.post(f"{api_url}{path}", json=payload, headers=_headers())
        response.raise_for_status()
        parsed = response.json()
    if not isinstance(parsed, dict):
        raise ValueError("Crawl4AI returned a non-object JSON response")
    return parsed


async def _target_error(url: str) -> str | None:
    """Return a policy error for a model/backend-provided target URL."""
    if not await async_is_safe_url(url):
        return "Blocked: URL targets a private or internal network address"
    blocked = check_website_access(url)
    if blocked:
        return blocked["message"]
    return None


def _error_document(url: str, error: str) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "raw_content": "", "error": error}


def _truncate_document(document: dict[str, Any]) -> dict[str, Any]:
    limit = _bounded_int("max_chars_per_page", DEFAULT_MAX_CHARS, 2_000, 200_000)
    result = dict(document)
    content = str(result.get("content") or "")
    if len(content) > limit:
        result["content"] = content[:limit] + f"\n\n[Truncated after {limit} characters]"
    raw = str(result.get("raw_content") or "")
    if len(raw) > limit:
        result["raw_content"] = raw[:limit]
    return result


async def _secure_document(document: dict[str, Any], fallback_url: str) -> dict[str, Any]:
    url = str(document.get("url") or fallback_url)
    error = await _target_error(url)
    if error:
        return _error_document(url, error)
    document["url"] = url
    return _truncate_document(document)


def _is_same_site(seed_url: str, candidate_url: str) -> bool:
    """Accept the seed host and its subdomains; reject unrelated crawl results."""
    seed_host = (urlparse(seed_url).hostname or "").lower().rstrip(".")
    candidate_host = (urlparse(candidate_url).hostname or "").lower().rstrip(".")
    if not seed_host or not candidate_host:
        return False
    return candidate_host == seed_host or candidate_host.endswith(f".{seed_host}")


async def _crawl4ai_fetch(url: str, *, format: str | None = None) -> dict[str, Any]:
    """Fetch one URL through Crawl4AI's flat rendered-content endpoints."""
    timeout = _bounded_int("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, 10, 300)
    if format == "html":
        raw = await _post_json("/html", {"url": url}, timeout=timeout)
        document = _normalize_document(raw, fallback_url=url)
    else:
        raw = await _post_json("/md", {"url": url, "f": "fit"}, timeout=timeout)
        document = _normalize_document(raw, fallback_url=url)
        if not document.get("content") and not document.get("error"):
            raw = await _post_json("/md", {"url": url, "f": "raw"}, timeout=timeout)
            document = _normalize_document(raw, fallback_url=url)
    if not document:
        return _error_document(url, "Crawl4AI returned no document")
    return await _secure_document(document, url)


async def _fetch_document(url: str, *, format: str | None = None) -> dict[str, Any]:
    """Fetch one URL with a layered strategy and typed failure classification:

      1. obvious-PDF extension → Hermes' document extractor (read_extract)
      2. best-effort HEAD probe → PDF-by-content-type + early-out on 404/410
      3. Crawl4AI /md (fit → raw) or /html
      4. content-quality gate → typed error, with a scrapling stealth
         escalation for challenge/blocked walls
    """
    if _url_is_pdf(url):
        return await _extract_pdf(url)

    probe_status, content_type = await _probe_status(url)

    if (content_type or "").lower().startswith("application/pdf"):
        return await _extract_pdf(url)

    if probe_status in (404, 410):
        return _typed_error_document(
            url, "not_found", f"URL returned HTTP {probe_status} (not found)"
        )

    try:
        document = await _crawl4ai_fetch(url, format=format)
    except Exception as exc:  # noqa: BLE001 — Crawl4AI 500 / connection / timeout
        if probe_status:
            error_type = _status_to_error_type(probe_status) or "crawl4ai_error"
            message = f"Target returned HTTP {probe_status}"
        else:
            error_type = "crawl4ai_error"
            message = f"Crawl4AI extract failed: {exc}"
        # Escalate only when the TARGET is the suspect (it blocked/challenged
        # us) or the probe could not classify it at all. A mapped-healthy
        # probe (e.g. 200) with Crawl4AI erroring means our infrastructure —
        # never mask that outage with a doomed browser run.
        if probe_status is None or error_type in _STEALTHABLE_ERROR_TYPES:
            stealth = await _stealth_fetch(url)
            if stealth:
                return stealth
        return _typed_error_document(url, error_type, message)

    if document.get("error"):
        # Already a typed per-URL failure (policy block, success=False, etc.).
        return document

    gate = _classify_content(document.get("content") or "")
    if not gate:
        return document

    if gate in ("challenge", "blocked", "empty"):
        # JS-heavy pages can render as an "empty" shell over plain HTTP; a
        # real browser is the only way to render them. rate_limited stays
        # excluded: the browser shares the same IP and limit.
        stealth = await _stealth_fetch(url)
        if stealth:
            return stealth

    return _typed_error_document(
        url, gate, _GATE_MESSAGES.get(gate, "Page content failed quality checks")
    )


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Extract-only provider intended to pair with Hermes' bundled SearXNG plugin."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        return bool(_api_url())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        format_value = kwargs.get("format")
        concurrency = _bounded_int("concurrency", DEFAULT_CONCURRENCY, 1, 10)
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_one(url: str) -> dict[str, Any]:
            try:
                normalized_url = normalize_url_for_request(url)
                error = await _target_error(normalized_url)
                if error:
                    return _error_document(normalized_url, error)
                async with semaphore:
                    return await _fetch_document(normalized_url, format=format_value)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed URL must not fail the batch
                safe_url = locals().get("normalized_url", str(url))
                logger.warning("Crawl4AI extract failed for %s: %s", safe_url, exc)
                return _error_document(safe_url, f"Crawl4AI extract failed: {exc}")

        return list(await asyncio.gather(*(extract_one(url) for url in urls)))

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "free · self-hosted",
            "tag": "Extract backend; pair with bundled SearXNG for a local OSS stack.",
            "env_vars": [],
        }


async def crawl4ai_crawl_tool(args: dict[str, Any], **_: Any) -> str:
    """Run a bounded same-site crawl by walking links client-side.

    Crawl4AI 0.9.x rejects ``deep_crawl_strategy`` from untrusted (network)
    request bodies, so deep crawling is implemented here as a client-side BFS
    over the flat ``/md`` endpoint instead of a server-side deep-crawl strategy.
    """
    try:
        import re

        url = normalize_url_for_request(str(args.get("url") or "").strip())
        if not url:
            return json.dumps({"success": False, "error": "A URL is required"})
        error = await _target_error(url)
        if error:
            return json.dumps({"success": False, "error": error, "url": url})

        max_pages = max(1, min(int(args.get("max_pages", 10)), MAX_CRAWL_PAGES))
        max_depth = 2 if args.get("depth") == "advanced" else 1
        timeout = _bounded_int("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, 10, 300)

        link_re = re.compile(r"https?://[^\s<>\"')\]]+")

        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(url, 0)]
        documents: list[dict[str, Any]] = []

        while queue and len(documents) < max_pages:
            current, depth = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)

            try:
                raw = await _post_json("/md", {"url": current, "f": "fit"}, timeout=timeout)
                document = _normalize_document(raw, fallback_url=current)
                if not document.get("content") and not document.get("error"):
                    raw = await _post_json("/md", {"url": current, "f": "raw"}, timeout=timeout)
                    document = _normalize_document(raw, fallback_url=current)
            except Exception as exc:  # noqa: BLE001 - one failed page must not kill the crawl
                document = _error_document(current, f"Crawl4AI fetch failed: {exc}")

            documents.append(await _secure_document(document, current))

            if depth < max_depth:
                text = str(document.get("content") or document.get("raw_content") or "")
                for match in link_re.finditer(text):
                    candidate = match.group(0).rstrip(".,;!?)")
                    if candidate not in seen and _is_same_site(url, candidate):
                        queue.append((candidate, depth + 1))

        return json.dumps({"results": documents}, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - tool errors are returned to the model
        safe_url = locals().get("url", str(args.get("url") or ""))
        logger.warning("Crawl4AI crawl failed for %s: %s", safe_url, exc)
        return json.dumps(
            {"success": False, "error": f"Crawl4AI crawl failed: {exc}", "url": safe_url},
            ensure_ascii=False,
        )
