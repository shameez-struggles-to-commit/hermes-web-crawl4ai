from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest

from hermes_web_crawl4ai import provider as provider_module
from hermes_web_crawl4ai.provider import (
    Crawl4AIWebSearchProvider,
    _normalize_document,
    _normalize_documents,
    crawl4ai_crawl_tool,
)


def test_normalize_document_prefers_fit_markdown() -> None:
    result = _normalize_document(
        {
            "url": "https://example.com/docs",
            "metadata": {"title": "Docs"},
            "markdown": {
                "fit_markdown": "Fit markdown",
                "raw_markdown": "Raw markdown",
            },
            "success": True,
        }
    )

    assert result["url"] == "https://example.com/docs"
    assert result["title"] == "Docs"
    assert result["content"] == "Fit markdown"
    assert result["raw_content"] == "Raw markdown"


@pytest.mark.parametrize("key", ["results", "result"])
def test_normalize_documents_accepts_known_list_envelopes(key: str) -> None:
    documents = _normalize_documents(
        {key: [{"url": "https://example.com", "markdown": "hello", "success": True}]}
    )
    assert documents == [
        {
            "url": "https://example.com",
            "title": "",
            "content": "hello",
            "raw_content": "hello",
            "metadata": {},
        }
    ]


def test_provider_is_extract_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "_api_url", lambda: "http://localhost:11235")
    provider = Crawl4AIWebSearchProvider()

    assert provider.name == "crawl4ai"
    assert provider.supports_search() is False
    assert provider.supports_extract() is True
    assert provider.is_available() is True


def test_api_url_reads_non_secret_setting_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "_plugin_config",
        lambda: {"api_url": "http://configured.example:11235/"},
    )
    assert provider_module._api_url() == "http://configured.example:11235"

    monkeypatch.setattr(provider_module, "_plugin_config", lambda: {})
    assert provider_module._api_url() == ""


def test_normalize_document_prefers_redirect_destination_for_policy_checks() -> None:
    result = _normalize_document(
        {
            "url": "https://safe.example/start",
            "redirected_url": "http://127.0.0.1/private",
            "markdown": "content",
            "success": True,
        }
    )

    assert result["url"] == "http://127.0.0.1/private"


@pytest.mark.asyncio
async def test_extract_preserves_order_and_returns_per_url_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = Crawl4AIWebSearchProvider()
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    monkeypatch.setattr(
        provider_module,
        "_fetch_document",
        AsyncMock(
            side_effect=[
                {
                    "url": "https://one.example",
                    "title": "One",
                    "content": "first",
                    "raw_content": "first",
                    "metadata": {},
                },
                RuntimeError("boom"),
            ]
        ),
    )

    results = await provider.extract(["https://one.example", "https://two.example"])

    assert [item["url"] for item in results] == [
        "https://one.example",
        "https://two.example",
    ]
    assert results[0]["content"] == "first"
    assert "boom" in results[1]["error"]


@pytest.mark.asyncio
async def test_extract_contains_policy_check_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Crawl4AIWebSearchProvider()
    monkeypatch.setattr(
        provider_module,
        "_target_error",
        AsyncMock(side_effect=RuntimeError("DNS failed")),
    )

    results = await provider.extract(["https://example.com"])

    assert results[0]["url"] == "https://example.com"
    assert "DNS failed" in results[0]["error"]


@pytest.mark.asyncio
async def test_crawl_tool_rejects_unsafe_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module,
        "_target_error",
        AsyncMock(return_value="Blocked: URL targets a private or internal network address"),
    )
    post = AsyncMock()
    monkeypatch.setattr(provider_module, "_post_json", post)

    result = json.loads(await crawl4ai_crawl_tool({"url": "http://127.0.0.1"}))

    assert result["success"] is False
    assert "private or internal" in result["error"]
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_crawl_tool_clamps_pages_and_walks_same_site_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    links = "\n".join(f"https://example.com/docs/{index}" for index in range(25))
    pages: dict[str, dict] = {
        "https://example.com/docs": {
            "url": "https://example.com/docs",
            "markdown": f"seed\n{links}",
            "success": True,
        },
        **{
            f"https://example.com/docs/{index}": {
                "url": f"https://example.com/docs/{index}",
                "markdown": f"page {index}",
                "success": True,
            }
            for index in range(25)
        },
    }
    calls: list[dict] = []

    async def post(path: str, payload: dict, timeout: int) -> dict:
        assert path == "/md"
        calls.append(dict(payload))
        return pages[payload["url"]]

    monkeypatch.setattr(provider_module, "_post_json", post)

    result = json.loads(
        await crawl4ai_crawl_tool({"url": "https://example.com/docs", "max_pages": 999})
    )

    # Client-side BFS over the flat /md endpoint — never the old /crawl payload.
    assert calls and all(
        "browser_config" not in payload and payload.get("f") == "fit" for payload in calls
    )
    # MAX_CRAWL_PAGES clamps the requested 999 to 20 fetched documents.
    assert len(result["results"]) == 20
    assert result["results"][0]["url"] == "https://example.com/docs"


@pytest.mark.asyncio
async def test_crawl_tool_rejects_cross_site_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    pages: dict[str, dict] = {
        "https://example.com": {
            "url": "https://example.com",
            "markdown": "seed https://example.com/a and https://unrelated.example/b",
            "success": True,
        },
        "https://example.com/a": {
            "url": "https://example.com/a",
            "markdown": "same-site child",
            "success": True,
        },
        "https://unrelated.example/b": {
            "url": "https://unrelated.example/b",
            "markdown": "must never be fetched",
            "success": True,
        },
    }
    calls: list[str] = []

    async def post(path: str, payload: dict, timeout: int) -> dict:
        calls.append(payload["url"])
        return pages[payload["url"]]

    monkeypatch.setattr(provider_module, "_post_json", post)

    result = json.loads(await crawl4ai_crawl_tool({"url": "https://example.com"}))

    # Cross-site links are filtered before fetching (queue-level _is_same_site).
    assert "https://unrelated.example/b" not in calls
    assert calls == ["https://example.com", "https://example.com/a"]
    assert [doc["url"] for doc in result["results"]] == [
        "https://example.com",
        "https://example.com/a",
    ]


# ---------------------------------------------------------------------------
# Stealth escalation (exception path + content gate) — plan v3
# ---------------------------------------------------------------------------


def _escalation_doc(url: str = "https://blocked.example/page") -> dict:
    return {
        "url": url,
        "title": "",
        "content": "rescued content " * 20,
        "raw_content": "rescued content " * 20,
        "metadata": {"resolved_by": "scrapling_stealth"},
    }


async def _stub_probe(monkeypatch: pytest.MonkeyPatch, status, content_type=None):
    monkeypatch.setattr(
        provider_module, "_probe_status", AsyncMock(return_value=(status, content_type))
    )


async def _run_fetch_document(url: str = "https://blocked.example/page", **kwargs) -> dict:
    return await provider_module._fetch_document(url, **kwargs)


@pytest.mark.asyncio
async def test_exception_probe_403_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, 403)
    boom = Exception("Internal server error")
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=boom)
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_awaited_once_with("https://blocked.example/page")
    assert result["metadata"]["resolved_by"] == "scrapling_stealth"
    assert "error" not in result


@pytest.mark.asyncio
async def test_exception_probe_503_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cloudflare's managed challenge is served as HTTP 503 → upstream_error.
    await _stub_probe(monkeypatch, 503)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("500"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_awaited_once()
    assert result["metadata"]["resolved_by"] == "scrapling_stealth"


@pytest.mark.asyncio
async def test_exception_probe_none_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, None)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("boom"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_awaited_once()
    assert result["metadata"]["resolved_by"] == "scrapling_stealth"


@pytest.mark.asyncio
async def test_exception_probe_429_no_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, 429)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("500"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["error_type"] == "rate_limited"


@pytest.mark.asyncio
async def test_exception_probe_200_no_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Healthy target + Crawl4AI exception = OUR infra is suspect: never mask
    # the outage with a doomed browser run.
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("refused"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["error_type"] == "crawl4ai_error"


@pytest.mark.asyncio
async def test_exception_probe_400_no_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, 400)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("500"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["error_type"] == "crawl4ai_error"


@pytest.mark.asyncio
async def test_exception_probe_405_no_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, 405)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("500"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["error_type"] == "crawl4ai_error"


@pytest.mark.asyncio
async def test_exception_stealth_none_preserves_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stub_probe(monkeypatch, 403)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("boom"))
    )
    monkeypatch.setattr(provider_module, "_stealth_fetch", AsyncMock(return_value=None))

    result = await _run_fetch_document()

    assert result["error_type"] == "blocked"
    assert result["error"] == "Target returned HTTP 403"


@pytest.mark.asyncio
async def test_gate_challenge_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module,
        "_crawl4ai_fetch",
        AsyncMock(
            return_value={
                "url": "https://blocked.example/page",
                "title": "",
                "content": "Just a moment...",
                "raw_content": "",
                "metadata": {},
            }
        ),
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_awaited_once()
    assert result["metadata"]["resolved_by"] == "scrapling_stealth"


@pytest.mark.asyncio
async def test_gate_empty_escalates_and_falls_back_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module,
        "_crawl4ai_fetch",
        AsyncMock(
            return_value={
                "url": "https://blocked.example/page",
                "title": "",
                "content": "tiny",
                "raw_content": "",
                "metadata": {},
            }
        ),
    )
    stealth = AsyncMock(return_value=None)
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_awaited_once()
    assert result["error_type"] == "empty"


@pytest.mark.asyncio
async def test_gate_rate_limited_content_no_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module,
        "_crawl4ai_fetch",
        AsyncMock(
            return_value={
                "url": "https://blocked.example/page",
                "title": "",
                "content": "Too many requests — rate limit exceeded. Try again soon.",
                "raw_content": "",
                "metadata": {},
            }
        ),
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["error_type"] == "rate_limited"


@pytest.mark.asyncio
async def test_gate_clean_content_no_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module,
        "_crawl4ai_fetch",
        AsyncMock(
            return_value={
                "url": "https://ok.example/page",
                "title": "Fine",
                "content": "x" * 500,
                "raw_content": "x" * 500,
                "metadata": {},
            }
        ),
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document()

    stealth.assert_not_awaited()
    assert result["content"] == "x" * 500


# --- _stealth_fetch internals: real _stealth_fetch, fake fetcher seam ---


class _FakePage:
    def __init__(self, text: str, url: str) -> None:
        self._text = text
        self.url = url

    def get_all_text(self) -> str:
        return self._text


def _install_fake_fetcher(monkeypatch: pytest.MonkeyPatch, page) -> None:
    class _FakeFetcher:
        def fetch(self, url: str, **kwargs):
            assert kwargs.get("solve_cloudflare") is True
            return page

    monkeypatch.setattr(
        provider_module,
        "_resolve_scrapling_fetcher_async",
        AsyncMock(return_value=_FakeFetcher),
    )
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_stealth_result_truncated_to_max_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_text = "word " * 6000  # ~30k chars > 20k default cap
    _install_fake_fetcher(
        monkeypatch, _FakePage(long_text, "https://blocked.example/page")
    )

    result = await provider_module._stealth_fetch("https://blocked.example/page")

    assert result is not None
    assert len(result["content"]) < 30_000
    assert result["content"].endswith("[Truncated after 20000 characters]")
    assert result["metadata"]["resolved_by"] == "scrapling_stealth"


@pytest.mark.asyncio
async def test_stealth_rejects_private_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fetcher(
        monkeypatch,
        _FakePage("plenty of real looking content here " * 10, "http://127.0.0.1:9/x"),
    )

    async def policy(url: str):
        if "127.0.0.1" in url:
            return "Blocked: URL targets a private or internal network address"
        return None

    spy = AsyncMock(side_effect=policy)
    monkeypatch.setattr(provider_module, "_target_error", spy)

    result = await provider_module._stealth_fetch("https://public.example/redirect")

    assert result is None
    # The browser's FINAL url (private) was policy-checked, not just the original.
    assert any("127.0.0.1" in str(call.args[0]) for call in spy.await_args_list)


@pytest.mark.asyncio
async def test_stealth_logs_host_only_no_query_leak(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    _install_fake_fetcher(
        monkeypatch,
        _FakePage(
            "legit content " * 40,
            "https://ok.example/page?token=sekrit&next=/admin",
        ),
    )

    with caplog.at_level(logging.INFO, logger="hermes_web_crawl4ai.provider"):
        result = await provider_module._stealth_fetch(
            "https://ok.example/page?token=sekrit"
        )

    assert result is not None
    # The ORIGINAL url is returned; the final redirect url (query-stripped)
    # lives in metadata — never a credential-bearing URL in the doc body.
    assert result["url"] == "https://ok.example/page?token=sekrit"
    assert result["metadata"]["final_url"] == "https://ok.example/page"
    joined = " | ".join(rec.message for rec in caplog.records)
    assert "outcome=rescued" in joined
    assert "host=ok.example" in joined
    assert "sekrit" not in joined
    assert "token=" not in joined
    assert "token=" not in result["metadata"]["final_url"]


@pytest.mark.asyncio
async def test_unavailable_resolution_logged(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(
        provider_module,
        "_resolve_scrapling_fetcher_async",
        AsyncMock(return_value=None),
    )

    with caplog.at_level(logging.INFO, logger="hermes_web_crawl4ai.provider"):
        result = await provider_module._stealth_fetch("https://blocked.example/x")

    assert result is None
    assert any("outcome=unavailable" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_crawl_tool_does_not_escalate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Documents the exclusion: the plugin-local crawl tool never escalates.
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    monkeypatch.setattr(
        provider_module, "_post_json", AsyncMock(side_effect=Exception("down"))
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = json.loads(await crawl4ai_crawl_tool({"url": "https://example.com"}))

    # Per-page handler converts failures to error documents; no escalation.
    stealth.assert_not_awaited()
    assert len(result["results"]) == 1
    assert result["results"][0]["content"] == ""
    assert "Crawl4AI fetch failed" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_extract_cancelled_during_stealth_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    await _stub_probe(monkeypatch, 403)
    monkeypatch.setattr(
        provider_module, "_crawl4ai_fetch", AsyncMock(side_effect=Exception("500"))
    )

    slow_started = asyncio.Event()
    release = asyncio.Event()

    async def slow_stealth(url: str):
        slow_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            raise
        return _escalation_doc()

    monkeypatch.setattr(provider_module, "_stealth_fetch", slow_stealth)

    provider = Crawl4AIWebSearchProvider()
    monkeypatch.setattr(provider_module, "_api_url", lambda: "http://localhost:11235")

    task = asyncio.create_task(provider.extract(["https://blocked.example/page"]))
    await asyncio.wait_for(slow_started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()


@pytest.mark.asyncio
async def test_html_format_escalation_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _stub_probe(monkeypatch, 200)
    monkeypatch.setattr(
        provider_module,
        "_crawl4ai_fetch",
        AsyncMock(
            return_value={
                "url": "https://blocked.example/page",
                "title": "",
                "content": "Just a moment...",
                "raw_content": "",
                "metadata": {},
            }
        ),
    )
    stealth = AsyncMock(return_value=_escalation_doc())
    monkeypatch.setattr(provider_module, "_stealth_fetch", stealth)

    result = await _run_fetch_document(format="html")

    stealth.assert_awaited_once()
    # Documented limitation: stealth always returns text, never HTML.
    assert result["content"] == "rescued content " * 20


# ---------------------------------------------------------------------------
# Round-2 review fixes: cross-logger leaks, URL hygiene, crawl output bounds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stealth_silences_scrapling_logger(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    class _NoisyFetcher:
        def fetch(self, url: str, **kwargs):
            logging.getLogger("scrapling").info(
                "Fetched (200) <GET %s?token=sekrit>", url
            )
            return _FakePage("legit content " * 40, "https://ok.example/page")

    monkeypatch.setattr(
        provider_module,
        "_resolve_scrapling_fetcher_async",
        AsyncMock(return_value=_NoisyFetcher),
    )
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))

    prior_level = logging.getLogger("scrapling").level
    with caplog.at_level(logging.INFO, logger="scrapling"):
        result = await provider_module._stealth_fetch("https://ok.example/page")

    assert result is not None
    assert not any("sekrit" in rec.getMessage() for rec in caplog.records)
    # Level restored to whatever it was before the fetch (INFO when scrapling
    # is loaded for real; NOTSET in this offline test environment).
    assert logging.getLogger("scrapling").level == prior_level


@pytest.mark.asyncio
async def test_stealth_policy_check_uses_query_stripped_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fetcher(
        monkeypatch,
        _FakePage("legit content " * 40, "https://ok.example/page?token=sekrit"),
    )
    seen: list[str] = []

    async def policy(url: str):
        seen.append(url)
        return None

    monkeypatch.setattr(provider_module, "_target_error", policy)

    result = await provider_module._stealth_fetch("https://ok.example/page")

    assert result is not None
    # The policy layer saw the final URL WITHOUT the query string — a
    # downstream logger there can never print ?token=sekrit.
    assert seen == ["https://ok.example/page"]


@pytest.mark.asyncio
async def test_stealth_returns_original_url_not_credential_bearing_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fetcher(
        monkeypatch,
        _FakePage(
            "legit content " * 40,
            "https://ok.example/redirected?token=opaque-secret&next=/admin",
        ),
    )
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))

    result = await provider_module._stealth_fetch("https://ok.example/page")

    assert result is not None
    assert result["url"] == "https://ok.example/page"
    # Query-stripped final URL preserved for traceability.
    assert result["metadata"]["final_url"] == "https://ok.example/redirected"
    assert "opaque-secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_crawl_tool_rejects_returned_document_that_left_seed_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    monkeypatch.setattr(
        provider_module,
        "_post_json",
        AsyncMock(
            return_value={
                "url": "https://unrelated.example/page",
                "markdown": "content from an unrelated host",
                "success": True,
            }
        ),
    )

    result = json.loads(await crawl4ai_crawl_tool({"url": "https://example.com"}))

    assert result["results"][0]["content"] == ""
    assert "left the seed site" in result["results"][0]["error"]


@pytest.mark.asyncio
async def test_crawl_tool_advanced_depth_walks_two_levels_not_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_target_error", AsyncMock(return_value=None))
    pages: dict[str, dict] = {
        "https://example.com/root": {
            "url": "https://example.com/root",
            "markdown": "root https://example.com/l1",
            "success": True,
        },
        "https://example.com/l1": {
            "url": "https://example.com/l1",
            "markdown": "l1 https://example.com/l2",
            "success": True,
        },
        "https://example.com/l2": {
            "url": "https://example.com/l2",
            "markdown": "l2 https://example.com/l3",
            "success": True,
        },
        "https://example.com/l3": {
            "url": "https://example.com/l3",
            "markdown": "must never be fetched in advanced mode",
            "success": True,
        },
    }
    calls: list[str] = []

    async def post(path: str, payload: dict, timeout: int) -> dict:
        calls.append(payload["url"])
        return pages[payload["url"]]

    monkeypatch.setattr(provider_module, "_post_json", post)

    result = json.loads(
        await crawl4ai_crawl_tool(
            {"url": "https://example.com/root", "depth": "advanced"}
        )
    )

    assert calls == [
        "https://example.com/root",
        "https://example.com/l1",
        "https://example.com/l2",
    ]
    assert len(result["results"]) == 3
