# Hermes Web Crawl4AI

Standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin
for private, self-hosted web extraction with Crawl4AI.

This is the standalone continuation of the Crawl4AI work from
[NousResearch/hermes-agent#6325](https://github.com/NousResearch/hermes-agent/pull/6325).
It intentionally makes no changes to Hermes core and is not listed in a Hermes
plugin registry while compatibility testing is underway.

### Stealth escalation (optional)

When extraction hits a bot wall, the plugin falls back to an in-process
stealth fetch via [scrapling]. Stealth engages in two places:

1. **Exception path** — Crawl4AI errored AND the target looks like the
   suspect: the probe saw the target block/challenge us (HTTP 401/403/5xx —
   Cloudflare's managed challenge is served as 503), or the probe could not
   classify the target at all. A mapped-healthy probe (e.g. 200) with
   Crawl4AI erroring is treated as OUR infrastructure failing and is
   returned as a typed error — stealth never masks a Crawl4AI outage.
2. **Content gate** — Crawl4AI "succeeded" but the content is a
   challenge/block wall or an empty shell (common for JS-heavy pages a
   plain HTTP fetch renders thin).

Every stealth result is policy-checked against the browser's *final* URL
(query-stripped — matching is host-based) and truncated to
`max_chars_per_page`. Logs are host-only with a grep-able
`scrapling_stealth` token and outcome (`unavailable | failed |
blocked_final_url | thin_content | rescued`); scrapling's own INFO logging
(full request URLs) is silenced during the fetch. The returned document
carries the *original* URL — redirect targets can carry credential-like
query parameters. The final URL (query-stripped) is kept in
`metadata.final_url`.

**SSRF scope, stated honestly:** the final-URL check is *output filtering*
— content a redirect delivered from an unsafe host is never returned. It
does NOT block the request at redirect time: intermediate hops,
subresources, and scripts still load wherever the target sends them.
Request-time blocking is not exposed by scrapling's public fetch API and
remains future work.

Known limits, stated plainly:

- **Latency:** stealth runs take seconds to minutes (measured: 1.9s for
  example.com, 18.8s for a ProductHunt turnstile solve, 207s for an IMDb
  title page). The 30s fetch timeout is per wait-stage, not wall-clock.
- **No cross-call resource bound:** the `concurrency` setting limits URLs
  within one `extract()` call only; concurrent calls (and other Hermes
  processes) can each run their own stealth fetches.
- **Cancellation is best-effort:** cancelling a request does not stop the
  browser run; it finishes its current stages in the background.
- **Text-only:** `format=html` requests that escalate receive plain text
  (scrapling's `get_all_text()`), never HTML.
- The plugin-local `crawl4ai_crawl` tool never escalates to stealth.

The pinned dependency
`scrapling[fetchers]==0.4.14` self-installs on first use (set
`HERMES_WEB_CRAWL4AI_NO_INSTALL=1` to forbid runtime installs — note this
only disables *installing*: if scrapling is already importable, stealth
still runs). The self-install deliberately runs pip in a sanitized,
proxy-free environment (no `HTTPS_PROXY`/`SSL_CERT_FILE`/etc. — an
injected proxy could otherwise redirect the install). Hosts behind a
corporate proxy should install eagerly instead:

    pip install -e ".[stealth]"

After the first install, provision the browser engines once with
`scrapling install` (scrapling's installer drives the Playwright CLI to
fetch chromium; the stealth fetcher itself runs on patchright, which
shares Playwright's browser cache).

[scrapling]: https://scrapling.readthedocs.io/

## What it provides

- `crawl4ai`, an extract-only `WebSearchProvider` used by Hermes' existing
  `web_extract` tool.
- `crawl4ai_crawl`, an optional plugin-local tool for bounded deep crawling.
- Compatibility with Crawl4AI's flat `/md` and `/html` endpoints and their
  response envelopes.
- Hermes URL-safety and website-policy checks on requested and returned URLs.

Hermes already bundles a SearXNG search provider, so this plugin does not copy
or replace it. Pair the two through Hermes' per-capability configuration.

## Status

Alpha and intended for direct testing. Do not depend on the current API as a
stable release yet. The initial target is the security-hardened Crawl4AI
v0.8.9 REST service.

Validated locally on 2026-07-12 against the pinned Crawl4AI 0.8.9 container:
real `/md` and `/html` requests pass; crawling runs client-side over the
flat `/md` endpoint. Broader deployment testing is still wanted before
registry or catalog promotion.

## Install for testing

```bash
hermes plugins install axAilotl/hermes-web-crawl4ai --enable
```

The plugin can also be cloned directly to:

```text
~/.hermes/plugins/hermes-web-crawl4ai/
```

Then enable it:

```bash
hermes plugins enable hermes-web-crawl4ai
```

## Configure

Put the non-secret Crawl4AI service URL in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-web-crawl4ai
  entries:
    hermes-web-crawl4ai:
      config:
        api_url: http://localhost:11235
        timeout_seconds: 120
        concurrency: 4
        max_chars_per_page: 20000

web:
  search_backend: searxng
  extract_backend: crawl4ai
```

Configure Hermes' bundled SearXNG provider through `hermes tools` as usual.

If the Crawl4AI instance requires authentication, put only its secret token in
`~/.hermes/.env`:

```dotenv
CRAWL4AI_API_TOKEN=replace-me
```

Restart Hermes after enabling or changing the plugin. Run `hermes tools` and
ensure the standard `web` toolset is enabled. Enable the `crawl4ai` plugin
toolset too if you want the model to see `crawl4ai_crawl`.

## Behavior

For known URLs, Hermes continues to call its stable core tool:

```text
web_extract -> Crawl4AIWebSearchProvider.extract()
```

The plugin-local `crawl4ai_crawl` tool is reserved for cases where multiple
linked pages under one site are needed. Its limits are deliberate:

- `basic`: maximum link depth 1
- `advanced`: maximum link depth 2
- at most 20 pages per call
- links are walked client-side (same-site only, absolute and `href=`-relative);
  every discovered URL is policy-preflighted before it is fetched, and
  returned documents whose URL left the seed host/subdomain are rejected
- seed URLs carrying credential-like query parameters (e.g. `?token=…`) are
  refused, matching Hermes core's `web_extract` input guard
- configurable per-page character cap, maximum 200,000
- public HTTP(S) targets only unless the user's Hermes URL-safety policy allows otherwise

Hermes validates the seed and returned URLs, while Crawl4AI performs the
actual fetches. Use a current security-patched Crawl4AI release: post-response
filtering cannot undo a request already made by an outdated or misconfigured
Crawl4AI server.

## Local development

Use a current Hermes checkout as the host contract:

```bash
git clone https://github.com/NousResearch/hermes-agent.git ../hermes-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ../hermes-agent
pip install -e '.[dev]'
pytest -q
ruff check .
```

The test suite includes a real Hermes `PluginManager` discovery test, not just
mocked registration. `tests/test_live_crawl4ai.py` is an opt-in real-service
suite; set `CRAWL4AI_TEST_URL` to run it. The GitHub workflow also exposes a
manual containerized live-test job.

Live service validation is intentionally manual during the alpha period. Point
`api_url` at a disposable Crawl4AI instance, then verify:

1. `web_extract` returns markdown for one and several public URLs.
2. HTML extraction works when Hermes requests `format=html`.
3. A failed URL returns a per-URL error without failing the whole batch.
4. `crawl4ai_crawl` respects depth and page limits.
5. Redirected private/internal URLs are rejected.

## Distribution policy

This repository is the distribution unit. It will not be proposed for the
Hermes core tree or any plugin catalog until live compatibility and security
testing are complete.

## License

MIT
