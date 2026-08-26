from __future__ import annotations

import json
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
