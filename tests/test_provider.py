import json

import httpx
import pytest

from adaptive_rag.models import RouteDecision
from adaptive_rag.providers import OllamaCloud, ProviderError


def test_cloud_json_repair_and_auth():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.host == "ollama.com"
        assert request.headers["authorization"] == "Bearer test-key"
        assert "format" not in requests[-1]
        content = "not JSON" if len(requests) == 1 else '```json\n{"route":"web","reason":"unrelated"}\n```'
        return httpx.Response(200, json={"message": {"content": content}})

    provider = OllamaCloud("test-key", transport=httpx.MockTransport(handler))
    try:
        assert provider.decide("route", "route question", {}, RouteDecision).route == "web"
        assert len(requests) == 2
    finally:
        provider.close()


def test_invalid_decisions_are_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {"content": '{"route":"bogus","reason":"bad"}'}})

    provider = OllamaCloud("test-key", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderError, match="after two attempts"):
            provider.decide("route", "", {}, RouteDecision)
        assert len(calls) == 2
    finally:
        provider.close()


def test_http_errors_do_not_expose_key_or_server_body():
    provider = OllamaCloud("test-secret", transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "sensitive body"})))
    try:
        with pytest.raises(ProviderError) as error:
            provider.search("query")
        assert str(error.value) == "Ollama /api/web_search: HTTP 401"
    finally:
        provider.close()


def test_search_parses_and_filters_results():
    provider = OllamaCloud("test-key", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": [
            {"title": "Valid", "url": "https://example.org", "content": "Evidence"},
            {"url": "javascript:alert(1)", "content": "invalid"},
            {"url": "https://empty.org", "content": ""},
        ]})))
    try:
        docs = provider.search("query")
        assert len(docs) == 1 and docs[0]["kind"] == "web"
        assert provider.search("query")[0]["id"] == docs[0]["id"]
    finally:
        provider.close()
