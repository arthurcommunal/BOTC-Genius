"""WikiClient tests: the network is replaced by httpx.MockTransport, no real request is sent."""

import httpx
import pytest

from botc.corpus.wiki_client import WikiApiError, WikiBlockedError, WikiClient


def make_client(handler, max_retries: int = 2) -> WikiClient:
    return WikiClient(api_url="https://wiki.test/api.php", min_interval=0, max_retries=max_retries,
                      transport=httpx.MockTransport(handler))


def test_sends_json_format_and_user_agent():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"query": {}})

    make_client(handler).get(action="query", meta="siteinfo")
    params = seen[0].url.params
    assert params["format"] == "json"
    assert params["formatversion"] == "2"
    assert params["meta"] == "siteinfo"
    assert "maxlag" in params
    assert "botc" in seen[0].headers["User-Agent"]


def test_retries_on_503_then_succeeds():
    answers = iter([httpx.Response(503, headers={"Retry-After": "0"}), httpx.Response(200, json={"ok": True})])
    assert make_client(lambda request: next(answers)).get(action="query") == {"ok": True}


def test_retries_on_maxlag_error():
    answers = iter([httpx.Response(200, headers={"Retry-After": "0"}, json={"error": {"code": "maxlag"}}),
                    httpx.Response(200, json={"ok": True})])
    assert make_client(lambda request: next(answers)).get(action="query") == {"ok": True}


def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "0"})

    with pytest.raises(WikiApiError):
        make_client(handler, max_retries=2).get(action="query")
    assert len(calls) == 3  # the first try + 2 retries


@pytest.mark.parametrize("status", [403, 418])
def test_stops_immediately_when_blocked(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status)

    with pytest.raises(WikiBlockedError):
        make_client(handler).get(action="query")
    assert len(calls) == 1  # no retry: we do not insist


def test_api_error_is_raised():
    handler = lambda request: httpx.Response(200, json={"error": {"code": "missingtitle", "info": "No page"}})
    with pytest.raises(WikiApiError, match="missingtitle"):
        make_client(handler).get(action="parse", page="Nope")


def test_query_all_follows_continuation():
    def handler(request: httpx.Request) -> httpx.Response:
        if "apcontinue" not in request.url.params:
            return httpx.Response(200, json={"continue": {"apcontinue": "B", "continue": "-||"},
                                             "query": {"allpages": [{"title": "A"}]}})
        assert request.url.params["apcontinue"] == "B"
        assert request.url.params["list"] == "allpages"  # the original parameters are sent again
        return httpx.Response(200, json={"query": {"allpages": [{"title": "B"}]}})

    batches = list(make_client(handler).query_all(list="allpages"))
    assert [page["title"] for batch in batches for page in batch["allpages"]] == ["A", "B"]
