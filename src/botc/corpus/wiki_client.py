"""A small, polite client for the MediaWiki API of the BOTC wiki.

Politeness rules:
- one request at a time, at most one every `min_interval` seconds;
- a descriptive User-Agent;
- `maxlag`: the server can tell us to wait when it is under load;
- on 429 / 5xx / maxlag we wait (Retry-After or exponential backoff) and retry a few times;
- on 403 / 418 we stop: the server refuses us, we do not try to work around it.

MediaWiki API docs: https://www.mediawiki.org/wiki/API:Main_page
"""

import time
from collections.abc import Iterator
from typing import Any

import httpx

API_URL = "https://wiki.bloodontheclocktower.com/api.php"
USER_AGENT = "botc-rules-corpus/0.1 (non-commercial student project)"


RETRY_STATUSES = {429, 500, 502, 503, 504}
BLOCKED_STATUSES = {403, 418}


class WikiBlockedError(RuntimeError):
    """The server refused our requests (403/418). We stop instead of insisting."""


class WikiApiError(RuntimeError):
    """The API answered 200 but with an `error` object in the JSON."""


class WikiClient:
    def __init__(self, api_url: str = API_URL, min_interval: float = 2.0, max_retries: int = 5,
                 maxlag: int = 5, transport: httpx.BaseTransport | None = None):
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._maxlag = maxlag
        self._last_request = float("-inf")
        self._api_url = api_url
        # `transport` lets the tests replace the network with a fake one (httpx.MockTransport).
        self._http = httpx.Client(headers={"User-Agent": USER_AGENT},
                                  timeout=30.0, transport=transport)

    def __enter__(self) -> "WikiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _wait_turn(self) -> None:
        """Sleep so that two requests are always at least `min_interval` seconds apart."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def get(self, **params: Any) -> dict:
        """Send one API request and return the decoded JSON.

        Example: client.get(action="query", meta="siteinfo")
        """
        # format=json + formatversion=2 give the modern, simpler JSON layout.
        params = {"format": "json", "formatversion": 2, "maxlag": self._maxlag, **params}

        for attempt in range(self._max_retries + 1):
            self._wait_turn()
            response = self._http.get(self._api_url, params=params)

            if response.status_code in BLOCKED_STATUSES:
                raise WikiBlockedError(f"HTTP {response.status_code}: the wiki refuses our requests, stopping.")

            if response.status_code in RETRY_STATUSES:
                self._backoff(attempt, response)
                continue

            response.raise_for_status()
            data = response.json()

            error = data.get("error")
            if error is not None:
                if error.get("code") == "maxlag":  # the server is busy: wait and retry
                    self._backoff(attempt, response)
                    continue
                raise WikiApiError(f"{error.get('code')}: {error.get('info')}")

            return data

        raise WikiApiError(f"Giving up after {self._max_retries} retries: {params}")

    def _backoff(self, attempt: int, response: httpx.Response) -> None:
        """Wait before retrying: honour Retry-After if present, else 2, 4, 8... seconds."""
        if attempt == self._max_retries:
            return
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after is not None else 2.0 ** (attempt + 1)
        print(f"HTTP {response.status_code}, retrying in {delay}s")
        time.sleep(delay)

    def query_all(self, **params: Any) -> Iterator[dict]:
        """Run an `action=query` request and follow the continuation until the end.

        Lists are paginated: when there is more to read, the response contains a `continue`
        object. We send it back with the same parameters to get the next batch.
        Yields the `query` part of each batch.
        """
        params = {"action": "query", **params}
        cont: dict = {}
        while True:
            data = self.get(**{**params, **cont})
            if "query" in data:
                yield data["query"]
            if "continue" not in data:
                return
            cont = data["continue"]
