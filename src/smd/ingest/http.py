"""Throttled, retrying HTTP client for NSE archives and APIs."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import httpx

from smd.utils import get_logger

log = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
NSE_HOME = "https://www.nseindia.com/"
RETRY_STATUS = {401, 403, 408, 429, 500, 502, 503, 504}


@dataclass
class FetchResult:
    url: str
    status: int
    content: bytes


class ThrottledClient:
    """httpx client with a global request-rate limit, exponential backoff with jitter,
    and NSE cookie priming for `www.nseindia.com/api` endpoints.

    A 404 is returned (not raised) - the caller decides whether it means "holiday".
    """

    def __init__(
        self,
        rate_limit_per_sec: float = 2.0,
        max_retries: int = 5,
        timeout_sec: float = 30,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        self.min_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
        self.max_retries = max_retries
        self._sleep = sleep
        self._last = 0.0
        self._primed = False
        self.n_requests = 0
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Referer": NSE_HOME},
            timeout=timeout_sec,
            follow_redirects=True,
            transport=transport,
        )

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()

    def _prime(self) -> None:
        self._throttle()
        try:
            self.client.get(NSE_HOME)
            self.n_requests += 1
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            log.warning("cookie priming failed: %s", exc)
        self._primed = True

    def get(self, url: str, params: dict | None = None) -> FetchResult:
        needs_cookies = "nseindia.com/api" in url
        if needs_cookies and not self._primed:
            self._prime()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.client.get(url, params=params)
                self.n_requests += 1
                if r.status_code == 200 or r.status_code == 404:
                    return FetchResult(str(r.url), r.status_code, r.content)
                if r.status_code not in RETRY_STATUS:
                    return FetchResult(str(r.url), r.status_code, r.content)
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                if needs_cookies and r.status_code in (401, 403):
                    self._prime()
            except httpx.HTTPError as exc:
                last_exc = exc
            backoff = min(60.0, 2.0**attempt) * (0.5 + random.random())
            log.debug("retry %d for %s in %.1fs (%s)", attempt + 1, url, backoff, last_exc)
            self._sleep(backoff)
        raise RuntimeError(f"giving up on {url}: {last_exc}")

    def close(self) -> None:
        self.client.close()
