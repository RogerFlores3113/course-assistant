"""A polite web fetcher for saving course pages.

It obeys robots.txt, waits between requests to the same site, caps the size of what it downloads, speaks only
http and https, refuses addresses that point inside the student's own network, sends no cookies and never signs in.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

DEFAULT_UA = "CourseAssistant/0.1 (local study app; obeys robots.txt)"
MAX_REDIRECTS = 5
MAX_ROBOTS_BYTES = 512 * 1024
MAX_CRAWL_DELAY = 120.0


class FetchError(Exception):
    """The page could not be fetched. The message is safe to show the student."""


class FetchBlocked(FetchError):
    """The site's robots.txt asks robots not to fetch this page (or the address is not allowed)."""


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str
    content: bytes


def default_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"I couldn't find the site {host}. Check the link and your internet connection.") from exc
    return [info[4][0] for info in infos]


def is_public_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified or not ip.is_global)


class PoliteFetcher:
    """One per process: it holds the robots.txt cache and the per-host clock."""

    def __init__(self, user_agent: str = DEFAULT_UA, min_interval: float = 15.0, max_bytes: int = 25_000_000,
                 transport=None, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 resolver: Callable[[str, int], list[str]] | None = None,
                 timeout: float = 20.0, max_redirects: int = MAX_REDIRECTS):
        self.user_agent = user_agent
        self.min_interval = float(min_interval)
        self.max_bytes = int(max_bytes)
        self.transport = transport
        self.clock = clock
        self.sleep = sleep
        self.resolver = resolver or default_resolver
        self.timeout = timeout
        self.max_redirects = max_redirects
        self._robots: dict[str, RobotFileParser | bool] = {}  # True = allow all, False = disallow all
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ safety checks
    @staticmethod
    def _parts(url: str):
        try:
            p = urlsplit(url.strip())
            host = p.hostname
            port = p.port
        except ValueError as exc:
            raise FetchError("That doesn't look like a web address I can open.") from exc
        if p.scheme not in ("http", "https"):
            raise FetchError("I can only open web addresses that start with http:// or https://.")
        if not host:
            raise FetchError("That web address has no site name in it.")
        if p.username or p.password:
            raise FetchError("I don't open addresses that carry a user name or password.")
        return p, host.lower(), port or (443 if p.scheme == "https" else 80)

    def _check_host(self, host: str, port: int) -> None:
        addrs = [host] if _is_ip(host) else self.resolver(host, port)
        if not addrs:
            raise FetchError(f"I couldn't find the site {host}.")
        for a in addrs:
            if not is_public_ip(a):
                raise FetchBlocked(f"I won't open {host}: it points to a private or local address on your own "
                                   "network, not to a public website.")

    # ------------------------------------------------------------ one HTTP request, no redirects followed
    def _request(self, url: str, *, limit: int) -> tuple[httpx.Response, bytes]:
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.5",
                   "Accept-Language": "en"}
        try:
            with httpx.Client(transport=self.transport, follow_redirects=False, timeout=self.timeout,
                              headers=headers, cookies=None) as client:
                with client.stream("GET", url) as resp:
                    declared = resp.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > limit and 200 <= resp.status_code < 300:
                        raise FetchError(_too_big(limit))
                    buf = bytearray()
                    for part in resp.iter_bytes():
                        buf.extend(part)
                        if len(buf) > limit:
                            raise FetchError(_too_big(limit))
                    return resp, bytes(buf)
        except FetchError:
            raise
        except httpx.TimeoutException as exc:
            raise FetchError("The site took too long to answer. Try again later.") from exc
        except httpx.HTTPError as exc:
            raise FetchError("I couldn't reach that site. Check the link and your internet connection.") from exc

    def _safe_request(self, url: str, *, limit: int, rate_limit: bool) -> tuple[httpx.Response, bytes, str]:
        """Follows up to max_redirects hops, re-checking scheme, host address and (optionally) pacing every hop."""
        current = url
        for _hop in range(self.max_redirects + 1):
            p, host, port = self._parts(current)
            self._check_host(host, port)
            if rate_limit:
                self._wait_turn(host, port)
            try:
                resp, body = self._request(current, limit=limit)
            finally:
                if rate_limit:
                    self._mark(host, port)
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                current = urljoin(current, resp.headers["location"])
                if rate_limit and not self.allowed(current):
                    raise FetchBlocked("This site's robots.txt asks robots not to open the page it sent me to.")
                continue
            return resp, body, current
        raise FetchError("That link bounced through too many redirects, so I stopped.")

    # ------------------------------------------------------------ pacing
    def _interval(self, key: str) -> float:
        rp = self._robots.get(key)
        delay = 0.0
        if isinstance(rp, RobotFileParser):
            try:
                delay = float(rp.crawl_delay(self.user_agent) or 0)
            except (TypeError, ValueError):
                delay = 0.0
        return max(self.min_interval, min(delay, MAX_CRAWL_DELAY))

    def _wait_turn(self, host: str, port: int) -> None:
        key = f"{host}:{port}"
        with self._lock:
            last = self._last.get(key)
            interval = self._interval(self._robots_key(host, port))
        if last is not None:
            wait = interval - (self.clock() - last)
            if wait > 0:
                self.sleep(wait)

    def _mark(self, host: str, port: int) -> None:
        with self._lock:
            self._last[f"{host}:{port}"] = self.clock()

    # ------------------------------------------------------------ robots.txt
    @staticmethod
    def _robots_key(host: str, port: int) -> str:
        return f"{host}:{port}"

    def _load_robots(self, scheme: str, host: str, port: int) -> RobotFileParser | bool:
        key = self._robots_key(host, port)
        if key in self._robots:
            return self._robots[key]
        netloc = host if port in (80, 443) else f"{host}:{port}"
        if ":" in host:
            netloc = f"[{host}]" + ("" if port in (80, 443) else f":{port}")
        robots_url = f"{scheme}://{netloc}/robots.txt"
        result: RobotFileParser | bool
        try:
            resp, body, _ = self._safe_request(robots_url, limit=MAX_ROBOTS_BYTES, rate_limit=False)
            if resp.status_code in (401, 403):
                result = False
            elif 400 <= resp.status_code < 500:
                result = True
            elif resp.status_code >= 500:
                result = False
            else:
                rp = RobotFileParser()
                rp.parse(body.decode("utf-8", errors="replace").splitlines())
                result = rp
        except FetchBlocked:
            raise
        except FetchError:
            result = False  # can't read it: assume the site would say no
        self._robots[key] = result
        return result

    def allowed(self, url: str) -> bool:
        try:
            p, host, port = self._parts(url)
            self._check_host(host, port)
            rp = self._load_robots(p.scheme, host, port)
        except FetchError:
            return False
        if isinstance(rp, bool):
            return rp
        return rp.can_fetch(self.user_agent, url)

    # ------------------------------------------------------------ public
    def get(self, url: str) -> FetchResult:
        p, host, port = self._parts(url)
        self._check_host(host, port)
        if not self.allowed(url):
            raise FetchBlocked(f"{host} asks robots not to open this page (its robots.txt says no), so I didn't "
                               "save it. You can open it in your browser and save it as a PDF to add it yourself.")
        resp, body, final = self._safe_request(url, limit=self.max_bytes, rate_limit=True)
        if resp.status_code >= 400:
            raise FetchError(_status_message(resp.status_code, host))
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        text = ""
        if ctype.startswith("text/") or "html" in ctype or "xml" in ctype or "json" in ctype or not ctype:
            enc = resp.charset_encoding or "utf-8"
            try:
                text = body.decode(enc, errors="replace")
            except LookupError:
                text = body.decode("utf-8", errors="replace")
        return FetchResult(url=url, final_url=final, status=resp.status_code, content_type=ctype, text=text,
                           content=body)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _too_big(limit: int) -> str:
    return f"That page is bigger than {limit // 1_000_000} MB, so I stopped downloading it."


def _status_message(status: int, host: str) -> str:
    if status in (401, 403):
        return (f"{host} wants a sign-in or doesn't allow this page. This app never signs in for you. Save the page "
                "as a PDF from your browser and add that file instead.")
    if status == 404:
        return "That page wasn't found (404). Check the link."
    if status == 429:
        return f"{host} says I'm asking too often. Try again in a little while."
    return f"{host} answered with an error ({status}). Try again later."


_default: PoliteFetcher | None = None


def get_default_fetcher() -> PoliteFetcher:
    """The one shared fetcher for the process, so pacing and the robots cache hold across requests."""
    global _default
    if _default is None:
        _default = PoliteFetcher()
    return _default
