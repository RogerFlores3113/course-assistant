import httpx
import pytest

from course_assistant.fetcher import FetchBlocked, FetchError, PoliteFetcher, is_public_ip

PUBLIC = lambda host, port: ["93.184.216.34"]  # noqa: E731


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make(handler, **kw):
    clock = Clock()
    f = PoliteFetcher(transport=httpx.MockTransport(handler), clock=clock.now, sleep=clock.sleep,
                      resolver=kw.pop("resolver", PUBLIC), **kw)
    return f, clock


def site(robots="User-agent: *\nDisallow: /private/\n", pages=None, log=None):
    pages = pages or {}

    def handler(req: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(str(req.url))
        assert "cookie" not in req.headers and "authorization" not in req.headers
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text=robots) if robots is not None else httpx.Response(404)
        if req.url.path in pages:
            body, ctype = pages[req.url.path]
            return httpx.Response(200, content=body, headers={"content-type": ctype})
        return httpx.Response(404)
    return handler


def test_get_ok_decodes_text():
    f, _ = make(site(pages={"/a": (b"<html>hi</html>", "text/html; charset=utf-8")}))
    r = f.get("https://example.edu/a")
    assert (r.status, r.content_type, r.text, r.final_url) == (200, "text/html", "<html>hi</html>", "https://example.edu/a")
    assert r.content == b"<html>hi</html>" and r.url == "https://example.edu/a"


def test_robots_disallow_blocks():
    log = []
    f, _ = make(site(pages={"/private/x": (b"secret", "text/html")}, log=log))
    assert f.allowed("https://example.edu/private/x") is False
    assert f.allowed("https://example.edu/public") is True
    with pytest.raises(FetchBlocked) as e:
        f.get("https://example.edu/private/x")
    assert "robots" in str(e.value)
    assert "https://example.edu/private/x" not in log  # never requested


def test_robots_cached_per_host():
    log = []
    f, _ = make(site(pages={"/a": (b"a", "text/plain"), "/b": (b"b", "text/plain")}, log=log), min_interval=0)
    f.get("https://example.edu/a")
    f.get("https://example.edu/b")
    assert log.count("https://example.edu/robots.txt") == 1


def test_robots_missing_allows_and_errors_deny():
    f, _ = make(site(robots=None, pages={"/a": (b"a", "text/plain")}))
    assert f.get("https://example.edu/a").text == "a"

    def forbidden(req):
        return httpx.Response(403) if req.url.path == "/robots.txt" else httpx.Response(200, text="x")
    f, _ = make(forbidden)
    assert f.allowed("https://example.edu/a") is False

    def broken(req):
        return httpx.Response(503) if req.url.path == "/robots.txt" else httpx.Response(200, text="x")
    f, _ = make(broken)
    assert f.allowed("https://example.edu/a") is False


def test_rate_limit_spacing_per_host():
    pages = {"/a": (b"a", "text/plain"), "/b": (b"b", "text/plain"), "/c": (b"c", "text/plain")}
    f, clock = make(site(pages=pages), min_interval=15)
    f.get("https://one.edu/a")
    assert clock.slept == []  # first request goes at once
    f.get("https://one.edu/b")
    assert clock.slept == [15]
    clock.t += 6
    f.get("https://one.edu/c")
    assert clock.slept == [15, 9]
    f.get("https://two.edu/a")  # a different host has its own clock
    assert clock.slept == [15, 9]


def test_crawl_delay_respected():
    pages = {"/a": (b"a", "text/plain"), "/b": (b"b", "text/plain")}
    f, clock = make(site(robots="User-agent: *\nCrawl-delay: 40\n", pages=pages), min_interval=15)
    f.get("https://one.edu/a")
    f.get("https://one.edu/b")
    assert clock.slept == [40]


@pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "::1", "0.0.0.0",
                                  "172.16.0.1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "240.0.0.1", "100.64.0.1"])
def test_non_public_addresses_rejected(addr):
    assert not is_public_ip(addr)


def test_public_address_ok():
    assert is_public_ip("93.184.216.34") and is_public_ip("2606:2800:220:1:248:1893:25c8:1946")


def test_private_resolution_refused():
    log = []
    f, _ = make(site(log=log), resolver=lambda h, p: ["10.0.0.7"])
    with pytest.raises(FetchBlocked):
        f.get("https://intranet.example.edu/page")
    assert log == []
    assert f.allowed("https://intranet.example.edu/page") is False


def test_mixed_resolution_refused_and_literal_ips():
    f, _ = make(site(), resolver=lambda h, p: ["93.184.216.34", "127.0.0.1"])
    with pytest.raises(FetchBlocked):
        f.get("https://rebind.example.edu/")
    f, _ = make(site())
    for url in ("http://127.0.0.1/", "http://[::1]/x", "http://169.254.169.254/latest/meta-data", "http://192.168.0.1/"):
        with pytest.raises(FetchBlocked):
            f.get(url)


def test_scheme_and_credentials_rejected():
    f, _ = make(site())
    for url in ("ftp://example.edu/a", "file:///etc/passwd", "javascript:alert(1)", "example.edu/a", "",
                "https://user:pw@example.edu/a"):
        with pytest.raises(FetchError):
            f.get(url)


def test_redirects_followed_and_rechecked():
    log = []

    def handler(req):
        log.append(str(req.url))
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        if req.url.path == "/old":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, text="new page", headers={"content-type": "text/html"})
    f, _ = make(handler, min_interval=0)
    r = f.get("https://example.edu/old")
    assert r.final_url == "https://example.edu/new" and r.text == "new page" and r.url.endswith("/old")


def test_redirect_to_private_host_refused():
    def resolver(host, port):
        return ["10.1.1.1"] if host == "internal.edu" else ["93.184.216.34"]

    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "http://internal.edu/admin"})
    f, _ = make(handler, resolver=resolver, min_interval=0)
    with pytest.raises(FetchBlocked):
        f.get("https://example.edu/go")


def test_redirect_to_other_scheme_refused():
    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "file:///etc/passwd"})
    f, _ = make(handler, min_interval=0)
    with pytest.raises(FetchError):
        f.get("https://example.edu/go")


def test_redirect_loop_limit_is_five():
    hops = []

    def handler(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        hops.append(req.url.path)
        return httpx.Response(302, headers={"location": f"/n{len(hops)}"})
    f, _ = make(handler, min_interval=0)
    with pytest.raises(FetchError, match="redirect"):
        f.get("https://example.edu/start")
    assert len(hops) == 6  # the first request plus five redirects


def test_redirect_target_robots_checked():
    def handler(req):
        if req.url.host == "other.edu" and req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        if req.url.host == "example.edu":
            return httpx.Response(302, headers={"location": "https://other.edu/page"})
        return httpx.Response(200, text="should not be fetched")
    f, _ = make(handler, min_interval=0)
    with pytest.raises(FetchBlocked):
        f.get("https://example.edu/a")


def test_size_cap_streamed_and_declared():
    big = b"x" * 5000
    f, _ = make(site(pages={"/big": (big, "text/plain")}), max_bytes=1000)
    with pytest.raises(FetchError, match="bigger"):
        f.get("https://example.edu/big")

    def lying(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=b"tiny", headers={"content-length": "99999999"})
    f, _ = make(lying, max_bytes=1000)
    with pytest.raises(FetchError, match="bigger"):
        f.get("https://example.edu/x")


def test_error_statuses_are_friendly():
    f, _ = make(site(pages={}), min_interval=0)
    with pytest.raises(FetchError, match="404"):
        f.get("https://example.edu/missing")

    def denied(req):
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(403)
    f, _ = make(denied, min_interval=0)
    with pytest.raises(FetchError, match="sign-in"):
        f.get("https://example.edu/x")


def test_binary_content_has_no_text():
    f, _ = make(site(pages={"/f.pdf": (b"%PDF-1.4 binary", "application/pdf")}))
    r = f.get("https://example.edu/f.pdf")
    assert r.content_type == "application/pdf" and r.text == "" and r.content.startswith(b"%PDF")


def test_no_cookies_sent_back():
    seen = []

    def handler(req):
        seen.append(req.headers.get("cookie"))
        if req.url.path == "/robots.txt":
            return httpx.Response(404, headers={"set-cookie": "a=1"})
        return httpx.Response(200, text="ok", headers={"set-cookie": "sid=abc"})
    f, _ = make(handler, min_interval=0)
    f.get("https://example.edu/1")
    f.get("https://example.edu/2")
    assert seen == [None, None, None]


def test_user_agent_sent():
    uas = []

    def handler(req):
        uas.append(req.headers["user-agent"])
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(200, text="ok")
    f, _ = make(handler, user_agent="TestBot/1")
    f.get("https://example.edu/a")
    assert set(uas) == {"TestBot/1"}
