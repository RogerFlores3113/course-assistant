import base64
import json

import anthropic
import httpx
import pytest

from course_assistant import providers as P
from course_assistant.store import Store


class Block:
    def __init__(self, type, text=None):
        self.type, self.text = type, text


class Resp:
    def __init__(self, content, stop_reason="end_turn"):
        self.content, self.stop_reason = content, stop_reason


class FakeClient:
    def __init__(self, resp=None, exc=None):
        self.calls, self.resp, self.exc = [], resp, exc
        self.beta = self
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.resp


def _req():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_anthropic_exact_kwargs_and_text_only():
    c = FakeClient(Resp([Block("thinking", "hmm"), Block("text", "Hello "), Block("text", "world")]))
    p = P.AnthropicProvider(client=c)
    assert p.model == "claude-opus-5-5"
    out = p.complete("SYS", [{"role": "user", "content": "hi"}], max_tokens=123)
    assert out == "Hello world"
    assert c.calls == [dict(model="claude-opus-5-5", max_tokens=123, system="SYS",
                            messages=[{"role": "user", "content": "hi"}], thinking={"type": "adaptive"},
                            output_config={"effort": "medium"}, betas=["server-side-fallback-2026-07-01"],
                            fallbacks="default")]
    assert "tool_choice" not in c.calls[0]


def test_model_allowlist():
    assert P.AnthropicProvider(model="claude-sonnet-5-5", client=FakeClient()).model == "claude-sonnet-5-5"
    assert P.AnthropicProvider(model="gpt-9", client=FakeClient()).model == "claude-opus-5-5"


def test_refusal_checked_before_content():
    c = FakeClient(Resp([Block("text", "secret")], stop_reason="refusal"))
    with pytest.raises(P.ProviderRefused) as e:
        P.AnthropicProvider(client=c).complete("s", [])
    assert "secret" not in str(e.value)
    assert isinstance(e.value, P.ProviderError)


def test_refusal_with_no_content_attr():
    class R:
        stop_reason = "refusal"
    with pytest.raises(P.ProviderRefused):
        P.AnthropicProvider(client=FakeClient(R())).complete("s", [])


@pytest.mark.parametrize("exc,needle", [
    (anthropic.AuthenticationError("x", response=httpx.Response(401, request=_req()), body=None), "key"),
    (anthropic.RateLimitError("x", response=httpx.Response(429, request=_req()), body=None), "too fast"),
    (anthropic.APIConnectionError(request=_req()), "reach"),
    (anthropic.APITimeoutError(request=_req()), "reach"),
    (anthropic.InternalServerError("x", response=httpx.Response(500, request=_req()), body=None), "500"),
])
def test_error_mapping(exc, needle):
    with pytest.raises(P.ProviderError) as e:
        P.AnthropicProvider(client=FakeClient(exc=exc)).complete("s", [])
    assert needle in str(e.value)
    assert not isinstance(e.value, P.ProviderRefused)


def test_anthropic_image():
    c = FakeClient(Resp([Block("text", "notes text")]))
    out = P.AnthropicProvider(client=c).complete_image("S", b"\x89PNG", "image/png", "Transcribe")
    assert out == "notes text"
    content = c.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                      "data": base64.b64encode(b"\x89PNG").decode()}}
    assert content[1] == {"type": "text", "text": "Transcribe"}


def local(handler, **kw):
    return P.LocalProvider("http://127.0.0.1:1234", "m1", transport=httpx.MockTransport(handler), **kw)


def test_local_request_shape():
    seen = {}

    def h(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " answer "}}]})

    out = local(h).complete("SYS", [{"role": "user", "content": "q"}], max_tokens=50)
    assert out == "answer"
    assert seen["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert seen["body"]["model"] == "m1" and seen["body"]["max_tokens"] == 50
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["messages"][1]["content"] == "q"


def test_local_base_url_with_v1_suffix():
    assert P.LocalProvider("http://x:1/v1/").base_url == "http://x:1"


def test_local_not_running():
    def h(req):
        raise httpx.ConnectError("refused")
    with pytest.raises(P.ProviderError) as e:
        local(h).complete("s", [])
    assert "couldn't reach your local model at http://127.0.0.1:1234" in str(e.value)
    assert "LM Studio running with a model loaded" in str(e.value)


def test_local_timeout_http_error_and_bad_json():
    def t(req):
        raise httpx.ReadTimeout("slow")
    with pytest.raises(P.ProviderError, match="too long"):
        local(t).complete("s", [])
    with pytest.raises(P.ProviderError, match="500"):
        local(lambda r: httpx.Response(500)).complete("s", [])
    with pytest.raises(P.ProviderError, match="did not look like a model server"):
        local(lambda r: httpx.Response(200, json={"oops": 1})).complete("s", [])
    with pytest.raises(P.ProviderError, match="did not look like a model server"):
        local(lambda r: httpx.Response(200, text="not json")).complete("s", [])


def test_local_image_data_url():
    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    local(h).complete_image("S", b"abc", "image/jpeg", "what")
    part = seen["body"]["messages"][1]["content"][1]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(b"abc").decode()


def test_fake_provider_list_and_callable():
    f = P.FakeProvider(["a", "b"])
    assert [f.complete("s1", [{"role": "user", "content": "1"}]), f.complete("s2", []), f.complete("s3", [])] == ["a", "b", "b"]
    assert f.calls[0]["system"] == "s1" and f.calls[0]["messages"][0]["content"] == "1"
    g = P.FakeProvider(lambda s, m: f"echo {len(m)}")
    assert g.complete("s", [{}, {}]) == "echo 2"
    assert f.complete_image("s", b"x", "image/png", "p") == "fake image text" and len(f.image_calls) == 1


def test_get_provider(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Store(":memory:")
    assert P.get_provider(s) is None
    s.set_setting("ai_provider", "none")
    assert P.get_provider(s) is None
    s.set_setting("ai_provider", "anthropic")
    assert P.get_provider(s) is None  # no key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    p = P.get_provider(s)
    assert isinstance(p, P.AnthropicProvider) and p.model == "claude-opus-5-5"
    s.set_setting("anthropic_api_key", "sk-set")
    s.set_setting("anthropic_model", "claude-haiku-5-5")
    assert P.get_provider(s).model == "claude-haiku-5-5"
    s.set_setting("ai_provider", "local")
    s.set_setting("local_model", "qwen")
    lp = P.get_provider(s)
    assert isinstance(lp, P.LocalProvider) and lp.model == "qwen" and lp.base_url == "http://127.0.0.1:1234"
    s.set_setting("local_base_url", "http://10.0.0.5:8080")
    assert P.get_provider(s).base_url == "http://10.0.0.5:8080"


def test_local_502_blames_the_path_not_the_model():
    with pytest.raises(P.ProviderError) as e:
        local(lambda r: httpx.Response(502)).complete("s", [])
    msg = str(e.value)
    assert "502" in msg and "proxy" in msg and "http://127.0.0.1:1234" in msg
    assert "model name" not in msg.lower().split("proxy")[0]


@pytest.mark.parametrize("code,word", [(401, "refused"), (403, "refused"), (404, "404"), (400, "model name"), (503, "proxy")])
def test_local_status_messages_are_specific(code, word):
    with pytest.raises(P.ProviderError, match=word):
        local(lambda r: httpx.Response(code)).complete("s", [])


def test_is_loopback_url():
    for u in ("http://127.0.0.1:1234", "http://localhost:1234/v1", "http://[::1]:8000", "127.0.0.1:5", "http://127.5.5.5"):
        assert P.is_loopback_url(u), u
    for u in ("http://example.com", "http://192.168.1.5:1234", "https://api.anthropic.com", ""):
        assert not P.is_loopback_url(u), u


def test_local_loopback_ignores_proxy_env_but_remote_keeps_it(monkeypatch):
    seen = []
    real = httpx.Client

    def spy(*a, **kw):
        seen.append(kw.get("trust_env", True))
        return real(*a, **kw)

    monkeypatch.setattr(P.httpx, "Client", spy)
    ok = lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})  # noqa: E731
    P.LocalProvider("http://127.0.0.1:1234", transport=httpx.MockTransport(ok)).complete("s", [])
    P.LocalProvider("http://localhost:1234", transport=httpx.MockTransport(ok)).complete("s", [])
    P.LocalProvider("http://lab-pc.local:1234", transport=httpx.MockTransport(ok)).complete("s", [])
    assert seen == [False, False, True]


def test_local_really_bypasses_a_proxy_on_loopback(monkeypatch):
    """A real loopback server, with HTTP(S)_PROXY pointed at a dead port and NO_PROXY unset: the call must still work."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for k in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")
    try:
        out = P.LocalProvider(f"http://127.0.0.1:{srv.server_port}", "m").complete("s", [{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert out == "pong"
