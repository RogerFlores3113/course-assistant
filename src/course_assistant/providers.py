"""AI providers: the Anthropic API (student's own key), a local OpenAI-compatible server (LM Studio), and a fake for tests."""
from __future__ import annotations

import base64
import os
from typing import Any, Callable

import httpx

DEFAULT_MODEL = "claude-opus-5-5"
ALLOWED_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
DEFAULT_LOCAL_URL = "http://127.0.0.1:1234"
BETAS = ["server-side-fallback-2026-07-01"]
TIMEOUT_SECONDS = 180.0


class ProviderError(Exception):
    """Anything that went wrong talking to the AI. The message is written for the student and shown as is."""


class ProviderRefused(ProviderError):
    """The model declined to answer this request."""


class Provider:
    name = "none"
    model = ""

    def complete(self, system: str, messages: list[dict], *, max_tokens: int = 4000) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_image(self, system: str, image_bytes: bytes, media_type: str, prompt: str) -> str:
        raise ProviderError("This AI provider can't read images.")


# --------------------------------------------------------------------------- Anthropic
class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL, *, client: Any = None,
                 timeout: float = TIMEOUT_SECONDS):
        self.model = model if model in ALLOWED_MODELS else DEFAULT_MODEL
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self.client = client

    def _call(self, system: str, messages: list[dict], max_tokens: int) -> str:
        import anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                betas=list(BETAS), fallbacks="default")
        except anthropic.AuthenticationError as e:
            raise ProviderError("The Anthropic API key was not accepted. Check the key in Settings.") from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Anthropic says you are sending requests too fast. Wait a minute and try again.") from e
        except anthropic.APIConnectionError as e:  # includes timeouts
            raise ProviderError("Could not reach Anthropic. Check your internet connection and try again.") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic had a problem on its side (error {getattr(e, 'status_code', '?')}). "
                                "Try again in a moment.") from e
        if getattr(resp, "stop_reason", None) == "refusal":
            raise ProviderRefused("The AI declined to answer this request. Try rewording the question.")
        parts = [b.text for b in (getattr(resp, "content", None) or [])
                 if getattr(b, "type", None) == "text" and getattr(b, "text", None)]
        return "".join(parts).strip()

    def complete(self, system: str, messages: list[dict], *, max_tokens: int = 4000) -> str:
        return self._call(system, messages, max_tokens)

    def complete_image(self, system: str, image_bytes: bytes, media_type: str, prompt: str) -> str:
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.b64encode(image_bytes).decode("ascii")}},
            {"type": "text", "text": prompt},
        ]
        return self._call(system, [{"role": "user", "content": content}], 4000)


# --------------------------------------------------------------------------- local (LM Studio etc.)
class LocalProvider(Provider):
    name = "local"

    def __init__(self, base_url: str = DEFAULT_LOCAL_URL, model: str = "", *, transport: Any = None,
                 timeout: float = 300.0):
        self.base_url = (base_url or DEFAULT_LOCAL_URL).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or "local-model"
        self._transport = transport
        self._timeout = timeout

    def _post(self, messages: list[dict], max_tokens: int) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        try:
            with httpx.Client(transport=self._transport, timeout=self._timeout) as c:
                r = c.post(url, json=body)
        except httpx.TimeoutException as e:
            raise ProviderError("The local model took too long to answer. Try a smaller model or a shorter question.") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach the local model server at {self.base_url}. "
                                "Start LM Studio (or your server) and load a model, then try again.") from e
        if r.status_code >= 400:
            raise ProviderError(f"The local model server returned an error ({r.status_code}). "
                                "Check that a model is loaded and the model name in Settings is right.")
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise ProviderError("The local model server sent a reply the app could not read.") from e
        return (text or "").strip()

    def complete(self, system: str, messages: list[dict], *, max_tokens: int = 4000) -> str:
        return self._post([{"role": "system", "content": system}, *messages], max_tokens)

    def complete_image(self, system: str, image_bytes: bytes, media_type: str, prompt: str) -> str:
        url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]
        return self._post([{"role": "system", "content": system}, {"role": "user", "content": content}], 4000)


# --------------------------------------------------------------------------- fake
class FakeProvider(Provider):
    """For tests. responses is a list (used in order, the last repeats) or a callable(system, messages) -> str."""
    name = "fake"

    def __init__(self, responses: list[str] | Callable[[str, list[dict]], str] | str = "ok", model: str = "fake-model"):
        self.model = model
        self.responses = [responses] if isinstance(responses, str) else responses
        self.calls: list[dict[str, Any]] = []
        self.image_calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict], *, max_tokens: int = 4000) -> str:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        if callable(self.responses):
            return self.responses(system, messages)
        if not self.responses:
            return ""
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[i]

    def complete_image(self, system: str, image_bytes: bytes, media_type: str, prompt: str) -> str:
        self.image_calls.append({"system": system, "bytes": image_bytes, "media_type": media_type, "prompt": prompt})
        return "fake image text"


# --------------------------------------------------------------------------- factory
def get_provider(store) -> Provider | None:
    """Build the provider the student configured, or None when none is set up."""
    kind = (store.get_setting("ai_provider", "") or "").strip().lower()
    if kind == "anthropic":
        key = store.get_setting("anthropic_api_key", "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            return None
        return AnthropicProvider(key, store.get_setting("anthropic_model", "") or DEFAULT_MODEL)
    if kind == "local":
        return LocalProvider(store.get_setting("local_base_url", "") or DEFAULT_LOCAL_URL,
                             store.get_setting("local_model", ""))
    return None
