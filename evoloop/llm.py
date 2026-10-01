"""Interface de LLM + cliente Anthropic (stdlib, NÃO testado ao vivo aqui: sem chave de API neste ambiente)."""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    tokens: int


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class ToolResponse:
    text: str
    calls: list
    tokens: int


@dataclass
class ChatResponse:
    blocks: list            # blocos no formato Anthropic: {"type":"text","text"} | {"type":"tool_use","id","name","input"}
    stop_reason: str
    tokens: int

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.blocks if b.get("type") == "text")

    @property
    def tool_uses(self) -> list:
        return [b for b in self.blocks if b.get("type") == "tool_use"]


class LLM(Protocol):
    def complete(self, role: str, system: str, user: str) -> LLMResponse: ...
    def complete_tools(self, role: str, system: str, user: str, tools: list, tool_choice: str = "auto") -> ToolResponse: ...


class AnthropicLLM:
    """roles: policy|planner|reflect|forger|judge. Mesmo modelo congelado em todos os papéis (só o harness evolui)."""
    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024):
        self.model = model or os.environ.get("EVOLOOP_MODEL", "claude-sonnet-5-5")
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY ausente")

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0 if role in ("policy", "judge") else 0.7,
                           "system": system or "You are a helpful assistant.", "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                     {"content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return LLMResponse(text, int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)))


def _anthropic_tools(self, role, system, user, tools, tool_choice="auto"):
    """Function calling / tool use NATIVO (Messages API): `tools=[{name,description,input_schema}]`, `tool_choice={"type":"auto"}`.
    O modelo decide sozinho se chama uma ferramenta; a resposta traz blocos `tool_use`. NÃO testado ao vivo (sem chave aqui)."""
    body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": 0.0, "system": system or "You are a helpful assistant.",
                       "messages": [{"role": "user", "content": user}], "tools": tools, "tool_choice": {"type": tool_choice}}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    calls = [ToolCall(b["name"], b.get("input", {})) for b in d.get("content", []) if b.get("type") == "tool_use"]
    u = d.get("usage", {})
    return ToolResponse(text, calls, int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)))


AnthropicLLM.complete_tools = _anthropic_tools


def _anthropic_chat(self, system, messages, tools, tool_choice="auto", max_tokens=None):
    """Conversa multi-turno com tool use (tool_result pode conter blocos de texto E imagem base64). NÃO testado ao vivo (sem chave)."""
    body = {"model": self.model, "max_tokens": max_tokens or max(self.max_tokens, 4096), "temperature": 0.0, "system": system,
            "messages": messages}
    if tools:
        body["tools"], body["tool_choice"] = tools, {"type": tool_choice}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return ChatResponse(d.get("content", []), d.get("stop_reason", ""), int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)))


AnthropicLLM.chat = _anthropic_chat


class OpenAICompatibleLLM:
    """Cliente sem dependências para Ollama, llama.cpp, vLLM e APIs OpenAI-compatible.

    O endpoint deve implementar ``/v1/chat/completions``. As mensagens internas no
    formato Anthropic são convertidas para o formato OpenAI, inclusive tool calls.
    """
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 4096, timeout: int = 300):
        self.model = model or os.environ.get("EVOLOOP_MODEL", "qwen3:4b")
        self.base_url = (base_url or os.environ.get("EVOLOOP_BASE_URL", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.key = api_key if api_key is not None else os.environ.get("EVOLOOP_MODEL_API_KEY", "ollama")
        self.max_tokens, self.timeout = max_tokens, timeout

    @staticmethod
    def _tools(tools):
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}})}} for t in tools]

    @staticmethod
    def _messages(messages):
        out = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue
            if msg["role"] == "assistant":
                text = "".join(x.get("text", "") for x in content if x.get("type") == "text")
                calls = [{"id": x["id"], "type": "function", "function": {"name": x["name"],
                          "arguments": json.dumps(x.get("input", {}), ensure_ascii=False)}}
                         for x in content if x.get("type") == "tool_use"]
                item = {"role": "assistant", "content": text or None}
                if calls: item["tool_calls"] = calls
                out.append(item)
            else:
                for x in content:
                    if x.get("type") == "tool_result":
                        value = x.get("content", "")
                        if not isinstance(value, str):
                            value = json.dumps(value, ensure_ascii=False)
                        out.append({"role": "tool", "tool_call_id": x["tool_use_id"], "content": value})
        return out

    def _request(self, body):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     json.dumps(body, ensure_ascii=False).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace")
            raise RuntimeError(f"modelo respondeu HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"não foi possível conectar ao modelo em {self.base_url}: {exc.reason}") from exc

    @staticmethod
    def _usage(data):
        usage = data.get("usage") or {}
        return int(usage.get("total_tokens") or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

    def chat(self, system, messages, tools, tool_choice="auto", max_tokens=None):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}] + self._messages(messages),
                "temperature": 0, "max_tokens": max_tokens or self.max_tokens, "stream": False}
        if tools:
            body["tools"], body["tool_choice"] = self._tools(tools), tool_choice
        data = self._request(body)
        try:
            msg, finish = data["choices"][0]["message"], data["choices"][0].get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("resposta inválida do servidor de modelo") from exc
        blocks = []
        if msg.get("content"):
            blocks.append({"type": "text", "text": msg["content"]})
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try: args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError: args = {"_raw": raw}
            blocks.append({"type": "tool_use", "id": call.get("id") or "call", "name": fn.get("name", ""), "input": args})
        return ChatResponse(blocks, finish, self._usage(data))

    def complete(self, role, system, user):
        response = self.chat(system or "You are a helpful assistant.", [{"role": "user", "content": user}], [], max_tokens=self.max_tokens)
        return LLMResponse(response.text, response.tokens)

    def complete_tools(self, role, system, user, tools, tool_choice="auto"):
        response = self.chat(system or "You are a helpful assistant.", [{"role": "user", "content": user}], tools, tool_choice,
                             self.max_tokens)
        return ToolResponse(response.text, [ToolCall(x["name"], x["input"]) for x in response.tool_uses], response.tokens)


def extract_json(text: str):
    """Extrai o primeiro objeto JSON de uma resposta de LLM (tolera cercas ```)."""
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        s = max(parts[1::2], key=len, default=s)
        s = s[4:] if s.startswith("json") else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("sem JSON na resposta")
    return json.loads(s[i:j + 1])


class CountingLLM:
    """Envolve qualquer LLM e contabiliza chamadas/tokens por papel (custo do ciclo)."""
    def __init__(self, inner: LLM):
        self.inner = inner
        self.calls: dict[str, int] = {}
        self.tokens: dict[str, int] = {}

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        r = self.inner.complete(role, system, user)
        self.calls[role] = self.calls.get(role, 0) + 1
        self.tokens[role] = self.tokens.get(role, 0) + r.tokens
        return r

    def complete_tools(self, role: str, system: str, user: str, tools: list, tool_choice: str = "auto") -> ToolResponse:
        r = self.inner.complete_tools(role, system, user, tools, tool_choice)
        self.calls[role] = self.calls.get(role, 0) + 1
        self.tokens[role] = self.tokens.get(role, 0) + r.tokens
        return r

    def chat(self, system, messages, tools, tool_choice="auto", max_tokens=None) -> ChatResponse:
        r = self.inner.chat(system, messages, tools, tool_choice, max_tokens)
        self.calls["chat"] = self.calls.get("chat", 0) + 1
        self.tokens["chat"] = self.tokens.get("chat", 0) + r.tokens
        return r
