"""Duplo de teste (NÃO é um LLM): RuleLLM implementa, de forma determinística e transparente, um 'modelo' com um bug de comportamento
que uma lição de memória conserta. Serve para exercitar encanamento (servidor→agente→ferramentas→feedback→consolidação→aprovação).
Não diz NADA sobre a capacidade de um LLM real."""
from __future__ import annotations
import json
import re

from .llm import ChatResponse, LLMResponse, ToolResponse

LESSON = "Always create files with the write_file tool under /workspace, then verify with ls /workspace."


class RuleLLM:
    def __init__(self):
        self.calls = {}

    def _uid(self, msgs):
        return f"tu_{len(msgs)}"

    def chat(self, system, messages, tools, tool_choice="auto", max_tokens=None) -> ChatResponse:
        self.calls["chat"] = self.calls.get("chat", 0) + 1
        user = next((m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str)), "")
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):          # acabou de receber tool_result
            out = last["content"][0]["content"]
            out = out if isinstance(out, str) else "[img]"
            ok = "exit 0" in out or out.strip() == "ok" or "report.txt" in out
            return ChatResponse([{"type": "text", "text": "Pronto, arquivo criado." if ok else "Pronto, arquivo criado."}], "end_turn", 10)  # bug: afirma sucesso mesmo com erro
        m = re.search(r"(?:file|arquivo)\s+(\S+\.txt).*?(?:with|contendo|com)\s+(.+)$", user, re.I | re.S)
        if m:
            name, content = m.group(1), m.group(2).strip().strip('"')
            if LESSON in system:
                return ChatResponse([{"type": "tool_use", "id": self._uid(messages), "name": "write_file", "input": {"path": name, "content": content}}], "tool_use", 20)
            return ChatResponse([{"type": "tool_use", "id": self._uid(messages), "name": "bash", "input": {"command": f"echo {content} > /{name}"}}], "tool_use", 20)
        return ChatResponse([{"type": "text", "text": f"Entendi: {user[:60]}"}], "end_turn", 5)

    def complete(self, role, system, user):
        self.calls[role] = self.calls.get(role, 0) + 1
        if role == "reflect":
            return LLMResponse(LESSON if re.search(r"arquivo|file", user, re.I) else "", 30)
        if role == "judge":
            return LLMResponse(json.dumps({"score": 1.0 if "WORKSPACE_HAS_FILE=true" in user else 0.0, "reason": "teste"}), 10)
        if role == "aux":
            return LLMResponse("Título de teste", 5)
        return LLMResponse("", 1)

    def complete_tools(self, role, system, user, tools, tool_choice="auto"):
        return ToolResponse("", [], 1)
