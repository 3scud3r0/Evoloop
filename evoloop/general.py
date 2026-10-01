"""Agente geral (ReAct com function calling nativo) sobre o Harness evolutivo: system_prompt, macros forjadas, sub-agentes, memória."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field

from .env import ExecEnv
from .harness import Harness
from .memory import Memory
from .tools_os import NATIVE, SAFETY_PREAMBLE, ToolError, execute_native, run_macro

GENERAL_CONFIG = {"max_steps": 25, "max_wall_s": 600, "memory_k": 3, "gui": False}


def general_seed_harness() -> Harness:
    from .tools_os import SEED_SYSTEM_PROMPT
    h = Harness()
    h.components = {"system_prompt": SEED_SYSTEM_PROMPT, "planner_prompt": "", "router_rules": ""}
    h.config = {**GENERAL_CONFIG}
    h.note = "seed-general"
    return h


@dataclass
class TurnResult:
    text: str
    steps: int = 0
    tokens: int = 0
    status: str = "ok"                      # ok | max_steps | timeout | error
    transcript: list = field(default_factory=list)


def _norm_messages(history: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in history:
        role, text = m["role"], m["content"] if isinstance(m["content"], str) else str(m["content"])
        if role not in ("user", "assistant") or not text.strip():
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


class GeneralAgent:
    def __init__(self, harness: Harness, llm, env: ExecEnv, memory: Memory | None = None):
        self.h, self.llm, self.env, self.memory = harness, llm, env, memory

    # ---- ferramentas visíveis ao modelo --------------------------------------------------------------------------
    def tool_schemas(self, names: list[str] | None = None, allow_delegate: bool = True) -> list[dict]:
        sch = [NATIVE[n] for n in ("bash", "read_file", "write_file")]
        if self.h.config.get("gui"):
            sch += [NATIVE["screenshot"], NATIVE["computer"]]
        for n, t in sorted(self.h.tools.items()):
            if t.get("kind") == "macro":
                sch.append({"name": "macro_" + n, "description": t["spec"]["description"], "input_schema": t["spec"]["input_schema"]})
        if allow_delegate:
            for n, a in sorted(self.h.subagents.items()):
                sch.append({"name": "delegate_" + n, "description": a["description"],
                            "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}})
        return [s for s in sch if names is None or s["name"] in names] if names is not None else sch

    def system(self, user_text: str, extra: str = "") -> str:
        s = SAFETY_PREAMBLE + "\n" + self.h.components["system_prompt"]
        if extra:
            s += "\n" + extra
        k = int(self.h.config.get("memory_k", 0))
        if self.memory is not None and k and user_text:
            # memória pequena (<=12 lições): inclui TODAS (regras gerais de feedback quase nunca casam por palavra com o pedido); senão, recuperação
            les = [t for _, _, t in self.memory.items()] if len(self.memory) <= 12 else self.memory.recall(user_text, k)
            if les:
                s += "\nLESSONS LEARNED FROM PAST USER FEEDBACK (follow them):\n- " + "\n- ".join(les)
        return s

    def _exec_tool(self, name: str, a: dict, depth: int, on_event):
        if name.startswith("macro_"):
            m = self.h.tools.get(name[6:])
            if not m:
                raise ToolError("macro inexistente")
            return run_macro(self.env, m, a).render()
        if name.startswith("delegate_"):
            if depth > 0 or name[9:] not in self.h.subagents:
                raise ToolError("delegação inválida")
            sub = self.h.subagents[name[9:]]
            res = self._loop(self.system(a.get("task", ""), sub["system_prompt"]), [{"role": "user", "content": str(a.get("task", ""))}],
                             self.tool_schemas(sub.get("tools") or None, allow_delegate=False), 12, depth + 1, on_event, time.time() + 300)
            return res.text or f"(sub-agente terminou sem texto; status={res.status})"
        return execute_native(self.env, name, a)

    def _loop(self, system, messages, tools, max_steps, depth, on_event, deadline) -> TurnResult:
        res, msgs = TurnResult(""), list(messages)
        for step in range(max_steps):
            if time.time() > deadline:
                res.status, res.text = "timeout", res.text or "Tempo limite atingido antes de concluir."
                return res
            try:
                r = self.llm.chat(system, msgs, tools, "auto")
            except Exception as e:                      # noqa: BLE001
                res.status, res.text = "error", f"Erro ao chamar o modelo: {str(e)[:200]}"
                return res
            res.tokens += r.tokens; res.steps += 1
            if r.text.strip() and on_event:
                on_event("text", r.text)
            uses = r.tool_uses
            if not uses:
                res.text = r.text.strip() or res.text
                return res
            msgs.append({"role": "assistant", "content": r.blocks})
            results = []
            for u in uses:
                if on_event:
                    on_event("tool_use", {"name": u["name"], "input": u["input"]})
                try:
                    out, err = self._exec_tool(u["name"], u["input"], depth, on_event), False
                except Exception as e:                  # noqa: BLE001
                    out, err = f"ERROR: {type(e).__name__}: {str(e)[:300]}", True
                preview = out if isinstance(out, str) else "[imagem]"
                res.transcript.append({"tool": u["name"], "input": u["input"], "output": preview[:1500], "error": err})
                if on_event:
                    on_event("tool_result", {"name": u["name"], "output": preview[:400], "error": err})
                results.append({"type": "tool_result", "tool_use_id": u["id"], "content": out, "is_error": err})
            msgs.append({"role": "user", "content": results})
        res.status, res.text = "max_steps", (res.text or "") + "\n(Parei: limite de passos atingido sem conclusão.)"
        return res

    def run(self, history: list[dict], on_event=None) -> TurnResult:
        msgs = _norm_messages(history)
        if not msgs:
            return TurnResult("Não recebi nenhuma mensagem do usuário.", status="error")
        last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and isinstance(m["content"], str)), "")
        return self._loop(self.system(last_user), msgs, self.tool_schemas(), int(self.h.config.get("max_steps", 25)), 0, on_event,
                          time.time() + float(self.h.config.get("max_wall_s", 600)))
