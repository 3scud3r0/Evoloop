"""Núcleo cognitivo: memória -> planner -> router -> (ferramenta | LLM congelado) -> verificador (com retry)."""
from __future__ import annotations
import re

from .experience import Episode
from .forge import ToolForge
from .harness import Harness
from .llm import LLM
from .memory import Memory
from .sandbox import Sandbox
from .swarm import ORCH_SYSTEM, QA_SYSTEM, delegate_schemas, tool_schema, validate_args
from .tasks import Task


def route(instruction: str, tools: dict, config: dict, rules: str) -> tuple[str, str | None]:
    """Router: regras explícitas do harness ('route: <regex> -> tool:NOME|llm') > casamento das ferramentas > llm."""
    if not config.get("use_tools", True):
        return "llm", None
    for line in rules.splitlines():
        m = re.match(r"\s*route:\s*(.+?)\s*->\s*(?:tool:(\S+)|(llm))\s*$", line)
        if m and re.search(m.group(1), instruction, re.I):
            return ("tool", m.group(2)) if m.group(2) in tools else ("llm", None)
    low = instruction.lower()
    for name, t in sorted(tools.items()):
        keys = t["spec"].get("applies_to", [])
        if keys and all(re.search(k, low) for k in keys):
            return "tool", name
    return "llm", None


class Verifier:
    """Verificador independente do ator (só vê tarefa+resposta). Propositalmente FRACO aqui (checa só forma):
    o avaliador usa o oráculo estrito; a diferença entre os dois é o espaço do Goodhart e é reportada."""
    def verify(self, task: Task, answer: str) -> tuple[bool, str]:
        ok = task.shape_ok(answer)
        return ok, "" if ok else "answer has the wrong shape for this task"


class Agent:
    def __init__(self, harness: Harness, llm: LLM, memory: Memory, sandbox: Sandbox, forge: ToolForge | None = None, verifier: Verifier | None = None):
        self.h, self.llm, self.memory, self.sandbox, self.forge = harness, llm, memory, sandbox, forge
        self.verifier = verifier or Verifier()
        self.tools = dict(harness.tools)          # cópia local: forge sob demanda não vaza para o harness
        self._no_forge: set = set()
        self.stats = {"route_llm_calls": 0, "route_tool_calls": 0, "route_invalid": 0, "delegations": 0, "qa_calls": 0, "qa_rejects": 0}

    def _forge_on_demand(self, task: Task, trace: list) -> None:
        key = task.instruction
        if key in self._no_forge or self.forge is None:
            return
        res = self.forge.forge_for(task.instruction)
        trace.append(f"forge_on_demand:{res.stage}")
        if res.ok:
            self.tools[res.name] = res.tool
        else:
            self._no_forge.add(key)

    # ---- orquestrador: delega a um sub-agente via function calling -----------------------------------------------
    def _delegate(self, task: Task, trial: int, trace: list) -> tuple[str | None, int]:
        schemas = delegate_schemas(self.h.subagents)
        r = self.llm.complete_tools("delegate", ORCH_SYSTEM, f"{task.prompt()}\nTRIAL {trial}", schemas)
        self.stats["delegations"] += 1
        for c in r.calls:
            n = c.name.removeprefix("delegate_")
            if n in self.h.subagents:
                trace.append(f"delegate:{n}")
                return n, r.tokens
        trace.append("delegate:none")
        return None, r.tokens

    # ---- router: regex | llm (function calling nativo) | hybrid -------------------------------------------------------
    def _route(self, task: Task, trial: int, tools: dict, trace: list) -> tuple[str, str | None, dict, int]:
        h, mode, toks = self.h, self.h.config.get("router_mode", "regex"), 0
        if not h.config.get("use_tools", True) or not tools:
            return "llm", None, {}, 0
        if mode in ("regex", "hybrid"):
            kind, name = route(task.instruction, tools, h.config, h.components.get("router_rules", ""))
            if kind == "tool":
                return "tool", name, {"text": task.input}, 0
            if mode == "regex":
                return "llm", None, {}, 0
        schemas = [tool_schema(n, t) for n, t in sorted(tools.items())]
        r = self.llm.complete_tools("router", "Decide whether one of the tools solves the task; if so call it with the task input. Otherwise answer directly.",
                                    f"{task.prompt()}\nTRIAL {trial}", schemas, "auto")
        self.stats["route_llm_calls"] += 1
        toks = r.tokens
        for c in r.calls:
            sch = next((x for x in schemas if x["name"] == c.name), None)
            if sch is None or validate_args(sch, c.input):
                self.stats["route_invalid"] += 1
                trace.append(f"route_llm:invalid:{c.name}")
                continue
            self.stats["route_tool_calls"] += 1
            return "tool", c.name, c.input, toks
        trace.append("route_llm:none")
        return "llm", None, {}, toks

    def _qa(self, task: Task, ctx: str, ans: str, trial: int, att: int) -> tuple[bool, int]:
        r = self.llm.complete("qa", ctx + "\n" + QA_SYSTEM, f"{task.prompt()}\nANSWER: {ans}\nTRIAL {trial}\nATTEMPT {att}")
        self.stats["qa_calls"] += 1
        ok = r.text.strip().upper().startswith("OK")
        self.stats["qa_rejects"] += (not ok)
        return ok, r.tokens

    def run(self, task: Task, trial: int) -> Episode:
        h, tokens, trace, route_tag = self.h, 0, [], "llm"
        system, tools = h.components["system_prompt"], self.tools
        if h.config.get("orchestrate") and h.subagents:
            sub, tk = self._delegate(task, trial, trace)
            tokens += tk
            if sub:
                spec = h.subagents[sub]
                system = spec["system_prompt"]
                tools = {n: t for n, t in self.tools.items() if n in spec.get("tools", [])}
                route_tag = f"sub:{sub}/"
        kind, tool, targs, tk = self._route(task, trial, tools, trace)
        tokens += tk
        if kind == "llm" and h.config.get("forge_on_demand") and h.config.get("use_tools", True) and not route_tag.startswith("sub:"):
            self._forge_on_demand(task, trace)
            tools = self.tools
            kind, tool, targs, tk = self._route(task, trial, tools, trace)
            tokens += tk
        qa_on = bool(h.config.get("qa_agent"))
        if kind == "tool":
            r = self.sandbox.run(tools[tool]["code"], "run", targs)
            ans = str(r.value) if r.ok else ""
            ok, why = (self.verifier.verify(task, ans) if r.ok else (False, r.error))
            if ok and qa_on:
                ok, tk = self._qa(task, system, ans, trial, 0)
                tokens += tk
                why = "" if ok else "qa rejected"
            trace.append(f"tool:{tool}:{'ok' if ok else 'fail:' + why[:60]}")
            if ok:
                return Episode(task.id, task.family, trial, ans, 0.0, True, 1, tokens, route_tag + f"tool:{tool}", trace, task.instruction, task.input, task.expected)
        lessons = self.memory.recall(task.instruction, h.config.get("memory_k", 0)) if h.config.get("memory_k") else []
        ctx = system
        if h.config.get("use_planner"):
            p = self.llm.complete("planner", h.components["planner_prompt"], task.prompt())
            tokens += p.tokens
            ctx += "\n" + h.components["planner_prompt"]
        if lessons:
            ctx += "\nMEMORY:\n" + "\n".join(lessons)
        ans, ok, feedback, att = "", False, "", 0
        for att in range(1, 2 + int(h.config.get("max_retries", 0))):
            r = self.llm.complete("policy", ctx, f"{task.prompt()}\nTRIAL {trial}\nATTEMPT {att}\n{feedback}")
            tokens += r.tokens
            ans = r.text.strip()
            ok, why = self.verifier.verify(task, ans)
            if ok and qa_on:
                ok, tk = self._qa(task, ctx, ans, trial, att)
                tokens += tk
                why = "" if ok else "independent reviewer disagrees"
            trace.append(f"llm#{att}:{'ok' if ok else 'reject:' + why[:40]}")
            if ok:
                break
            feedback = f"FEEDBACK: previous answer rejected: {why}"
        return Episode(task.id, task.family, trial, ans, 0.0, ok, att, tokens, route_tag + "llm", trace, task.instruction, task.input, task.expected)
