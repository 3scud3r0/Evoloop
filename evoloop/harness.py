"""Harness = tudo que envolve o LLM congelado: prompts, regras de roteamento, ferramentas, política de memória/config."""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass, field

SEED_COMPONENTS = {
    "system_prompt": "You are a helpful assistant. Solve the task and reply with only the answer.",
    "planner_prompt": "Plan briefly before answering.",
    "router_rules": "",
}
SEED_CONFIG = {"max_retries": 1, "use_tools": True, "use_planner": True, "memory_k": 2, "forge_on_demand": False,
               "router_mode": "regex", "orchestrate": False, "qa_agent": False}
TEXT_COMPONENTS = ("system_prompt", "planner_prompt", "router_rules")


@dataclass
class Harness:
    components: dict = field(default_factory=lambda: dict(SEED_COMPONENTS))
    tools: dict = field(default_factory=dict)       # nome -> {spec:{...}, code:str, origin:str}
    config: dict = field(default_factory=lambda: dict(SEED_CONFIG))
    note: str = ""
    subagents: dict = field(default_factory=dict)    # nome -> {description, system_prompt, tools:[...]}

    def digest(self) -> str:
        blob = json.dumps({"c": self.components, "t": self.tools, "k": self.config, "s": self.subagents}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def clone(self, note: str = "") -> "Harness":
        h = copy.deepcopy(self)
        h.note = note or self.note
        return h

    def to_json(self) -> str:
        return json.dumps({"components": self.components, "tools": self.tools, "config": self.config, "note": self.note, "subagents": self.subagents},
                          sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "Harness":
        d = json.loads(s)
        return cls(d["components"], d["tools"], {**SEED_CONFIG, **d["config"]}, d.get("note", ""), d.get("subagents", {}))

    def text_view(self) -> str:
        """Visão textual completa (usada para diffs/vazamento/tamanho)."""
        parts = [f"[{k}]\n{v}" for k, v in sorted(self.components.items())]
        parts += [f"[tool {n}]\n{t['code']}" for n, t in sorted(self.tools.items())]
        parts += [f"[subagent {n}]\n{a['system_prompt']}" for n, a in sorted(self.subagents.items())]
        return "\n".join(parts)
