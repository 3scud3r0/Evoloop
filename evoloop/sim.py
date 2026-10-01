"""SimLLM: simulador DETERMINÍSTICO de um LLM congelado. Valida o MECANISMO do ciclo, NÃO capacidade real.

Modelo do mundo (explícito, para não haver ilusão): p(acerto) = base da família; sobe para 0.90 se o texto de
sistema (prompt/regra/lição) "cobre" a família (chaves regex); +0.12 se há regra genérica (passo-a-passo + verificar).
O ruído é semeado por (tarefa, tentativa, hash do contexto) => harnesses diferentes têm sorteios independentes.
"""
from __future__ import annotations
import hashlib
import json
import random
import re

from .llm import LLMResponse, ToolCall, ToolResponse
from .tasks import FAMILIES, covers, family_of_instruction

RULE_P, GENERIC_BONUS = 0.90, 0.12
GENERIC_RE = (r"step[- ]by[- ]step", r"verif|double-check|check")

# implementações "de biblioteca" que o forger simulado devolve (o LLM real escreveria as suas)
TOOL_CODE = {
    "reverse": "def run(text):\n    return text[::-1]\n",
    "upper_vowels": "def run(text):\n    return ''.join(c.upper() if c in 'aeiou' else c for c in text)\n",
    "digit_sum": "def run(text):\n    return str(sum(int(c) for c in text if c.isdigit()))\n",
    "sort_words": "def run(text):\n    return ' '.join(sorted(text.split()))\n",
    "count_vowels": "def run(text):\n    return str(sum(1 for c in text.lower() if c in 'aeiou'))\n",
    "caesar3": "def run(text):\n    return ''.join(chr((ord(c) - 97 + 3) % 26 + 97) if 'a' <= c <= 'z' else c for c in text)\n",
    "dedupe_words": ("def run(text):\n    out = []\n    for w in text.split():\n        if not out or out[-1] != w:\n"
                    "            out.append(w)\n    return ' '.join(out)\n"),
    "longest_word": "def run(text):\n    best = ''\n    for w in text.split():\n        if len(w) > len(best):\n            best = w\n    return best\n",
}


def _seed(*parts) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest())


def _tok(*s) -> int:
    return max(1, sum(len(x) for x in s) // 4)


def _perturb(expected: str, inp: str, rng: random.Random) -> str:
    kind = rng.choice(("swap", "swap", "drop", "same", "off1"))
    if kind == "drop" and len(expected) > 1:
        return expected[:-1]                       # quebra a forma (verificador fraco pega)
    if kind == "same" and inp != expected:
        return inp
    if kind == "off1" and expected.isdigit():
        return str(int(expected) + 1)              # preserva a forma
    if len(expected) > 2:
        i = rng.randrange(len(expected) - 1)
        return expected[:i] + expected[i + 1] + expected[i] + expected[i + 2:]
    return expected + "x"


class SimLLM:
    def __init__(self, seed: int = 0, overfit_rate: float = 0.25, bad_tool_rate: float = 0.25, generic_rate: float = 0.5,
                 dilution: float = 0.0, route_miss: float = 0.06, route_wrong: float = 0.04):
        # dilution: SUPOSIÇÃO de mundo (não medida) — cada família coberta além de 2 num mesmo contexto tira `dilution` de p
        # (interferência de instruções em prompts longos). Com dilution=0 sub-agentes NÃO trazem ganho de acerto no simulador.
        # route_miss/route_wrong: taxa com que o LLM decide não chamar / chamar a ferramenta errada (function calling não é perfeito).
        self.seed, self.overfit_rate, self.bad_tool_rate, self.generic_rate = seed, overfit_rate, bad_tool_rate, generic_rate
        self.dilution, self.route_miss, self.route_wrong = dilution, route_miss, route_wrong
        self.calls: dict[str, int] = {}

    def complete(self, role: str, system: str, user: str) -> LLMResponse:
        self.calls[role] = self.calls.get(role, 0) + 1
        t = getattr(self, "_" + role, None)
        if t is None:
            raise ValueError(f"papel desconhecido no SimLLM: {role}")
        text = t(system, user)
        return LLMResponse(text, _tok(system, user, text) + 5)

    # ---- política: resolve a tarefa com probabilidade dependente do harness --------------------------------
    def _policy(self, system: str, user: str) -> str:
        return self._answer(system, user, "")

    def _answer(self, system: str, user: str, salt: str) -> str:
        fam = family_of_instruction(user)
        m = re.search(r"### TASK (\S+)", user)
        tid = m.group(1) if m else "?"
        inp = (re.search(r"^INPUT: (.*)$", user, re.M) or [None, ""])[1]
        att = int((re.search(r"ATTEMPT (\d+)", user) or [None, "1"])[1])
        trial = int((re.search(r"TRIAL (\d+)", user) or [None, "0"])[1])
        if fam is None:
            return "n/a"
        F = FAMILIES[fam]
        p = F.base_p
        if covers(fam, system):
            p = max(p, RULE_P)
        if all(re.search(g, system.lower()) for g in GENERIC_RE):
            p = min(0.95, p + GENERIC_BONUS)
        if self.dilution:
            n_cov = sum(1 for f in FAMILIES if covers(f, system))
            p = max(0.05, p - self.dilution * max(0, n_cov - 2))
        rng = _seed("policy" + salt, self.seed, tid, trial, att, hashlib.sha256(system.encode()).hexdigest())
        exp = F.solve(inp)
        return exp if rng.random() < p else _perturb(exp, inp, rng)

    # ---- QA (revisor independente): re-resolve com sorteio independente; aceita se as duas respostas coincidem -------
    def _qa(self, system: str, user: str) -> str:
        given = (re.search(r"^ANSWER: (.*)$", user, re.M) or [None, ""])[1].strip()
        other = self._answer(system, user, "qa").strip()
        return "OK" if other == given else "REJECT: independent re-derivation differs"

    # ---- arquiteto: propõe um sub-agente especialista (JSON) --------------------------------------------------
    def _architect(self, system: str, user: str) -> str:
        try:
            req = json.loads(user.split("### REQUEST\n", 1)[1])
        except Exception:                               # noqa: BLE001
            return "{}"
        fams = []
        for x in req.get("weak", []):
            f = family_of_instruction(x)
            if f and f not in fams:
                fams.append(f)
        fams = fams[:2]
        if not fams:
            return "{}"
        name = "spec_" + "_".join(f[:6] for f in fams)
        defs = "\n".join(f'When a task says "{FAMILIES[f].instruction}" follow that definition exactly, character by character.' for f in fams)
        return json.dumps({"name": name, "description": "Handles: " + " | ".join(FAMILIES[f].instruction for f in fams),
                           "system_prompt": "You are a specialist. Think step by step and verify your answer.\n" + defs, "tools": []})

    # ---- function calling: o LLM decide se/qual ferramenta chamar ------------------------------------------------
    def complete_tools(self, role: str, system: str, user: str, tools: list, tool_choice: str = "auto") -> ToolResponse:
        self.calls[role] = self.calls.get(role, 0) + 1
        calls = getattr(self, "_tools_" + role)(system, user, tools)
        return ToolResponse("", calls, _tok(system, user, json.dumps(tools)) + 8)

    def _decide(self, role: str, user: str, tools: list, arg_of) -> list:
        fam = family_of_instruction(user)
        if fam is None or not tools:
            return []
        m = re.search(r"### TASK (\S+)", user)
        trial = (re.search(r"TRIAL (\d+)", user) or [None, "0"])[1]
        canon = FAMILIES[fam].instruction
        match = [t for t in tools if canon in t["description"]]
        rng = _seed(role, self.seed, m.group(1) if m else "?", trial, ",".join(sorted(t["name"] for t in tools)))
        r = rng.random()
        if r < self.route_miss:
            return []
        if r < self.route_miss + self.route_wrong:
            others = [t for t in tools if t not in match]
            return [ToolCall(rng.choice(others)["name"], arg_of(user))] if others else []
        return [ToolCall(match[0]["name"], arg_of(user))] if match else []

    def _tools_router(self, system: str, user: str, tools: list) -> list:
        return self._decide("route", user, tools, lambda u: {"text": (re.search(r"^INPUT: (.*)$", u, re.M) or [None, ""])[1]})

    def _tools_delegate(self, system: str, user: str, tools: list) -> list:
        return self._decide("deleg", user, tools, lambda u: {"task": u[:400]})

    def _planner(self, system: str, user: str) -> str:
        return "PLAN: 1) identify the exact transformation 2) apply it 3) verify the result against the definition"

    # ---- reflexão (GEPA): propõe novo texto de componente a partir de exemplos de falha --------------------
    def _reflect(self, system: str, user: str) -> str:
        m = re.search(r"### CURRENT\n(.*?)\n### EXAMPLES\n(.*)$", user, re.S)
        cur, ex = (m.group(1), m.group(2)) if m else ("", "[]")
        try:
            rows = json.loads(ex)
        except Exception:                               # noqa: BLE001
            rows = []
        rng = _seed("reflect", self.seed, user)
        fails = {}
        for r in rows:
            f = family_of_instruction(r.get("instruction", ""))
            if f:
                fails.setdefault(f, r)
        add = []
        for f, r in list(fails.items())[:2]:
            if not covers(f, cur):
                add.append(f'When a task says "{FAMILIES[f].instruction}" follow that definition exactly, character by character.')
        if not all(re.search(g, cur.lower()) for g in GENERIC_RE) and rng.random() < self.generic_rate:
            add.append("Think step by step and verify your answer before responding.")
        if fails and rng.random() < self.overfit_rate:   # edição "esperta" mas específica do benchmark (vazamento)
            r = next(iter(fails.values()))
            add.append(f"Example: input '{r.get('input','')}' -> '{r.get('expected','')}'.")
        return (cur + ("\n" if cur and add else "") + "\n".join(add)).strip()

    # ---- forger: escreve uma ferramenta (JSON) --------------------------------------------------------------
    def _forger(self, system: str, user: str) -> str:
        fam = family_of_instruction(user)
        att = int((re.search(r"ATTEMPT (\d+)", user) or [None, "1"])[1])
        if fam is None:
            return "{}"
        F = FAMILIES[fam]
        rng = _seed("forge", self.seed, fam, att, hashlib.sha256(system.encode()).hexdigest()[:8])
        code = TOOL_CODE[fam]
        flaw = None
        if rng.random() < self.bad_tool_rate:
            flaw = rng.choice(("syntax", "import", "test", "truncate"))
        tin = [F.gen(_seed("t", fam, i)) for i in range(3)]
        if flaw == "syntax":
            code = code.replace("return", "retrun ", 1).replace("def run(text):", "def run(text)")
        elif flaw == "import":
            code = "import os\n" + code
        elif flaw == "test":
            code = code.replace("return ", "return '~' + ", 1)
        elif flaw == "truncate":       # passa nos testes (entradas curtas) mas erra em entradas longas
            code = code.replace("def run(text):", "def run(text):\n    text = text[:9]", 1)
            tin = [x[:9] for x in tin]
        cases = [{"input": {"text": x}, "expected": F.solve(x)} for x in tin]
        kw = [k for k in F.keys]
        return json.dumps({"name": f"tool_{fam}", "description": F.instruction, "applies_to": kw,
                           "input_schema": {"properties": {"text": {"type": "string"}}},
                           "output_schema": {"properties": {"result": {"type": "string"}}},
                           "code": code, "test_cases": cases})

    # ---- juiz do forge -------------------------------------------------------------------------------------
    def _judge(self, system: str, user: str) -> str:
        if "STATIC_OK=true" in user and "TESTS_PASSED=true" in user:
            return "APPROVE"
        return "REJECT: relatório de verificação não passou"
