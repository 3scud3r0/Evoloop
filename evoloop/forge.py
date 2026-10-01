"""Tool forging: LLM escreve ferramenta -> forma -> estático -> testes na sandbox -> juiz -> registro.
Porta para Python a ideia do AgentOS (framerslab/agentos, Apache-2.0: ForgeShapeValidator, SandboxedToolForge,
ForgeRejectionClassifier), reescrita — não é tradução linha a linha."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field

from .llm import LLM, extract_json
from .sandbox import Sandbox, static_check

FORGER_SYSTEM = ("You write tiny, pure, deterministic Python tools. Reply with ONE JSON object with keys: name, description, "
                 "applies_to (list of regex keywords that identify tasks this tool solves), input_schema{properties}, "
                 "output_schema{properties}, code (defines `def run(text): ...` returning a str; only math/re/string/itertools/"
                 "collections/functools/json/statistics/textwrap/unicodedata/operator/heapq/bisect imports), test_cases "
                 "(>=2 items {input:{text:...}, expected:...} with REAL inputs).")
JUDGE_SYSTEM = "You are a strict code reviewer for auto-generated tools. Answer APPROVE or 'REJECT: <reason>'."


def validate_shape(req: dict) -> list[str]:
    """Porta de validateForgeShape (AgentOS): reporta TODAS as violações de uma vez."""
    errs = []
    ip = (req.get("input_schema") or {}).get("properties")
    op = (req.get("output_schema") or {}).get("properties")
    if not isinstance(ip, dict) or not ip:
        errs.append("input_schema sem propriedades declaradas")
    if not isinstance(op, dict) or not op:
        errs.append("output_schema sem propriedades declaradas")
    tcs = req.get("test_cases")
    tcs = tcs if isinstance(tcs, list) else []
    if len(tcs) < 2:
        errs.append(f"são necessários >=2 test_cases, veio {len(tcs)}")
    empty = sum(1 for t in tcs if not isinstance(t, dict) or not isinstance(t.get("input"), dict) or not t["input"])
    if empty:
        errs.append(f"{empty} test_case(s) com input vazio")
    for k in ("name", "code", "applies_to"):
        if not req.get(k):
            errs.append(f"campo obrigatório ausente: {k}")
    return errs


@dataclass
class ForgeResult:
    ok: bool
    stage: str                       # ok | llm | shape | static | tests | judge
    errors: list = field(default_factory=list)
    tool: dict | None = None
    name: str = ""


class ToolForge:
    def __init__(self, llm: LLM, sandbox: Sandbox, max_attempts: int = 3):
        self.llm, self.sandbox, self.max_attempts = llm, sandbox, max_attempts
        self.rejections: Counter = Counter()      # estágio -> contagem (classificador de rejeição)
        self.forged = 0

    def submit(self, req: dict) -> ForgeResult:
        name = str(req.get("name", "?"))
        errs = validate_shape(req)
        if errs:
            return self._rej("shape", errs, name)
        errs = static_check(req["code"])
        if errs:
            return self._rej("static", errs, name)
        bad = []
        for i, tc in enumerate(req["test_cases"]):
            r = self.sandbox.run(req["code"], "run", tc["input"])
            if not r.ok:
                bad.append(f"teste {i}: {r.error}")
            elif str(r.value) != str(tc.get("expected")):
                bad.append(f"teste {i}: obtido {r.value!r} esperado {tc.get('expected')!r}")
        if bad:
            return self._rej("tests", bad, name)
        verdict = self.llm.complete("judge", JUDGE_SYSTEM, f"STATIC_OK=true\nTESTS_PASSED=true\nname={name}\ndescription={req.get('description','')}\n"
                                    f"code:\n{req['code']}").text.strip()
        if not verdict.upper().startswith("APPROVE"):
            return self._rej("judge", [verdict[:200]], name)
        self.forged += 1
        tool = {"spec": {k: req[k] for k in ("description", "applies_to", "input_schema", "output_schema", "test_cases") if k in req},
                "code": req["code"], "origin": "forge"}
        return ForgeResult(True, "ok", [], tool, name)

    def _rej(self, stage: str, errs: list, name: str) -> ForgeResult:
        self.rejections[stage] += 1
        return ForgeResult(False, stage, errs, None, name)

    def forge_for(self, instruction: str) -> ForgeResult:
        """Pede ao forger uma ferramenta para a instrução; laço de reparo com o feedback do estágio que falhou."""
        feedback, last = "", ForgeResult(False, "llm", ["não tentou"])
        for att in range(1, self.max_attempts + 1):
            try:
                txt = self.llm.complete("forger", FORGER_SYSTEM, f"### FORGE\nTask definition: {instruction}\nATTEMPT {att}\n{feedback}").text
                req = extract_json(txt)
            except Exception as e:            # noqa: BLE001
                last = self._rej("llm", [str(e)[:120]], "?")
                feedback = f"FEEDBACK: resposta inválida ({e})"
                continue
            last = self.submit(req)
            if last.ok:
                return last
            feedback = f"FEEDBACK: rejeitada no estágio {last.stage}: " + "; ".join(last.errors)[:400]
        return last
