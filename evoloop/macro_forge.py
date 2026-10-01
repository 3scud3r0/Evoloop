"""Forja de MACROS para o agente geral: o LLM escreve um script (bash/python) que roda DENTRO do ExecEnv; só vira ferramenta se passar
no teste que ele mesmo declarou, executado num workspace descartável. Roda apenas na consolidação (offline), nunca durante um pedido
do usuário — conteúdo não confiável (web/arquivos) não consegue induzir criação de ferramentas persistentes."""
from __future__ import annotations
import re
import tempfile

from .env import NamespaceEnv
from .llm import extract_json
from .tools_os import run_macro

FORGE_SYSTEM = ("You write a reusable macro for a sandboxed Linux agent. Reply with ONE JSON object: name (snake_case), description, "
                "input_schema (JSON schema, type object), interpreter ('bash' or 'python'), script (reads its arguments from the JSON in env var "
                "EVO_ARGS; works only under /workspace; no network), test: {setup_files: {path: content}, args: {...}, expect_stdout_regex: str|null, "
                "expect_files: {path: regex}}. The test must be realistic and verify the behavior.")
LINT = [r"rm\s+-[a-z]*r[a-z]*f?\s+/(\s|$|\*)", r"\bmkfs", r"\bdd\b.*\bof=/dev", r":\(\)\s*\{", r"(curl|wget)[^|\n]*\|\s*(ba)?sh", r"\bsudo\b",
        r"chmod\s+-R\s+7?77\s+/", r">\s*/dev/sd", r"/etc/(passwd|shadow)", r"\bnc\b.*-e", r"/dev/tcp/"]


def validate_macro(spec: dict, existing: dict) -> list[str]:
    errs = []
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", str(spec.get("name", ""))):
        errs.append("nome inválido")
    if spec.get("name") in existing:
        errs.append("nome já existe")
    if spec.get("interpreter") not in ("bash", "python"):
        errs.append("interpreter deve ser bash|python")
    sc = str(spec.get("script", ""))
    if not sc.strip() or len(sc) > 6000:
        errs.append("script vazio ou > 6000 chars")
    for pat in LINT:
        if re.search(pat, sc):
            errs.append(f"padrão perigoso no script: {pat}")
    if not isinstance(spec.get("input_schema"), dict) or spec["input_schema"].get("type") != "object":
        errs.append("input_schema deve ser objeto JSON-schema")
    t = spec.get("test") or {}
    if not (t.get("expect_stdout_regex") or t.get("expect_files")):
        errs.append("teste sem expectativa verificável")
    return errs


def run_macro_test(spec: dict) -> list[str]:
    import os
    t, errs = spec["test"], []
    with tempfile.TemporaryDirectory() as td:
        env = NamespaceEnv(td)
        for path, content in (t.get("setup_files") or {}).items():
            r = env.write_file(path, str(content))
            if r.rc != 0:
                return [f"setup falhou: {r.render()[:120]}"]
        macro = {"interpreter": spec["interpreter"], "script": spec["script"], "timeout": 60}
        r = run_macro(env, macro, t.get("args") or {})
        if r.rc != 0:
            errs.append("teste: " + r.render()[-200:])
        if t.get("expect_stdout_regex") and not re.search(t["expect_stdout_regex"], r.stdout):
            errs.append(f"stdout não casa {t['expect_stdout_regex']!r}: {r.stdout[-120:]!r}")
        for path, rx in (t.get("expect_files") or {}).items():
            fp = os.path.join(td, os.path.normpath(path).lstrip("/"))
            if not os.path.isfile(fp) or not re.search(rx, open(fp, errors="replace").read()):
                errs.append(f"arquivo {path} ausente ou sem conteúdo esperado")
    return errs


def forge_macro(llm, request: str, existing: dict, attempts: int = 2) -> tuple[dict | None, list[str]]:
    feedback, errs = "", ["não tentou"]
    for _ in range(attempts):
        try:
            spec = extract_json(llm.complete("forger", FORGE_SYSTEM, f"### REQUEST\n{request}\n{feedback}").text)
        except Exception as e:                          # noqa: BLE001
            errs, feedback = [f"resposta inválida: {str(e)[:80]}"], "FEEDBACK: invalid JSON"
            continue
        errs = validate_macro(spec, existing) or run_macro_test(spec)
        if not errs:
            ver = llm.complete("judge", "Strict reviewer of auto-generated shell/python macros. Reply APPROVE or 'REJECT: reason'.",
                               f"STATIC_OK=true\nTESTS_PASSED=true\nname={spec['name']}\n{spec['description']}\n{spec['script']}").text.strip()
            if ver.upper().startswith("APPROVE"):
                tool = {"kind": "macro", "spec": {"description": spec["description"], "input_schema": spec["input_schema"]},
                        "code": spec["script"], "script": spec["script"], "interpreter": spec["interpreter"], "timeout": 120, "origin": "macro-forge"}
                return {"name": spec["name"], "tool": tool}, []
            errs = [ver[:150]]
        feedback = "FEEDBACK: " + "; ".join(errs)[:400]
    return None, errs
