"""Execução de código forjado em DUAS camadas independentes:
  1) filtro estático (AST) — barato, pega o óbvio, NUNCA é fronteira de segurança (strings dinâmicas sempre o contornam);
  2) backend de isolamento (isolation.py) — é aqui que mora a segurança: namespaces+seccomp (verificado), nsjail/gVisor/Firecracker (não executados).
`require_isolation=True` (ou EVOLOOP_REQUIRE_ISOLATION=1) = falha fechada: recusa rodar se só houver rlimit.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import time
from dataclasses import dataclass

from .isolation import Backend, IsolationUnavailable, pick

ALLOWED_IMPORTS = {"math", "re", "string", "itertools", "collections", "functools", "json", "statistics",
                   "textwrap", "unicodedata", "heapq", "bisect"}               # `operator` REMOVIDO (attrgetter contornava o filtro de atributos)
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
                "breakpoint", "setattr", "delattr", "getattr", "memoryview", "exit", "quit", "help", "dir", "print", "type", "object"}
BANNED_ATTRS = {"gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame", "ag_code", "tb_frame", "tb_next",
                "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "co_code", "format", "format_map", "vformat", "get_field",
                "get_value", "Formatter", "mro", "attrgetter", "itemgetter", "methodcaller"}
BANNED_IMPORT_NAMES = {"Formatter", "attrgetter", "methodcaller"}
MAX_CODE = 20_000
MARK = "\x00__EVOLOOP_RESULT__\x00"
SAFE_BUILTINS = ("abs all any bin bool bytes chr dict divmod enumerate filter float format frozenset hash hex int "
                 "isinstance issubclass iter len list map max min next oct ord pow range repr reversed round set slice "
                 "sorted str sum tuple zip True False None Exception ValueError TypeError KeyError IndexError "
                 "ZeroDivisionError StopIteration ArithmeticError AttributeError").split()

_RUNNER = r'''
import sys, json, builtins
req = json.loads(sys.stdin.read())
ALLOWED = set(req["allowed"])
def _imp(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError("import proibido: " + name)
    return __import__(name, globals, locals, fromlist, level)
safe = {k: getattr(builtins, k) for k in req["safe"]}
safe["__import__"] = _imp
ns = {"__builtins__": safe, "__name__": "forged"}
exec(compile(req["code"], "<forged>", "exec"), ns)
out = ns[req["func"]](**req["args"])
sys.stdout.write(req["mark"] + json.dumps(out))
'''


def static_check(code: str) -> list[str]:
    """Lista de violações (vazia = ok). Heurística: não prova segurança."""
    errs: list[str] = []
    if len(code) > MAX_CODE:
        return [f"código maior que {MAX_CODE} caracteres"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"erro de sintaxe: {e.msg} (linha {e.lineno})"]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errs.append(f"import proibido: {a.name}")
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_IMPORTS or n.level:
                errs.append(f"import proibido: {n.module}")
            for a in n.names:
                if a.name in BANNED_IMPORT_NAMES or a.name.startswith("_"):
                    errs.append(f"nome importado proibido: {a.name}")
        elif isinstance(n, ast.Name) and n.id in BANNED_NAMES:
            errs.append(f"nome proibido: {n.id}")
        elif isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in BANNED_ATTRS):
            errs.append(f"atributo proibido: .{n.attr}")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and ("__" in n.value or n.value in BANNED_ATTRS):
            errs.append("string suspeita (dunder/introspecção)")
        elif isinstance(n, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            errs.append(f"construção proibida: {type(n).__name__}")
    return sorted(set(errs))


@dataclass
class SandboxResult:
    ok: bool
    value: object = None
    error: str = ""
    elapsed_ms: float = 0.0
    cached: bool = False


class Sandbox:
    def __init__(self, timeout_s: float = 3.0, mem_mb: int = 256, cache: bool = True, backend: str | Backend = "auto",
                 require_isolation: bool | None = None):
        self.timeout_s, self.mem_mb = timeout_s, mem_mb
        self._cache: dict[str, SandboxResult] | None = {} if cache else None
        self.spawns = 0
        self.backend = backend if isinstance(backend, Backend) else pick(backend)
        if require_isolation is None:
            require_isolation = os.environ.get("EVOLOOP_REQUIRE_ISOLATION") == "1"
        if require_isolation and self.backend.level < 1:
            raise IsolationUnavailable(f"isolamento exigido, mas o melhor backend disponível é '{self.backend.name}' (nível {self.backend.level})")

    @property
    def isolation(self) -> dict:
        return self.backend.describe()

    def run(self, code: str, func: str, args: dict) -> SandboxResult:
        key = hashlib.sha256(json.dumps([code, func, args, self.backend.name], sort_keys=True).encode()).hexdigest()
        if self._cache is not None and key in self._cache:
            r = self._cache[key]
            return SandboxResult(r.ok, r.value, r.error, r.elapsed_ms, True)
        errs = static_check(code)
        if errs:
            return SandboxResult(False, error="estático: " + "; ".join(errs))
        payload = json.dumps({"code": code, "func": func, "args": args, "allowed": sorted(ALLOWED_IMPORTS), "safe": SAFE_BUILTINS, "mark": MARK})
        t0 = time.perf_counter()
        self.spawns += 1
        p = self.backend.run_python(_RUNNER, payload, self.timeout_s, self.mem_mb)
        ms = (time.perf_counter() - t0) * 1e3
        if p.timed_out:
            res = SandboxResult(False, error=f"timeout>{self.timeout_s}s", elapsed_ms=ms)
        elif p.rc == 0 and MARK in p.out:
            try:
                res = SandboxResult(True, json.loads(p.out.split(MARK, 1)[1][:65536]), elapsed_ms=ms)
            except Exception as e:                      # noqa: BLE001
                res = SandboxResult(False, error=f"saída inválida: {e}", elapsed_ms=ms)
        else:
            tail = (p.err or "").strip().splitlines()[-1:] or [f"rc={p.rc}"]
            res = SandboxResult(False, error=f"runtime: {tail[0][:200]}", elapsed_ms=ms)
        if self._cache is not None:
            self._cache[key] = res
        return res
