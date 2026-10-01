"""Ambientes de execução do AGENTE GERAL (shell/arquivos/GUI). Diferente do sandbox.py (que roda só funções Python puras forjadas),
aqui há shell com binários, então a fronteira de segurança TEM que ser o ambiente:

  NamespaceEnv  nível 1, VERIFICADO aqui: user/pid/net/mount/ipc/uts ns, raiz tmpfs (/usr,/lib ro), só o workspace é gravável, sem rede,
                caps zeradas. Compartilha o kernel do host e NÃO filtra syscalls (sem seccomp: bash precisa de fork/exec).
  DockerEnv     o desenho do blueprint (contêiner persistente + só o Workspace montado). Comandos gerados e testados por unidade;
                NUNCA executado aqui (sem docker). Suporta gVisor (--runtime=runsc) por flag.
  UnsafeLocalEnv  só para testes; exige EVOLOOP_ALLOW_UNSAFE=1. Não é exposto na CLI.
"""
from __future__ import annotations
import base64
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .isolation import NamespaceBackend, IsolationUnavailable

MAX_OUT = 20_000


@dataclass
class ExecResult:
    rc: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def render(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("[timeout]")
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append("[stderr]\n" + self.stderr)
        parts.append(f"[exit {self.rc}]")
        return "\n".join(parts)


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[:MAX_OUT // 2] + f"\n…[{len(s) - MAX_OUT} chars omitidos]…\n" + s[-MAX_OUT // 2:]


def _run(cmd, stdin, timeout, **kw) -> ExecResult:
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, errors="replace", **kw)
        return ExecResult(p.returncode, _clip(p.stdout), _clip(p.stderr))
    except subprocess.TimeoutExpired as e:
        return ExecResult(-9, _clip((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")), "timeout", True)


class ExecEnv:
    name, level, verified = "base", -1, False
    workspace: Path

    def exec(self, cmd: str, timeout: float = 60, stdin: str = "") -> ExecResult:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"name": self.name, "level": self.level, "verified_here": self.verified, "workspace": str(self.workspace)}

    # ---- arquivos via exec (agnóstico ao ambiente); caminhos confinados a /workspace ------------------------------
    @staticmethod
    def safe_path(p: str) -> str:
        q = os.path.normpath(os.path.join("/workspace", p))
        if q != "/workspace" and not q.startswith("/workspace/"):
            raise ValueError(f"caminho fora de /workspace: {p}")
        return q

    def read_file(self, path: str, max_bytes: int = 200_000) -> ExecResult:
        return self.exec(f"head -c {int(max_bytes)} {shlex.quote(self.safe_path(path))}")

    def write_file(self, path: str, content: str) -> ExecResult:
        q = shlex.quote(self.safe_path(path))
        return self.exec(f"mkdir -p \"$(dirname {q})\" && base64 -d > {q}", stdin=base64.b64encode(content.encode()).decode())


_NS_SHELL = r"""
set -e
R=$(mktemp -d /tmp/evo.XXXXXX)
mount -t tmpfs -o size=4m,mode=755 tmpfs "$R"
for d in usr lib lib64 lib32 bin sbin; do
  if [ -L "/$d" ]; then ln -s "$(readlink "/$d")" "$R/$d"
  elif [ -d "/$d" ]; then mkdir "$R/$d"; mount --rbind "/$d" "$R/$d"; mount -o remount,bind,ro "$R/$d" 2>/dev/null || true; fi
done
mkdir "$R/proc" "$R/tmp" "$R/dev" "$R/workspace"
mount -t proc proc "$R/proc"
mount -t tmpfs -o size=256m,nosuid,nodev tmpfs "$R/tmp"
for n in null urandom zero; do : > "$R/dev/$n"; mount --bind "/dev/$n" "$R/dev/$n"; done
mount --bind "$WS" "$R/workspace"
exec "$CH" "$R" "$SP" --bounding-set=-all --inh-caps=-all --no-new-privs /bin/sh -c 'cd /workspace && ulimit -t "$CPU" -v "$MEMKB" -f "$FSKB" 2>/dev/null; eval "$EVO_CMD"'
"""


class NamespaceEnv(ExecEnv):
    name, level, verified = "namespaces-shell", 1, True

    def __init__(self, workspace: str | Path, mem_mb: int = 1024, fsize_mb: int = 256):
        if not NamespaceBackend.available():
            raise IsolationUnavailable("user namespaces indisponíveis")
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mem_mb, self.fsize_mb = mem_mb, fsize_mb

    def exec(self, cmd, timeout=60, stdin=""):
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/tmp", "LANG": "C.UTF-8", "CH": shutil.which("chroot"),
               "SP": shutil.which("setpriv"), "WS": str(self.workspace), "EVO_CMD": cmd, "CPU": str(int(timeout) + 2),
               "MEMKB": str(self.mem_mb * 1024), "FSKB": str(self.fsize_mb * 1024)}
        return _run(["unshare", "-Urmpfin", "--kill-child", "sh", "-c", _NS_SHELL], stdin, timeout, env=env, start_new_session=True)


class DockerEnv(ExecEnv):
    """Contêiner persistente (blueprint): só `workspace` montado; sem rede por padrão; sem capabilities; non-root; limites de pids/mem/cpu."""
    name, level, verified = "docker", 2, False

    def __init__(self, workspace: str | Path, image: str = "evoloop-agent:latest", name: str = "evoloop-ws", network: str = "none",
                 runtime: str | None = None, mem: str = "2g", cpus: str = "2", display: bool = True):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.image, self.cname, self.network, self.runtime, self.mem, self.cpus, self.display = image, name, network, runtime, mem, cpus, display
        self.level = 3 if runtime == "runsc" else 2

    def run_command(self) -> list[str]:
        c = ["docker", "run", "-d", "--name", self.cname, "--network", self.network, "--cap-drop=ALL", "--security-opt=no-new-privileges",
             "--pids-limit=256", f"--memory={self.mem}", f"--cpus={self.cpus}", "--user", "1000:1000", "--tmpfs", "/tmp:size=512m",
             "-v", f"{self.workspace}:/workspace:rw", "-w", "/workspace"]
        if self.runtime:
            c += [f"--runtime={self.runtime}"]
        return c + [self.image, "sleep", "infinity"]

    def exec_command(self, cmd: str, timeout: float) -> list[str]:
        return ["docker", "exec", "-i", "-w", "/workspace", "-e", "DISPLAY=:1", self.cname, "timeout", "-k", "5", str(int(timeout)), "bash", "-lc", cmd]

    def ensure(self) -> None:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.cname], capture_output=True, text=True)
        if r.returncode == 0 and "true" in r.stdout:
            return
        subprocess.run(["docker", "rm", "-f", self.cname], capture_output=True)
        p = subprocess.run(self.run_command(), capture_output=True, text=True)
        if p.returncode != 0:
            raise IsolationUnavailable("docker run falhou: " + p.stderr[-300:])

    def exec(self, cmd, timeout=60, stdin=""):
        self.ensure()
        return _run(self.exec_command(cmd, timeout), stdin, timeout + 10)

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.cname], capture_output=True)


class UnsafeLocalEnv(ExecEnv):
    name, level = "UNSAFE-local", 0

    def __init__(self, workspace: str | Path, env: dict | None = None):
        if os.environ.get("EVOLOOP_ALLOW_UNSAFE") != "1":
            raise IsolationUnavailable("UnsafeLocalEnv exige EVOLOOP_ALLOW_UNSAFE=1 (apenas testes)")
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.extra = env or {}

    def exec(self, cmd, timeout=60, stdin=""):
        cmd = cmd.replace("/workspace", str(self.workspace))
        return _run(["bash", "-c", cmd], stdin, timeout, cwd=str(self.workspace), env={**os.environ, **self.extra})


def make_env(kind: str, workspace: str, **kw) -> ExecEnv:
    if kind == "docker":
        e = DockerEnv(workspace, **kw)
        e.ensure()
        return e
    if kind == "namespaces":
        return NamespaceEnv(workspace)
    if kind == "auto":
        if shutil.which("docker"):
            try:
                return make_env("docker", workspace, **kw)
            except Exception:                               # noqa: BLE001
                pass
        return NamespaceEnv(workspace)
    raise ValueError(kind)
