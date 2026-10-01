"""Backends de isolamento para código forjado. Níveis: 0 rlimit · 1 namespaces Linux (user/pid/net/mount/ipc/uts + chroot ro +
caps zeradas) · 2 nsjail · 3 gVisor (docker --runtime=runsc) · 4 microVM Firecracker.

VERIFICADO NESTA MÁQUINA: apenas 0 e 1 (executados e atacados em tests/test_redteam.py).
NÃO EXECUTADO AQUI (sem nsjail/docker/runsc/KVM): 2 e 3 — a linha de comando é gerada e testada por unidade, nunca rodada.
Firecracker: só contrato + gerador de config; o agente-convidado (guest agent) NÃO foi implementado.
"""
from __future__ import annotations
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

RLIMIT_PRELUDE = """\
import resource as _r
_cpu = {cpu}
_r.setrlimit(_r.RLIMIT_CPU, (_cpu, _cpu))
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_FSIZE, (0, 0))
_r.setrlimit(_r.RLIMIT_NOFILE, (16, 16))
_r.setrlimit(_r.RLIMIT_CORE, (0, 0))
_r.setrlimit(_r.RLIMIT_NPROC, (0, 0))
del _r
"""


# seccomp-bpf (x86_64) instalado DENTRO do processo filho antes do código forjado: nega fork/vfork/clone-sem-THREAD/execve/socket/
# mount/ptrace/unshare/setns/bpf/io_uring/process_vm_*. clone3 devolve ENOSYS (glibc cai para clone). Em outras arquiteturas: sem filtro.
SECCOMP_PRELUDE = """\
import ctypes as _c, struct as _s, platform as _p
if _p.machine() == "x86_64":
    _deny = [57, 58, 59, 322, 41, 53, 42, 43, 49, 50, 51, 52, 101, 165, 166, 155, 161, 272, 308, 310, 311, 321, 425, 426, 427,
             250, 248, 249, 298, 304, 176, 175, 313, 246, 134, 139, 169, 167, 168, 163, 156]
    _ins = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xC000003E), (0x06, 0, 0, 0x80000000), (0x20, 0, 0, 0)]
    for _n in _deny:
        _ins += [(0x15, 0, 1, _n), (0x06, 0, 0, 0x00050000 | 1)]
    _ins += [(0x15, 0, 1, 435), (0x06, 0, 0, 0x00050000 | 38),
             (0x15, 0, 3, 56), (0x20, 0, 0, 16), (0x45, 1, 0, 0x10000), (0x06, 0, 0, 0x00050000 | 1), (0x06, 0, 0, 0x7fff0000)]
    _buf = b"".join(_s.pack("HBBI", *i) for i in _ins)
    _arr = _c.create_string_buffer(_buf, len(_buf))
    _prog = _s.pack("HL", len(_ins), _c.addressof(_arr))
    _pb = _c.create_string_buffer(_prog, len(_prog))
    _l = _c.CDLL(None, use_errno=True)
    _l.prctl.argtypes = [_c.c_int, _c.c_ulong, _c.c_ulong, _c.c_ulong, _c.c_ulong]
    _l.prctl(38, 1, 0, 0, 0)
    if _l.prctl(22, 2, _c.addressof(_pb), 0, 0) != 0:
        raise SystemExit("seccomp install failed")
    del _c, _s, _p, _l, _arr, _pb, _buf, _ins, _deny, _prog
"""


class IsolationUnavailable(RuntimeError):
    pass


@dataclass
class Proc:
    rc: int
    out: str
    err: str
    timed_out: bool = False


class Backend:
    name, level, verified = "base", -1, False

    @classmethod
    def available(cls) -> bool:
        return False

    def run_python(self, source: str, stdin: str, timeout: float, mem_mb: int) -> Proc:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "level": self.level, "verified_here": self.verified}


def _run(cmd, stdin, timeout, **kw) -> Proc:
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, **kw)
        return Proc(p.returncode, p.stdout, p.stderr)
    except subprocess.TimeoutExpired:
        return Proc(-9, "", "timeout", True)


class RlimitBackend(Backend):
    name, level, verified = "rlimit", 0, True

    @classmethod
    def available(cls):
        return True

    def run_python(self, source, stdin, timeout, mem_mb):
        cpu = int(timeout) + 1
        source = SECCOMP_PRELUDE + source

        def pre():
            os.setsid()
            for lim, v in ((resource.RLIMIT_CPU, (cpu, cpu)), (resource.RLIMIT_AS, (mem_mb << 20,) * 2), (resource.RLIMIT_FSIZE, (0, 0)),
                           (resource.RLIMIT_NOFILE, (16, 16)), (resource.RLIMIT_CORE, (0, 0)), (resource.RLIMIT_NPROC, (0, 0))):
                resource.setrlimit(lim, v)
        with tempfile.TemporaryDirectory() as td:
            return _run([sys.executable, "-I", "-S", "-c", source], stdin, timeout, cwd=td, env={}, preexec_fn=pre)


_NS_SCRIPT = r"""
set -e
R=$(mktemp -d /tmp/evo.XXXXXX)
mount -t tmpfs -o size=4m,mode=755 tmpfs "$R"
for d in usr lib lib64 lib32 bin sbin; do
  if [ -L "/$d" ]; then ln -s "$(readlink "/$d")" "$R/$d"
  elif [ -d "/$d" ]; then mkdir "$R/$d"; mount --rbind "/$d" "$R/$d"; mount -o remount,bind,ro "$R/$d" 2>/dev/null || true; fi
done
mkdir "$R/proc" "$R/tmp" "$R/dev"
mount -t proc proc "$R/proc"
mount -t tmpfs -o size=8m,nosuid,nodev,noexec tmpfs "$R/tmp"
for n in null urandom; do : > "$R/dev/$n"; mount --bind "/dev/$n" "$R/dev/$n"; done
exec "$CH" "$R" "$SP" --bounding-set=-all --inh-caps=-all --no-new-privs "$PY" -I -S -c "$1"
"""


class NamespaceBackend(Backend):
    """Sem rede (netns vazio), sem PIDs/mounts do host, raiz = tmpfs com /usr,/lib... somente-leitura, /tmp tmpfs noexec,
    capabilities zeradas + no_new_privs, rlimits aplicados dentro do processo. NÃO tem seccomp: syscalls de kernel continuam
    expostas (superfície de ataque ao kernel existe) — por isso nível 1, abaixo de gVisor/microVM."""
    name, level, verified = "namespaces", 1, True

    @classmethod
    @lru_cache(maxsize=1)
    def available(cls):
        if not (shutil.which("unshare") and shutil.which("setpriv") and shutil.which("chroot")):
            return False
        try:
            # Criar user namespaces pode ser permitido enquanto mounts necessários
            # continuam bloqueados (comum dentro de contêineres). Teste a operação
            # mínima que o backend realmente exige, não apenas ``unshare``.
            p = subprocess.run(["unshare", "-Urmpfin", "--kill-child", "sh", "-c",
                                "d=$(mktemp -d); mount -t tmpfs tmpfs $d && mkdir $d/p && mount -t proc proc $d/p "
                                "&& umount $d/p && umount $d && rmdir $d/p $d && echo ok"],
                               capture_output=True, text=True, timeout=5)
            return p.returncode == 0 and "ok" in p.stdout
        except Exception:                               # noqa: BLE001
            return False

    def run_python(self, source, stdin, timeout, mem_mb):
        pre = RLIMIT_PRELUDE.format(cpu=int(timeout) + 1, mem=mem_mb << 20) + SECCOMP_PRELUDE
        real = os.path.realpath(sys.executable)
        env = {"PY": real if real.startswith("/usr/") else "/usr/bin/python3", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
               "CH": shutil.which("chroot"), "SP": shutil.which("setpriv")}
        cmd = ["unshare", "-Urmpfin", "--kill-child", "sh", "-c", _NS_SCRIPT, "sh", pre + source]
        return _run(cmd, stdin, timeout, env=env, start_new_session=True)


class NsjailBackend(Backend):
    name, level = "nsjail", 2

    @classmethod
    def available(cls):
        return bool(shutil.which("nsjail"))

    def command(self, source: str, timeout: float, mem_mb: int) -> list[str]:
        return ["nsjail", "-Mo", "-q", "--user", "65534", "--group", "65534", "--time_limit", str(int(timeout) + 1),
                "--rlimit_as", str(mem_mb), "--rlimit_cpu", str(int(timeout) + 1), "--rlimit_fsize", "0", "--rlimit_nproc", "0",
                "--rlimit_nofile", "16", "-R", "/usr", "-R", "/lib", "-R", "/lib64", "-R", "/bin", "-T", "/tmp",
                "--", "/usr/bin/python3", "-I", "-S", "-c", source]       # netns novo por padrão no nsjail

    def run_python(self, source, stdin, timeout, mem_mb):
        return _run(self.command(source, timeout, mem_mb), stdin, timeout + 2)


class GVisorBackend(Backend):
    name, level = "gvisor-docker", 3

    @classmethod
    def available(cls):
        if not (shutil.which("docker") and shutil.which("runsc")):
            return False
        try:
            out = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"], capture_output=True, text=True, timeout=10).stdout
            return "runsc" in out
        except Exception:                               # noqa: BLE001
            return False

    def command(self, source: str, timeout: float, mem_mb: int, cname: str = "evoloop") -> list[str]:
        return ["docker", "run", "--rm", "-i", "--name", cname, "--runtime=runsc", "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit=16", f"--memory={mem_mb}m", "--cpus=1", "--user", "65534:65534",
                "--tmpfs", "/tmp:size=8m,noexec,nosuid", os.environ.get("EVOLOOP_IMAGE", "python:3.12-slim"),
                "python", "-I", "-S", "-c", source]

    def run_python(self, source, stdin, timeout, mem_mb):
        name = "evoloop-" + uuid.uuid4().hex[:10]
        r = _run(self.command(source, timeout, mem_mb, name), stdin, timeout + 5)
        if r.timed_out:
            subprocess.run(["docker", "kill", name], capture_output=True)
        return r


class FirecrackerBackend(Backend):
    """Contrato + gerador de config. NÃO executa: falta o agente-convidado (init no rootfs que lê o payload por vsock, roda o
    Python e devolve stdout) e a imagem kernel/rootfs. Pré-requisitos: /dev/kvm, binário `firecracker`, EVOLOOP_FC_KERNEL, EVOLOOP_FC_ROOTFS."""
    name, level = "firecracker", 4

    @classmethod
    def available(cls):
        return False          # propositalmente False até existir o agente-convidado

    @staticmethod
    def build_config(kernel: str, rootfs: str, mem_mb: int = 256, vcpus: int = 1) -> dict:
        return {"boot-source": {"kernel_image_path": kernel, "boot_args": "console=ttyS0 reboot=k panic=1 pci=off ro init=/sbin/guest-agent"},
                "drives": [{"drive_id": "rootfs", "path_on_host": rootfs, "is_root_device": True, "is_read_only": True}],
                "machine-config": {"vcpu_count": vcpus, "mem_size_mib": mem_mb, "smt": False},
                "network-interfaces": [],                 # sem NIC => sem rede
                "vsock": {"guest_cid": 3, "uds_path": "/tmp/evoloop-fc.vsock"}}

    def run_python(self, source, stdin, timeout, mem_mb):
        raise IsolationUnavailable("FirecrackerBackend: agente-convidado não implementado (ver docstring)")


ORDER = [FirecrackerBackend, GVisorBackend, NsjailBackend, NamespaceBackend, RlimitBackend]
BY_NAME = {"rlimit": RlimitBackend, "namespaces": NamespaceBackend, "nsjail": NsjailBackend, "gvisor": GVisorBackend, "firecracker": FirecrackerBackend}


def pick(name: str = "auto") -> Backend:
    if name != "auto":
        cls = BY_NAME[name]
        if not cls.available():
            raise IsolationUnavailable(f"backend '{name}' indisponível nesta máquina")
        return cls()
    for cls in ORDER:
        if cls.available():
            return cls()
    return RlimitBackend()
