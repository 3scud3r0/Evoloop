"""Red team do isolamento: cada camada é testada SEPARADAMENTE (o AST nunca é considerado fronteira)."""
import sys, unittest, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evoloop  # noqa: F401
from evoloop.isolation import (FirecrackerBackend, GVisorBackend, IsolationUnavailable, NamespaceBackend, NsjailBackend,
                               RlimitBackend, pick)
from evoloop.sandbox import Sandbox, static_check

AST_MUST_REJECT = {
    "operator.attrgetter (escape real da v0.1)": 'import operator\ndef run(text):\n    return operator.attrgetter("__class__")(())\n',
    "string.Formatter.get_field": 'import string\ndef run(text):\n    return string.Formatter().get_field("0.x", (1,), {})\n',
    "from string import Formatter": 'from string import Formatter\ndef run(text):\n    return 1\n',
    "str.format com dunder": 'def run(text):\n    return "{0.__class__}".format(text)\n',
    "dunder attr": 'def run(text):\n    return ().__class__\n',
    "gi_frame": 'def run(text):\n    g = (i for i in [1])\n    return g.gi_frame\n',
    "f_globals": 'def run(text):\n    return run.f_globals\n',
    "getattr": 'def run(text):\n    return getattr(text, "upper")()\n',
    "import os": 'import os\ndef run(text):\n    return os.getcwd()\n',
    "open": 'def run(text):\n    return open("/etc/passwd").read()\n',
    "type()": 'def run(text):\n    return type(text)\n',
    "string dunder": 'def run(text):\n    return "__class__"\n',
}

HOSTILE_RAW = r'''
import os, json, socket, subprocess, ctypes
res = {}
def t(name, f):
    try: res[name] = "ALLOWED " + repr(f())[:40]
    except BaseException as e: res[name] = "BLOCKED " + type(e).__name__
t("read /etc/passwd", lambda: open("/etc/passwd").read())
t("read /home", lambda: os.listdir("/home"))
t("write /usr", lambda: open("/usr/x", "w").write("x"))
t("net external", lambda: socket.create_connection(("1.1.1.1", 80), 2))
t("net loopback", lambda: socket.create_connection(("127.0.0.1", 22), 2))
t("fork", lambda: os.fork())
t("exec", lambda: os.execv("/usr/bin/true", ["true"]))
t("subprocess", lambda: subprocess.run(["true"]))
t("mount", lambda: (_ for _ in ()).throw(OSError(ctypes.get_errno())) if ctypes.CDLL(None, use_errno=True).mount(b"none", b"/tmp", b"tmpfs", 0, None) != 0 else "mounted")
t("kill init", lambda: os.kill(1, 9))
res["pids visible"] = [p for p in os.listdir("/proc") if p.isdigit()]
print(json.dumps(res))
'''


class AST(unittest.TestCase):
    def test_known_escapes_rejected(self):
        for name, code in AST_MUST_REJECT.items():
            self.assertTrue(static_check(code), name)

    def test_legit_tools_pass(self):
        from evoloop.sim import TOOL_CODE
        for f, c in TOOL_CODE.items():
            self.assertEqual(static_check(c), [], f)


class Contain(unittest.TestCase):
    def _attack(self, backend):
        r = backend.run_python(HOSTILE_RAW, "", 10, 256)
        self.assertEqual(r.rc, 0, r.err[-300:])
        return json.loads(r.out.strip().splitlines()[-1])

    @unittest.skipUnless(NamespaceBackend.available(), "user namespaces indisponíveis")
    def test_namespace_backend(self):
        res = self._attack(NamespaceBackend())
        for k in ("read /etc/passwd", "read /home", "write /usr", "net external", "net loopback", "fork", "exec", "subprocess", "mount"):
            self.assertTrue(res[k].startswith("BLOCKED"), (k, res[k]))
        self.assertEqual(res["pids visible"], ["1"])

    def test_rlimit_backend_still_has_seccomp(self):
        res = self._attack(RlimitBackend())
        for k in ("fork", "exec", "subprocess", "net external"):
            self.assertTrue(res[k].startswith("BLOCKED"), (k, res[k]))
        # honestidade: o nível 0 NÃO isola o sistema de arquivos do host
        self.assertTrue(res["read /etc/passwd"].startswith("ALLOWED"))

    def test_fail_closed(self):
        with self.assertRaises(IsolationUnavailable):
            Sandbox(backend="rlimit", require_isolation=True)

    def test_cannot_pick_unavailable(self):
        with self.assertRaises(IsolationUnavailable):
            pick("firecracker")


class Builders(unittest.TestCase):
    def test_gvisor_command(self):
        c = GVisorBackend().command("print(1)", 3, 256)
        for flag in ("--runtime=runsc", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"):
            self.assertIn(flag, c)

    def test_nsjail_command(self):
        c = NsjailBackend().command("print(1)", 3, 256)
        self.assertIn("--rlimit_nproc", c); self.assertNotIn("--disable_clone_newnet", c)

    def test_firecracker_config(self):
        cfg = FirecrackerBackend.build_config("/k", "/r")
        self.assertEqual(cfg["network-interfaces"], [])
        self.assertTrue(cfg["drives"][0]["is_read_only"])
        with self.assertRaises(IsolationUnavailable):
            FirecrackerBackend().run_python("1", "", 1, 64)


if __name__ == "__main__":
    unittest.main()
