import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evoloop  # noqa: F401
from raven_gates import TaskEval, fisher_one_sided, paired_lift
from rrsi_core import edit_budget
from evoloop.forge import ToolForge, validate_shape
from evoloop.harness import Harness
from evoloop.loop import EvoLoop, LoopConfig
from evoloop.memory import Memory
from evoloop.registry import Registry
from evoloop.sandbox import Sandbox, static_check
from evoloop.sim import SimLLM, TOOL_CODE
from evoloop.tasks import FAMILIES, make_tasks
from evoloop.embed import CachedEmbedder, ConceptEmbedder, HashEmbedder
from evoloop.swarm import propose_subagent, validate_subagent, tool_schema, validate_args
from evoloop.evaluator import Evaluator


class Sand(unittest.TestCase):
    def test_bans(self):
        for bad in ("import os\ndef run(text): return os.getcwd()", "def run(text): return open('x').read()",
                    "def run(text): return ().__class__", "def run(text): return eval('1')"):
            self.assertTrue(static_check(bad), bad)

    def test_timeout_and_memory(self):
        s = Sandbox(timeout_s=1)
        self.assertIn("timeout", s.run("def run(text):\n    while True: pass\n", "run", {"text": ""}).error)
        self.assertFalse(s.run("def run(text):\n    return len('a'*(10**10))\n", "run", {"text": ""}).ok)

    def test_library_tools_solve_tasks(self):
        s = Sandbox()
        for t in make_tasks(list(FAMILIES), 3, 5):
            r = s.run(TOOL_CODE[t.family], "run", {"text": t.input})
            self.assertTrue(r.ok and t.check(str(r.value)), t.family)


class Forge(unittest.TestCase):
    def test_shape_reports_all(self):
        self.assertGreaterEqual(len(validate_shape({})), 4)

    def test_bad_tools_rejected_good_accepted(self):
        f = ToolForge(SimLLM(0), Sandbox(), max_attempts=1)
        good = json.loads(SimLLM(0)._forger("", "### FORGE\nTask definition: " + FAMILIES["reverse"].instruction + "\nATTEMPT 1"))
        bad = dict(good, code="import os\n" + good["code"])
        self.assertEqual(f.submit(bad).stage, "static")
        wrong = dict(good, code=good["code"].replace("return ", "return '~'+"))
        self.assertEqual(f.submit(wrong).stage, "tests")


class Infra(unittest.TestCase):
    def test_registry_rollback(self):
        r, h = Registry(), Harness()
        v1 = r.add(h, None); r.promote(v1); v2 = r.add(h.clone(), v1); r.promote(v2)
        self.assertEqual(r.champion(), v2)
        self.assertEqual(r.rollback("x"), v1)
        self.assertEqual(r.champion(), v1)

    def test_memory_freeze(self):
        m = Memory(); m.remember("a", "k", "v"); m.freeze()
        with self.assertRaises(PermissionError):
            m.remember("a", "k2", "v2")

    def test_gates(self):
        self.assertLess(fisher_one_sided(9, 1, 3, 7), 0.05)
        ids = [str(i) for i in range(20)]
        a = {i: TaskEval(i, 3, 3) for i in ids}; b = {i: TaskEval(i, 1, 3) for i in ids}
        self.assertTrue(paired_lift(candidate_evals=a, control_evals=b, task_ids=ids).credited_2sigma)
        self.assertFalse(paired_lift(candidate_evals=b, control_evals=b, task_ids=ids).credited_2sigma)

    def test_anneal(self):
        b = [edit_budget(t, 6, 1, 4) for t in range(6)]
        self.assertEqual(b[0], 4); self.assertTrue(all(x >= y for x, y in zip(b, b[1:])))


def _lessons():
    return {f: f"Lesson: {FAMILIES[f].instruction} Apply this definition literally and check edge cases." for f in FAMILIES}


class SemanticMemory(unittest.TestCase):
    def recall1(self, emb, store="brute"):
        les, tasks = _lessons(), make_tasks(list(FAMILIES), 6, 4, "m", 1.0)
        m = Memory(embedder=emb, store=store)
        for f, l in les.items():
            m.remember("lesson", f, l)
        r = [m.recall(t.instruction, 1) for t in tasks]
        return sum(bool(x) and x[0] == les[t.family] for x, t in zip(r, tasks)) / len(tasks)

    def test_semantic_beats_lexical_on_paraphrase(self):
        lex, sem = self.recall1(None), self.recall1(CachedEmbedder(ConceptEmbedder()))
        self.assertLess(lex, 0.6); self.assertGreater(sem, 0.8)

    def test_hash_embedder_is_not_semantic(self):
        self.assertLess(self.recall1(CachedEmbedder(HashEmbedder())), 0.6)   # honestidade: n-gramas não resolvem sinônimos

    def test_chroma_store_if_installed(self):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            self.skipTest("chromadb não instalado")
        self.assertGreater(self.recall1(CachedEmbedder(ConceptEmbedder()), "chroma"), 0.8)

    def test_frozen_vector_memory(self):
        m = Memory(embedder=ConceptEmbedder()); m.remember("a", "k", "v"); m.freeze()
        with self.assertRaises(PermissionError):
            m.remember("a", "k2", "v2")


class Router(unittest.TestCase):
    def test_function_calling_beats_regex_on_paraphrase(self):
        llm = SimLLM(1); ev = Evaluator(llm, Memory(), Sandbox())
        h = Harness()
        for f in ("reverse", "digit_sum", "sort_words"):
            h.tools["tool_" + f] = {"spec": {"description": FAMILIES[f].instruction, "applies_to": list(FAMILIES[f].keys),
                                             "input_schema": {"properties": {"text": {"type": "string"}}}}, "code": TOOL_CODE[f], "origin": "lib"}
        ts = make_tasks(["reverse", "digit_sum", "sort_words"], 8, 3, "r", 1.0)
        a = ev.evaluate(h, ts, 3, log=False)
        h2 = h.clone(); h2.config["router_mode"] = "llm"
        b = ev.evaluate(h2, ts, 3, log=False)
        self.assertGreater(b.S, a.S + 0.3); self.assertGreater(b.agent_stats["route_tool_calls"], 0)

    def test_schema_validation(self):
        sch = tool_schema("t", {"spec": {"description": "d", "input_schema": {"properties": {"text": {"type": "string"}}}}})
        self.assertEqual(sch["input_schema"]["required"], ["text"])
        self.assertTrue(validate_args(sch, {}) and validate_args(sch, {"text": 1}) and validate_args(sch, {"text": "a", "x": 1}))
        self.assertEqual(validate_args(sch, {"text": "a"}), [])


class Swarm(unittest.TestCase):
    def test_subagent_validation_and_architect(self):
        spec, errs = propose_subagent(SimLLM(0), [FAMILIES["reverse"].instruction, FAMILIES["digit_sum"].instruction], {}, {})
        self.assertIsNotNone(spec, errs)
        self.assertEqual(validate_subagent(spec, {}, {}), [])
        self.assertTrue(validate_subagent(dict(spec, tools=["nao_existe"]), {}, {}))
        self.assertTrue(validate_subagent(dict(spec, name="Bad Name"), {}, {}))

    def test_delegation_and_qa(self):
        llm = SimLLM(2); ev = Evaluator(llm, Memory(), Sandbox())
        spec, _ = propose_subagent(llm, [FAMILIES["reverse"].instruction, FAMILIES["digit_sum"].instruction], {}, {})
        h = Harness(); h.subagents[spec["name"]] = {k: spec[k] for k in ("description", "system_prompt", "tools")}
        h.config["orchestrate"] = True
        ts = make_tasks(["reverse", "digit_sum"], 6, 3, "s", 1.0)
        b = ev.evaluate(h, ts, 2, log=False)
        self.assertGreater(b.agent_stats["delegations"], 0)
        self.assertTrue(any(e.route.startswith("sub:") for e in b.episodes))
        base = ev.evaluate(Harness(), make_tasks(list(FAMILIES), 6, 5, "q"), 3, log=False)
        h2 = Harness(); h2.config["qa_agent"] = True
        qa = ev.evaluate(h2, make_tasks(list(FAMILIES), 6, 5, "q"), 3, log=False)
        self.assertLess(qa.verifier_pass - qa.S, base.verifier_pass - base.S)   # revisor reduz o gap Goodhart


class E2E(unittest.TestCase):
    def run_loop(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            return EvoLoop(LoopConfig(rounds=4, seed=3, out=d, **kw), quiet=True).run()

    def test_improves_and_is_deterministic(self):
        a, b = self.run_loop(), self.run_loop()
        self.assertGreater(a["final"]["evolve_fresh"]["lift"], 0.3)
        self.assertEqual(a["final"]["evolve_fresh"], b["final"]["evolve_fresh"])

    def test_saboteur_never_promoted(self):
        rep = self.run_loop(chaos=True)
        self.assertFalse(any(w and w.startswith("chaos") for w in rep["promotions"]))
        self.assertGreater(rep["final"]["evolve_fresh"]["lift"], 0.3)


if __name__ == "__main__":
    unittest.main()
