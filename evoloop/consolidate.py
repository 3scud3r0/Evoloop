"""Consolidação ("sono"): pega turnos com feedback, gera candidatos (lição de memória · prompt via GEPA · macro forjada), REPRODUZ os turnos
num workspace descartável (restaurado do snapshot do turno), pontua com um juiz, aplica os gates do Raven e grava o vencedor como
PENDENTE. Só vira campeão com aprovação humana (ou auto_promote=True + canário + rollback).

LIMITES HONESTOS: o juiz (LLM ou função) e o feedback do usuário são sinais FRACOS e ruidosos — daí o gate de aprovação por padrão.
Turnos que usaram GUI/rede não são reproduzíveis (shell sem rede) e ficam fora da evidência."""
from __future__ import annotations
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from raven_gates import TaskEval, paired_lift

from .env import NamespaceEnv
from .general import GeneralAgent
from .harness import Harness
from .llm import extract_json
from .macro_forge import forge_macro
from .memory import Memory
from .registry import Registry
from .store import TurnStore

JUDGE_SYS = ("You grade whether an AI agent satisfied the user's request. Use the user's feedback as the rubric for what 'wrong' meant. "
             'Reply ONLY JSON {"score": 0..1, "reason": "..."}.')
REFLECT_SYS = ("You improve an agent from user feedback on failed tasks. Write ONE short, general rule (<=250 chars, imperative, no user data, "
               "no URLs, no secrets) that would have avoided the failures. Reply with ONLY the rule.")
BAD_LESSON = re.compile(r"ignore (all|previous|prior)|disable|bypass|sandbox|sudo|password|api[_ -]?key|token|secret|https?://|rm -rf|safety", re.I)
NONREPLAYABLE = ("screenshot", "computer")
NET = re.compile(r"\b(curl|wget|pip install|npm install|apt(-get)? install|git clone|ping|ssh)\b")


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    def score(self, turn: dict, text: str, transcript: list, ws_listing: str) -> tuple[float, str]:
        u = json.dumps({"request": turn["user_text"], "user_feedback": turn.get("feedback") or "", "final_answer": text[:1500],
                        "tool_calls": [{"tool": t["tool"], "input": str(t["input"])[:200], "output": t["output"][:200]} for t in transcript[:12]],
                        "workspace_files": ws_listing[:800]}, ensure_ascii=False)
        try:
            d = extract_json(self.llm.complete("judge", JUDGE_SYS, u).text)
            return max(0.0, min(1.0, float(d["score"]))), str(d.get("reason", ""))[:120]
        except Exception as e:                          # noqa: BLE001
            return 0.0, f"juiz falhou: {str(e)[:60]}"


class CallableJudge:
    def __init__(self, fn):
        self.fn = fn

    def score(self, turn, text, transcript, ws_listing):
        return self.fn(turn, text, transcript, ws_listing)


@dataclass
class NightReport:
    ts: float = field(default_factory=time.time)
    n_fail: int = 0
    n_pos: int = 0
    n_replayable: int = 0
    candidates: list = field(default_factory=list)
    pending_v: int | None = None
    promoted_v: int | None = None
    note: str = ""

    def as_dict(self):
        return self.__dict__


class Consolidator:
    def __init__(self, store: TurnStore, registry: Registry, memory: Memory, llm, judge, snaps_dir: str | Path, *, min_failures: int = 3,
                 min_lift: float = 0.2, z_min: float = 1.28, guard_drop: float = 0.15, trials: int = 2, max_replay: int = 14,
                 auto_promote: bool = False, use_gepa: bool = False, use_macros: bool = True, apply_lessons=None):
        self.store, self.reg, self.memory, self.llm, self.judge = store, registry, memory, llm, judge
        self.snaps = Path(snaps_dir)
        self.cfg = dict(min_failures=min_failures, min_lift=min_lift, z_min=z_min, guard_drop=guard_drop, trials=trials, max_replay=max_replay,
                        auto_promote=auto_promote, use_gepa=use_gepa, use_macros=use_macros)
        self.apply_lessons = apply_lessons                # callback(lessons: list[str]) -> grava na memória viva

    # ---- reprodução de um turno com (harness, lições) candidatos -------------------------------------------------
    def replay(self, turn: dict, harness: Harness, lessons: list[str], trial: int) -> float:
        mem = Memory(embedder=self.memory.embedder, store=self.memory.store_kind)
        for _, k, t in self.memory.items():
            mem.remember("lesson", k, t)
        for i, l in enumerate(lessons):
            mem.remember("lesson", f"cand{i}", l)
        mem.freeze()
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"
            snap = self.snaps / turn["snapshot"] if turn.get("snapshot") else None
            if snap and snap.is_dir():
                shutil.copytree(snap, ws, symlinks=True)
            else:
                ws.mkdir()
            env = NamespaceEnv(ws)
            res = GeneralAgent(harness, self.llm, env, mem).run(json.loads(turn["history"]))
            listing = env.exec("find . -type f -not -path './.evoloop/*' | head -40").stdout
            s, _ = self.judge.score(turn, res.text, res.transcript, listing + "\n" + _peek(ws))
        return s

    def candidates(self, champ: Harness, fails: list[dict]) -> list[dict]:
        out = []
        ev = [{"request": t["user_text"][:300], "feedback": (t.get("feedback") or "")[:300], "answer": t["answer"][:300]} for t in fails[:8]]
        txt = self.llm.complete("reflect", REFLECT_SYS, json.dumps(ev, ensure_ascii=False)).text.strip().strip('"')
        if txt and len(txt) <= 300 and not BAD_LESSON.search(txt):
            out.append({"name": "lesson", "harness": champ.clone("lesson"), "lessons": [txt], "what": f"lição: {txt}"})
        elif txt:
            out.append({"name": "lesson-rejected", "rejected": "crítico: lição vazia/longa/perigosa", "what": txt[:120]})
        if self.cfg["use_macros"]:
            bad_tools = [t for t in fails if sum(1 for x in json.loads(t["transcript"]) if x.get("error")) >= 2]
            if bad_tools:
                spec, errs = forge_macro(self.llm, f"Automate this recurring failing task: {bad_tools[0]['user_text'][:300]}", champ.tools)
                if spec:
                    h = champ.clone("macro")
                    h.tools[spec["name"]] = spec["tool"]
                    out.append({"name": "macro:" + spec["name"], "harness": h, "lessons": [], "what": f"macro {spec['name']}"})
                else:
                    out.append({"name": "macro-failed", "rejected": "; ".join(errs)[:150], "what": "forja de macro"})
        if self.cfg["use_gepa"] and len(fails) >= 3:
            try:
                g = self._gepa(champ, fails)
                if g:
                    out.append(g)
            except Exception as e:                      # noqa: BLE001
                out.append({"name": "gepa-failed", "rejected": str(e)[:120], "what": "gepa"})
        return out

    def _gepa(self, champ: Harness, fails: list[dict]):
        import gepa
        from gepa.core.adapter import EvaluationBatch
        me = self

        class Bridge:
            propose_new_texts = None
            n = 0

            def evaluate(self, batch, candidate, capture_traces=False):
                h = champ.clone("gepa"); h.components["system_prompt"] = candidate["system_prompt"]
                sc = [me.replay(t, h, [], 0) for t in batch]
                return EvaluationBatch(outputs=[""] * len(batch), scores=sc,
                                       trajectories=[{"request": t["user_text"][:200], "feedback": t.get("feedback") or "", "score": s} for t, s in zip(batch, sc)] if capture_traces else None)

            def make_reflective_dataset(self, cand, eb, comps):
                return {c: [t for t in (eb.trajectories or []) if t["score"] < 1][:6] for c in comps}

            def propose(self, cand, ds, comps, **kw):
                self.n += 1
                return {c: me.llm.complete("reflect", "Rewrite the agent system prompt to fix the failures below. Reply ONLY the new prompt.",
                                           json.dumps({"current": cand[c], "failures": ds.get(c, [])}, ensure_ascii=False)).text.strip() for c in comps}

        class _Q:
            def log(self, m): pass
        b = Bridge()
        seed = {"system_prompt": champ.components["system_prompt"]}
        res = gepa.optimize(seed_candidate=seed, trainset=fails[:6], valset=fails[:6], adapter=b, custom_candidate_proposer=b.propose,
                            max_metric_calls=40, reflection_minibatch_size=3, logger=_Q(), raise_on_exception=True, display_progress_bar=False)
        best = dict(res.best_candidate)
        if best == seed or BAD_LESSON.search(best["system_prompt"]) or len(best["system_prompt"]) > 3000:
            return None
        h = champ.clone("gepa"); h.components["system_prompt"] = best["system_prompt"]
        return {"name": "gepa", "harness": h, "lessons": [], "what": "prompt reescrito (GEPA)"}

    def run(self, since_ts: float = 0.0) -> NightReport:
        rep = NightReport()
        cv = self.reg.champion()
        champ = self.reg.get(cv)
        fails = [t for t in self.store.query("score=0 AND ts>=? AND weight>=0.5", (since_ts,), 100)]
        pos = [t for t in self.store.query("score=1 AND ts>=?", (since_ts,), 50)]
        rep.n_fail, rep.n_pos = len(fails), len(pos)

        def replayable(t):
            tr = json.loads(t["transcript"])
            return not any(x["tool"] in NONREPLAYABLE or (x["tool"] == "bash" and NET.search(str(x["input"]))) for x in tr)
        fr, pr = [t for t in fails if replayable(t)], [t for t in pos if replayable(t)]
        rep.n_replayable = len(fr)
        if len(fr) < self.cfg["min_failures"]:
            rep.note = f"evidência insuficiente: {len(fr)} falhas reproduzíveis (< {self.cfg['min_failures']})"
            return rep
        cands = self.candidates(champ, fr)
        live_lessons = []
        sample = (fr + pr)[: self.cfg["max_replay"]]
        best = None
        for c in cands:
            if "rejected" in c:
                rep.candidates.append({"name": c["name"], "rejected": c["rejected"], "what": c["what"]})
                continue
            a, b = {}, {}
            for t in sample:
                tid = f"turn{t['id']}"
                sa = [self.replay(t, c["harness"], c["lessons"], k) for k in range(self.cfg["trials"])]
                sb = [self.replay(t, champ, [], k) for k in range(self.cfg["trials"])]
                a[tid] = TaskEval(tid, sum(x >= 0.5 for x in sa), len(sa))
                b[tid] = TaskEval(tid, sum(x >= 0.5 for x in sb), len(sb))
            pl = paired_lift(candidate_evals=a, control_evals=b, task_ids=list(a), z_threshold=self.cfg["z_min"])
            pids = [f"turn{t['id']}" for t in pr if f"turn{t['id']}" in a]
            drop = (sum(b[i].pass_rate for i in pids) - sum(a[i].pass_rate for i in pids)) / len(pids) if pids else 0.0
            ok = bool(pl.promoted and pl.mean_lift >= self.cfg["min_lift"] and drop <= self.cfg["guard_drop"])
            info = {"name": c["name"], "what": c["what"], "lift": round(pl.mean_lift, 3), "z": None if pl.z == float("inf") else round(pl.z, 2),
                    "cand": round(pl.candidate_mean, 3), "ctrl": round(pl.control_mean, 3), "n": pl.n_tasks, "regress_on_positives": round(drop, 3), "passed_gates": ok}
            rep.candidates.append(info)
            if ok and (best is None or pl.mean_lift > best[0]):
                best = (pl.mean_lift, c, info)
        if best is None:
            rep.note = "nenhum candidato passou nos gates"
            return rep
        _, c, info = best
        ev = {"candidate": info, "lessons": c["lessons"], "n_fail": rep.n_fail, "created": time.time()}
        v = self.reg.add_pending(c["harness"], cv, ev)
        rep.pending_v = v
        if self.cfg["auto_promote"]:
            self.approve(v)
            rep.promoted_v = v
        return rep

    def approve(self, v: int) -> bool:
        ev = self.reg.resolve_pending(v, "approved")
        if ev is None:
            return False
        self.reg.promote(v, "aprovado")
        if ev.get("lessons") and self.apply_lessons:
            self.apply_lessons(ev["lessons"])
        return True

    def decline(self, v: int) -> bool:
        ev = self.reg.resolve_pending(v, "declined")
        if ev is not None:
            self.reg.reject(v, "recusado pelo usuário")
        return ev is not None


def _peek(ws: Path) -> str:
    out = []
    for f in sorted(ws.rglob("*"))[:10]:
        if f.is_file() and ".evoloop" not in f.parts:
            out.append(f"{f.relative_to(ws)}: {f.read_text(errors='replace')[:100]!r}")
    return "\n".join(out)
