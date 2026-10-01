"""O ciclo fechado: harness campeão -> propor (GEPA + RRSI) -> critic -> avaliar -> selecionar -> confirmar -> promover | rollback."""
from __future__ import annotations
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

from raven_gates import paired_lift
from rrsi_core import RRSIConfig, select_round

from . import ui
from .evaluator import EvalBundle, Evaluator
from .evolution import HCandidate, Critic, chaos_candidate, gepa_candidate, structural_candidates
from .experience import ExperienceDB
from .forge import ToolForge
from .harness import Harness
from .embed import CachedEmbedder, make_embedder
from .llm import CountingLLM
from .memory import Memory
from .registry import Registry
from .sandbox import Sandbox
from .sim import SimLLM
from .tasks import LOOP_FAMILIES, OOD_TEST, make_tasks


@dataclass
class LoopConfig:
    rounds: int = 6
    n_train: int = 6            # tarefas por família no conjunto de evolução
    k: int = 3                  # tentativas por tarefa (Evaluate do RRSI)
    n_conf: int = 8             # tarefas/família no conjunto de confirmação (fresco a cada round)
    k_conf: int = 3
    gepa_budget: int = 160
    z_min: float = 1.64         # teste pareado unilateral ~95%
    guard_drop: float = 0.25    # regressão máxima tolerada por família
    cal_reps: int = 8
    min_lift: float = 0.03      # efeito mínimo prático para promover (além do z)
    patience: int = 2           # para se saturado e sem promoções por `patience` rounds
    saturation_S: float = 0.97
    seed: int = 1
    chaos: bool = False
    use_gepa: bool = True
    use_structural: bool = True
    use_swarm: bool = True
    use_router: bool = True
    memory: str = "lexical"       # lexical | hash | concept | http | st
    vstore: str = "brute"         # brute | chroma
    sandbox: str = "auto"         # auto | rlimit | namespaces | nsjail | gvisor
    require_isolation: bool = False
    para_rate: float = 0.5        # fração de tarefas com instrução parafraseada
    dilution: float = 0.0         # suposição do SimLLM (ver sim.py)
    out: str = "runs/demo"
    llm: str = "sim"
    rrsi: RRSIConfig = field(default_factory=lambda: RRSIConfig(T=6, k=3, m=2, b_min=1, b_max=3, w=2, m_draft=1))


def _fam(task_id: str) -> str:
    return task_id.split("-", 1)[1].rsplit("-", 1)[0]


class EvoLoop:
    def __init__(self, cfg: LoopConfig, llm=None, quiet: bool = False):
        self.cfg, self.quiet = cfg, quiet
        cfg.rrsi.T, cfg.rrsi.k = cfg.rounds, cfg.k
        out = Path(cfg.out)
        out.mkdir(parents=True, exist_ok=True)
        for f in ("experience.db", "registry.db", "memory.db"):
            (out / f).unlink(missing_ok=True)
        inner = llm or self._make_llm()
        self.llm = CountingLLM(inner)
        self.sandbox = Sandbox(backend=cfg.sandbox, require_isolation=cfg.require_isolation)
        emb = make_embedder(cfg.memory)
        self.embedder = CachedEmbedder(emb) if emb else None
        self.exp = ExperienceDB(str(out / "experience.db"))
        self.reg = Registry(str(out / "registry.db"))
        self.memory = Memory(str(out / "memory.db"), embedder=self.embedder, store=cfg.vstore)
        self.evalr = Evaluator(self.llm, self.memory, self.sandbox, self.exp)
        self.forge = ToolForge(self.llm, self.sandbox)
        self.rng = random.Random(cfg.seed)
        self.rounds_log: list[dict] = []
        self.accepted: dict = {}
        self.no_promo = 0
        self.evolve_tasks = make_tasks(LOOP_FAMILIES, cfg.n_train, cfg.seed * 7 + 1, "e", cfg.para_rate)
        seed_h = Harness(note="seed")
        self.champ, self.champ_lessons = seed_h, ()
        self.champ_v = self.reg.add(seed_h, None, "seed")
        self.reg.promote(self.champ_v, "seed")
        self.seed_v = self.champ_v
        self.delta, self.S_star = 0.0, 0.0

    def _make_llm(self):
        if self.cfg.llm == "anthropic":
            from .llm import AnthropicLLM
            return AnthropicLLM()
        return SimLLM(self.cfg.seed, dilution=self.cfg.dilution)

    def say(self, *a):
        if not self.quiet:
            print(*a, flush=True)

    # ---- utilidades ------------------------------------------------------------------------------------------
    def _mem(self, lessons: tuple) -> Memory:
        m = Memory(embedder=self.embedder, store=self.cfg.vstore)
        for x in lessons:
            m.remember("lesson", hashlib.sha256(x.encode()).hexdigest()[:12], x)
        m.freeze()
        return m

    def ev(self, h: Harness, lessons: tuple, tasks, k: int, rep: int, split: str = "evolve", log: bool = True) -> EvalBundle:
        self.evalr.memory = self._mem(lessons)
        return self.evalr.evaluate(h, tasks, k, rep=rep, split=split, log=log)

    def calibrate(self) -> float:
        """δ = z·sd(ΔS nulo): réplicas independentes do MESMO harness medem o ruído do avaliador (RRSI, calibração)."""
        ss = [self.ev(self.champ, self.champ_lessons, self.evolve_tasks, self.cfg.k, 500 + i, log=False).S for i in range(self.cfg.cal_reps)]
        sd_null = math.sqrt(2) * stdev(ss)          # ΔS nulo = S_a - S_b de réplicas independentes => sd = √2·sd(S)
        return max(0.02, self.cfg.rrsi.delta_z * sd_null), ss

    def _guard(self, inc, cand) -> list[str]:
        def fam_means(e):
            acc: dict = {}
            for tid, tr in e.per_task.items():
                acc.setdefault(_fam(tid), []).append(tr.mean)
            return {f: mean(v) for f, v in acc.items()}
        a, b = fam_means(inc), fam_means(cand)
        extra = []
        # Achado real (ver docs/LIMITATIONS.md): a regra "shaped" do RRSI dentro da banda de ruído pode admitir uma REGRESSÃO de S
        # se ela economizar muitos tokens (-w_c·ΔC compensa w_s·ΔS). Guard adicional nosso: dentro da banda, ΔS não pode ser < -δ/4.
        if cand.S - inc.S < -self.delta / 4:
            extra.append(f"ΔS={cand.S - inc.S:+.3f} < -δ/4 (economia de tokens não compensa regressão)")
        return extra + [f"família {f} caiu {a[f]:.2f}->{b.get(f, 0):.2f}" for f in a if a[f] - b.get(f, 0) > self.cfg.guard_drop]

    # ---- um round --------------------------------------------------------------------------------------------
    def run_round(self, t: int) -> dict:
        cfg, t0 = self.cfg, time.time()
        champ, lessons = self.champ, self.champ_lessons
        cb = self.ev(champ, lessons, self.evolve_tasks, cfg.k, t)
        fails = [e for e in cb.episodes if e.score < 1]
        fdicts = [{"instruction": e.instruction, "input": e.input, "expected": e.expected, "answer": e.answer, "family": e.family} for e in fails]
        rates = cb.family_rates()
        gval = make_tasks(LOOP_FAMILIES, 4, cfg.seed * 7 + 100 + t, "gv", cfg.para_rate)
        critic = Critic(self.evolve_tasks + gval)
        stalled = self.no_promo >= cfg.rrsi.w
        cands: list[HCandidate] = []
        g = gepa_candidate(champ, lessons, self.evalr, self.llm, self.evolve_tasks, gval, cfg.gepa_budget, cfg.seed + t, critic, t) if (fdicts and cfg.use_gepa) else None
        if g:
            cands.append(g)
        if fdicts and cfg.use_structural:
            cands += structural_candidates(champ, lessons, fdicts, rates, t, cfg.rounds, cfg.rrsi, self.forge, critic, self.accepted, stalled, self.rng,
                                            cfg.use_swarm, round(max(0.0, cb.verifier_pass - cb.S), 3), cfg.use_router)
        if cfg.chaos:
            cands.append(chaos_candidate(champ, lessons, t))
        for c in cands:
            if c.gate_failure is None:
                b = self.ev(c.harness, c.lessons, self.evolve_tasks, cfg.k, t)
                c.ev, c.bundle = b.ev, b
        winner, decs = select_round(cands, cb.ev, self.S_star, self.delta, cfg.rrsi, self.accepted, self._guard)
        row = {"round": t, "champion_v": self.champ_v, "champion_S": round(cb.S, 4), "champion_families": rates, "delta": round(self.delta, 4),
               "S_star": round(self.S_star, 4), "stalled": stalled, "candidates": [], "winner": None, "confirm": None, "promoted": False,
               "rolled_back": False}
        for c, d in zip(cands, decs):
            row["candidates"].append({"variant": c.variant, "source": c.source, "components": c.components, "S": None if d.S is None else round(d.S, 4),
                                      "dS": None if d.delta_S is None else round(d.delta_S, 4), "dC": None if d.delta_C is None else round(d.delta_C, 3),
                                      "admissible": d.admissible, "reason": d.reason[:160] + (" | " + c.detail if c.detail else "")})
            self.exp.log_decision(t, c.variant, c.source, c.harness.digest(), d.admissible, False, d.reason, {"components": c.components})
        if winner is not None:
            row["winner"] = winner.variant
            row["confirm"], ok = self.confirm(winner, t)
            if ok:
                self.promote(winner, t, row)
            else:
                row.setdefault("notes", []).append(ui.amber(f"↩ descartado (rollback): {winner.variant} não passou na confirmação"))
                v = self.reg.add(winner.harness, self.champ_v, "descartado na confirmação")
                self.reg.reject(v, json.dumps(row["confirm"]))
        self.no_promo = 0 if row["promoted"] else self.no_promo + 1
        row["secs"] = round(time.time() - t0, 2)
        self.rounds_log.append(row)
        self._print_round(row)
        return row

    def confirm(self, cand: HCandidate, t: int):
        cfg = self.cfg
        ct = make_tasks(LOOP_FAMILIES, cfg.n_conf, cfg.seed * 1000 + t + 50, "c", cfg.para_rate)
        a = self.ev(cand.harness, cand.lessons, ct, cfg.k_conf, 100 + t, split="confirm")
        b = self.ev(self.champ, self.champ_lessons, ct, cfg.k_conf, 100 + t, split="confirm")
        pl = paired_lift(candidate_evals=a.evals, control_evals=b.evals, task_ids=[x.id for x in ct], z_threshold=cfg.z_min)
        info = {"n": pl.n_tasks, "cand": round(pl.candidate_mean, 4), "ctrl": round(pl.control_mean, 4), "lift": round(pl.mean_lift, 4),
                "z": None if math.isinf(pl.z) else round(pl.z, 2), "z_min": cfg.z_min, "min_lift": cfg.min_lift}
        return info, bool(pl.promoted and pl.credited_2sigma and pl.mean_lift >= cfg.min_lift)

    def promote(self, cand: HCandidate, t: int, row: dict, force: bool = False) -> None:
        prev_h, prev_l, prev_v = self.champ, self.champ_lessons, self.champ_v
        v = self.reg.add(cand.harness, prev_v, cand.variant)
        self.reg.promote(v, "forçado" if force else "gates ok")
        self.champ, self.champ_lessons, self.champ_v = cand.harness, cand.lessons, v
        if not force:
            self.S_star = max(self.S_star, cand.ev.S)
        for x in cand.lessons:
            if x not in prev_l:
                self.memory.remember("lesson", hashlib.sha256(x.encode()).hexdigest()[:12], x)
        for c in cand.components:
            self.accepted[c] = self.accepted.get(c, 0) + 1
        row["promoted"] = True
        self.exp.log_decision(t, cand.variant, cand.source, cand.harness.digest(), True, True, "promovido", {})
        # canário: conjunto fresco, novo campeão vs anterior; regressão além da banda de ruído => rollback
        ok, info = self.canary(cand.harness, cand.lessons, prev_h, prev_l, t)
        row["canary"] = info
        if not ok:
            self.reg.rollback(f"canário regrediu: {info}")
            self.champ, self.champ_lessons, self.champ_v = prev_h, prev_l, prev_v
            row["promoted"], row["rolled_back"] = False, True
            row.setdefault("notes", []).append(ui.magenta(f"⟲ ROLLBACK pós-promoção: {info}"))

    def canary(self, new_h, new_l, old_h, old_l, t: int):
        ct = make_tasks(LOOP_FAMILIES, self.cfg.n_conf, self.cfg.seed * 1000 + t + 70, "k", self.cfg.para_rate)
        a = self.ev(new_h, new_l, ct, self.cfg.k_conf, 200 + t, split="canary")
        b = self.ev(old_h, old_l, ct, self.cfg.k_conf, 200 + t, split="canary")
        pl = paired_lift(candidate_evals=a.evals, control_evals=b.evals, task_ids=[x.id for x in ct], z_threshold=self.cfg.z_min)
        info = {"new": round(pl.candidate_mean, 4), "old": round(pl.control_mean, 4), "lift": round(pl.mean_lift, 4)}
        return pl.mean_lift >= -self.delta, info

    # ---- execução completa -------------------------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        self.say(ui.cyan("▛▀▀ EVOLOOP ▀▀▜") + ui.dim(f"  seed={cfg.seed} rounds={cfg.rounds} llm={cfg.llm} chaos={cfg.chaos}"))
        b0 = self.ev(self.champ, self.champ_lessons, self.evolve_tasks, cfg.k, 0, log=False)
        self.delta, cal = self.calibrate()
        self.S_star = b0.S
        self.say(ui.lime(f"baseline S={b0.S:.3f}") + ui.dim(f"  δ(ruído)={self.delta:.3f} calibrado em {[round(x, 3) for x in cal]}"))
        self.stop_reason = "rounds esgotados"
        for t in range(cfg.rounds):
            row = self.run_round(t)
            if row["champion_S"] >= cfg.saturation_S and self.no_promo >= cfg.patience:
                self.stop_reason = f"saturado (S>={cfg.saturation_S}) sem promoções por {cfg.patience} rounds"
                self.say(ui.dim("   ■ " + self.stop_reason))
                break
        rep = self.final_report()
        Path(cfg.out, "report.json").write_text(json.dumps(rep, indent=1, default=str))
        Path(cfg.out, "champion_harness.json").write_text(self.champ.to_json())
        return rep

    def final_report(self) -> dict:
        cfg = self.cfg
        seed_h = self.reg.get(self.seed_v)
        res = {}
        for name, fams in (("evolve_fresh", LOOP_FAMILIES), ("ood_test", OOD_TEST)):
            ts = make_tasks(fams, 10, cfg.seed * 9999 + (1 if name == "evolve_fresh" else 2), "F", cfg.para_rate)
            a = self.ev(self.champ, self.champ_lessons, ts, 5, 900, split="final_" + name, log=False)
            b = self.ev(seed_h, (), ts, 5, 900, split="final_" + name, log=False)
            pl = paired_lift(candidate_evals=a.evals, control_evals=b.evals, task_ids=[t.id for t in ts], z_threshold=cfg.z_min)
            res[name] = {"seed_S": round(b.S, 4), "champion_S": round(a.S, 4), "lift": round(pl.mean_lift, 4),
                         "z": None if math.isinf(pl.z) else round(pl.z, 2), "champion_families": a.family_rates(),
                         "verifier_pass_champion": round(a.verifier_pass, 4), "goodhart_gap": round(a.verifier_pass - a.S, 4),
                         "tokens_per_trial": {"seed": round(b.ev.C or 0, 1), "champion": round(a.ev.C or 0, 1)}}
        return {"config": {k: v for k, v in cfg.__dict__.items() if k != "rrsi"}, "rrsi": cfg.rrsi.dump(), "champion_version": self.champ_v,
                "promotions": [r["winner"] for r in self.rounds_log if r["promoted"]], "rounds": self.rounds_log,
                "stop_reason": self.stop_reason, "final": res, "llm_calls": self.llm.calls, "llm_tokens": self.llm.tokens, "episodes": self.evalr.n_episodes,
                "sandbox_spawns": self.sandbox.spawns, "isolation": self.sandbox.isolation, "memory_backend": self.memory.backend,
                "champion_subagents": sorted(self.champ.subagents), "champion_config": self.champ.config, "forge_rejections": dict(self.forge.rejections), "registry": self.reg.history(),
                "memory_lessons": len(self.memory), "honestidade": (
                    "LLM=%s. Com SimLLM os ganhos validam o MECANISMO do ciclo (propor->testar->promover/reverter), não capacidade real. "
                    "Verificador propositalmente fraco; oráculo estrito só no avaliador. Sandbox = defesa em profundidade, não fronteira de segurança." % cfg.llm)}

    def _print_round(self, r: dict) -> None:
        self.say(ui.cyan(f"[round {r['round']}]") + f" campeão v{r['champion_v']} S={r['champion_S']:.3f}" + ui.dim(f"  δ={r['delta']:.3f} S*={r['S_star']:.3f}" + (" ESTAGNADO→explora" if r["stalled"] else "")))
        for c in r["candidates"]:
            tag = ui.lime("ok ") if c["admissible"] else ui.magenta("x  ")
            s = "  —  " if c["S"] is None else f"{c['S']:.3f}"
            self.say(f"   {tag}{c['variant']:<22} S={s} ΔS={c['dS'] if c['dS'] is not None else '—':>7} {c['components']}  {ui.dim(c['reason'][:78])}")
        if r["winner"]:
            cf = r["confirm"]
            self.say(f"   ▶ vencedor {r['winner']}  confirmação: lift={cf['lift']:+.3f} z={cf['z']} (mín {cf['z_min']}) → " + (ui.lime("PROMOVIDO") if r["promoted"] else ui.amber("não promovido")))
        else:
            self.say(ui.dim("   (nenhum candidato admissível — campeão mantido)"))
        for n in r.get("notes", []):
            self.say("   " + n)
