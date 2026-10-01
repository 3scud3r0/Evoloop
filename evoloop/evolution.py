"""Motor de evolução: GEPA (mutação reflexiva + Pareto) e proposer estrutural estilo RRSI, com crítico anti-vazamento."""
from __future__ import annotations
import difflib
import json
import random
import re
from dataclasses import dataclass, field

import gepa
from gepa.core.adapter import EvaluationBatch
from rrsi_core import Candidate, RRSIConfig, edit_budget
from rrsi_core.components import K_STR, normalize

from .evaluator import Evaluator
from .forge import ToolForge
from .swarm import propose_subagent
from .harness import Harness
from .llm import LLM
from .tasks import FAMILIES, LOOP_FAMILIES, Task

REFLECT_SYSTEM = ("You improve the text of ONE component of an LLM agent harness from failure examples. Generalize: write "
                  "reusable rules, never copy example inputs/outputs. Reply with ONLY the new component text.")
DOMAIN_SIGNALS = [
    ("subagent", [r"(?m)^\+register_subagent\(", r"(?m)^\+config\.(orchestrate|qa_agent)"]),
    ("control_flow", [r"(?m)^\+route:", r"(?m)^\+config\.router_mode"]),
    ("config", [r"(?m)^\+config\.(max_retries|use_tools|use_planner|memory_k)"]),
]


def render(h: Harness, lessons: tuple = ()) -> str:
    """Visão textual estável do harness (base dos diffs que classificam cada edição por componente)."""
    parts = [f"[{k}]\n{v}" for k, v in sorted(h.components.items())]
    parts += [f"config.{k} = {v}" + ("  # register_tool at runtime" if k == "forge_on_demand" and v else "") for k, v in sorted(h.config.items())]
    for n, t in sorted(h.tools.items()):
        parts.append(f'register_tool("{n}")\n{t["code"]}')
    for n, a in sorted(h.subagents.items()):
        parts.append(f'register_subagent("{n}")  # tools={a.get("tools", [])}\n{a["system_prompt"]}')
    parts += [f'.remember("{x}")' for x in lessons]
    return "\n".join(parts)


def diff_text(old: str, new: str) -> str:
    return "\n".join(l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0) if not l.startswith(("---", "+++", "@@")))


@dataclass
class HCandidate(Candidate):
    harness: Harness | None = None
    lessons: tuple = ()
    source: str = ""
    bundle: object = None
    notes: list = field(default_factory=list)


def tag_edits(base: Harness, base_lessons: tuple, cand: Harness, lessons: tuple, declared: list[tuple[str, str]]) -> list[dict]:
    """Uma edição por (componente declarado, hipótese); o componente é revalidado contra o diff real (RRSI: normalize)."""
    d = diff_text(render(base, base_lessons), render(cand, lessons))
    return [{"component": normalize(dc, d, DOMAIN_SIGNALS), "declared": dc, "hypothesis": hyp} for dc, hyp in declared]


class Critic:
    """Regularizador (papel do 'critic' do RRSI): rejeita edições específicas do benchmark e crescimento descontrolado.
    Só olha texto controlado pelo harness (componentes, código de ferramentas, lições) e casa literais com fronteira de palavra."""
    def __init__(self, tasks: list[Task], max_chars: int = 5000, max_growth: int = 1800):
        self.lits = {t.id for t in tasks} | {t.input for t in tasks if len(t.input) >= 5} | {t.expected for t in tasks if len(t.expected) >= 5}
        self.max_chars, self.max_growth = max_chars, max_growth

    @staticmethod
    def _user_text(h: Harness, lessons: tuple) -> str:
        return "\n".join(list(h.components.values()) + [t["code"] for t in h.tools.values()] + [a["system_prompt"] for a in h.subagents.values()] + list(lessons))

    def _found(self, text: str) -> list[str]:
        return sorted(l for l in self.lits if re.search(r"(?<![A-Za-z0-9])" + re.escape(l) + r"(?![A-Za-z0-9])", text))

    def review(self, base: Harness, cand: Harness, lessons: tuple = (), base_lessons: tuple = ()) -> list[str]:
        old = set(self._found(self._user_text(base, base_lessons)))
        leak = [l for l in self._found(self._user_text(cand, lessons)) if l not in old]
        errs = [f"vazamento: contém literal do conjunto de evolução {leak[0]!r}"] if leak else []
        size, size0 = sum(len(v) for v in cand.components.values()), sum(len(v) for v in base.components.values())
        for n, a in cand.subagents.items():
            if n not in base.subagents:
                hit = self._found(a["system_prompt"] + a["description"])
                if hit:
                    errs.append(f"vazamento em sub-agente {n}: {hit[0]!r}")
        if size > self.max_chars or size - size0 > self.max_growth:
            errs.append(f"crescimento excessivo dos componentes de texto ({size0}->{size} chars)")
        return errs


class GepaBridge:
    """GEPAAdapter: candidato GEPA = {componente_texto: texto}; avalia com o Evaluator real."""
    propose_new_texts = None      # GEPA lê este atributo direto; None => usa custom_candidate_proposer

    def __init__(self, base: Harness, evaluator: Evaluator, llm: LLM, k: int = 1):
        self.base, self.evaluator, self.llm, self.k = base, evaluator, llm, k
        self.n_proposals = 0

    def _h(self, cand: dict) -> Harness:
        h = self.base.clone()
        h.components.update(cand)
        return h

    def evaluate(self, batch, candidate, capture_traces=False):
        b = self.evaluator.evaluate(self._h(candidate), batch, self.k, rep=7, split="gepa", log=False)
        scores, outs, trajs = [], [], []
        for t in batch:
            eps = [e for e in b.episodes if e.task_id == t.id]
            scores.append(sum(e.score for e in eps) / len(eps))
            outs.append(eps[0].answer)
            trajs.append({"instruction": t.instruction, "input": t.input, "expected": t.expected, "answer": eps[0].answer,
                          "score": scores[-1], "trace": eps[0].trace})
        return EvaluationBatch(outputs=outs, scores=scores, trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = [dict(t) for t in (eval_batch.trajectories or []) if t["score"] < 1.0][:8]
        return {c: rows for c in components_to_update}

    def propose(self, candidate, reflective_dataset, components_to_update, *, metadata=None):
        out = {}
        self.n_proposals += 1
        for c in components_to_update:
            ex = json.dumps([{k: r[k] for k in ("instruction", "input", "expected", "answer")} for r in reflective_dataset.get(c, [])])
            out[c] = self.llm.complete("reflect", REFLECT_SYSTEM, f"### COMPONENT {c}\n### CURRENT\n{candidate[c]}\n### EXAMPLES\n{ex}").text
        return out


class _Rec:
    """Logger do GEPA: silencioso, mas guarda mensagens para detectar falhas engolidas pela biblioteca."""
    def __init__(self):
        self.msgs: list[str] = []

    def log(self, message: str):
        self.msgs.append(str(message))


def gepa_candidate(champion: Harness, lessons: tuple, evaluator: Evaluator, llm: LLM, train: list[Task], val: list[Task],
                   budget: int, seed: int, critic: Critic, rnd: int) -> HCandidate | None:
    bridge = GepaBridge(champion, evaluator, llm)
    seed_c = {k: champion.components[k] for k in ("system_prompt", "planner_prompt")}
    rec = _Rec()
    res = gepa.optimize(seed_candidate=seed_c, trainset=train, valset=val, adapter=bridge, custom_candidate_proposer=bridge.propose,
                        max_metric_calls=budget, reflection_minibatch_size=4, seed=seed, logger=rec, raise_on_exception=True,
                        display_progress_bar=False)
    broken = [m for m in rec.msgs if "reflection failed" in m.lower() or "traceback" in m.lower()]
    if broken:
        raise RuntimeError("GEPA engoliu falha no proposer: " + broken[0][:200])
    if bridge.n_proposals == 0:
        return None      # minibatches já perfeitos: GEPA (skip_perfect_score) não tem o que refletir
    best = dict(res.best_candidate)
    if best == seed_c:
        return None
    h = champion.clone(f"gepa r{rnd}")
    h.components.update(best)
    errs = critic.review(champion, h, lessons, lessons)
    edits = tag_edits(champion, lessons, h, lessons, [("prompt", "GEPA: mutação reflexiva + fronteira de Pareto")])
    c = HCandidate(f"gepa-r{rnd}", edits, harness=h, lessons=lessons, source="gepa")
    if errs:
        c.gate_failure, c.detail = "critic_reject", "; ".join(errs)
    return c


def structural_candidates(champion: Harness, lessons: tuple, failures: list[dict], family_rates: dict, rnd: int, T: int, cfg: RRSIConfig,
                          forge: ToolForge, critic: Critic, incumbent_counts: dict, stalled: bool, rng: random.Random,
                          swarm: bool = True, gap: float = 0.0, router: bool = True) -> list[HCandidate]:
    """Proposer estrutural (RRSI): edições que ADICIONAM maquinaria (ferramenta, memória, roteamento, config). Até b_t edições
    por candidato (recozimento cosseno: cedo agrupa, tarde fica esparso). Se estagnado, reserva m_draft slots exploratórios."""
    b_t = edit_budget(rnd, T, cfg.b_min, cfg.b_max)
    weak = sorted((f for f in LOOP_FAMILIES if family_rates.get(f, 1.0) < 0.95), key=lambda f: family_rates.get(f, 1.0))
    if not weak:
        return []
    cands: list[HCandidate] = []
    for ci in range(cfg.m + (cfg.m_draft if stalled else 0)):
        explore = stalled and ci >= cfg.m
        h, ls, decl, notes = champion.clone(f"struct r{rnd}#{ci}"), tuple(lessons), [], []
        menu: list[tuple] = []
        for f in weak:
            tool = next((n for n in h.tools if n == f"tool_{f}"), None)
            if tool is None:
                menu.append(("forge", f))
            elif family_rates.get(f, 1) < 0.5:
                menu.append(("route_around", f))
            if not any(FAMILIES[f].instruction in x for x in ls):
                menu.append(("lesson", f))
        if h.config["max_retries"] < 3:
            menu.append(("retries", None))
        if router and h.tools and h.config["router_mode"] == "regex":
            menu.append(("router_hybrid", None))      # function calling nativo como fallback do regex (control_flow)
        if swarm:
            covered = " ".join(a["description"] for a in h.subagents.values())
            uncovered = [f for f in weak if FAMILIES[f].instruction not in covered]
            if len(uncovered) >= 2:
                menu.append(("subagent", uncovered[:2]))   # arquiteto cria especialista p/ as 2 famílias mais fracas
            if not h.config["qa_agent"] and gap > 0.08:
                menu.append(("qa", None))                  # verificador passa lixo (gap Goodhart) => revisor independente
        if explore:                                   # slot exploratório = mecanismo geral ainda não tentado (RRSI: novidade)
            untried = [m for m in ([("on_demand", None)] if not h.config.get("forge_on_demand") else []) +
                       [m for m in menu if m[0] in ("subagent", "qa", "router_hybrid") and incumbent_counts.get("subagent" if m[0] != "router_hybrid" else "control_flow", 0) == 0]]
            menu = untried[ci % max(1, len(untried)):] + untried[:ci % max(1, len(untried))] if untried else menu
        else:
            rng.shuffle(menu)
        for kind, f in menu[: max(1, b_t if not explore else 1)]:
            if kind == "router_hybrid":
                h.config["router_mode"] = "hybrid"
                decl.append(("control_flow", "roteamento híbrido: regex + function calling nativo (tool_choice=auto)"))
                continue
            if kind == "subagent":
                spec, errs = propose_subagent(forge.llm, [FAMILIES[x].instruction for x in f], h.tools, h.subagents)
                if spec is None:
                    notes.append("arquiteto rejeitado: " + "; ".join(errs)[:100])
                    continue
                h.subagents[spec["name"]] = {k: spec[k] for k in ("description", "system_prompt", "tools")}
                h.config["orchestrate"] = True
                decl.append(("subagent", f"especialista {spec['name']} + orquestrador"))
                continue
            if kind == "qa":
                h.config["qa_agent"] = True
                decl.append(("subagent", "revisor QA independente (re-derivação)"))
                continue
            if kind == "forge":
                r = forge.forge_for(FAMILIES[f].instruction)
                if not r.ok:
                    notes.append(f"forge {f} falhou no estágio {r.stage}")
                    continue
                h.tools[r.name] = r.tool
                decl.append(("client_tool", f"ferramenta forjada para {f}"))
            elif kind == "lesson":
                ls += (f"Lesson: {FAMILIES[f].instruction} Apply this definition literally and check edge cases.",)
                decl.append(("memory", f"lição de memória para {f}"))
            elif kind == "route_around":
                h.components["router_rules"] = (h.components["router_rules"] + f"\nroute: {FAMILIES[f].keys[0]} -> llm").strip()
                decl.append(("control_flow", f"desviar {f} de ferramenta com baixo rendimento"))
            elif kind == "retries":
                h.config["max_retries"] += 1
                decl.append(("config", "mais uma tentativa com feedback do verificador"))
            elif kind == "on_demand":
                h.config["forge_on_demand"] = True
                decl.append(("client_tool", "forge sob demanda em tempo de execução (mecanismo geral)"))
        if not decl:
            continue
        c = HCandidate(f"struct-r{rnd}#{ci}" + ("-explore" if explore else ""), tag_edits(champion, lessons, h, ls, decl), harness=h, lessons=ls, source="structural", notes=notes)
        errs = critic.review(champion, h, ls, lessons)
        if errs:
            c.gate_failure, c.detail = "critic_reject", "; ".join(errs)
        cands.append(c)
    return cands


def chaos_candidate(champion: Harness, lessons: tuple, rnd: int) -> HCandidate:
    """Saboteur: candidato plausível mas destrutivo (para provar que os gates barram regressão)."""
    h = champion.clone(f"chaos r{rnd}")
    h.components["system_prompt"] = "Answer."
    h.components["planner_prompt"] = ""
    h.config.update({"max_retries": 0, "use_tools": False, "use_planner": False})
    return HCandidate(f"chaos-r{rnd}", tag_edits(champion, lessons, h, lessons, [("prompt", "saboteur"), ("config", "saboteur")]), harness=h, lessons=lessons, source="chaos")


__all__ = ["HCandidate", "Critic", "gepa_candidate", "structural_candidates", "chaos_candidate", "render", "K_STR"]
