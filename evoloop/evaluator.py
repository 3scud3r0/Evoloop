"""Avaliador: roda um harness em tarefas x k tentativas com o oráculo estrito. Devolve EvalResult (RRSI) + TaskEval (Raven)."""
from __future__ import annotations
from dataclasses import dataclass

from raven_gates import TaskEval
from rrsi_core import EvalResult, TaskResult, aggregate

from .agent import Agent, Verifier
from .experience import Episode, ExperienceDB
from .forge import ToolForge
from .harness import Harness
from .llm import LLM
from .memory import Memory
from .sandbox import Sandbox
from .tasks import Task


@dataclass
class EvalBundle:
    ev: EvalResult
    evals: dict
    episodes: list
    verifier_pass: float
    forge_rejections: dict
    agent_stats: dict = None

    @property
    def S(self) -> float:
        return self.ev.S

    def family_rates(self) -> dict:
        acc: dict = {}
        for e in self.episodes:
            acc.setdefault(e.family, []).append(e.score)
        return {f: round(sum(v) / len(v), 3) for f, v in sorted(acc.items())}


class Evaluator:
    def __init__(self, llm: LLM, memory: Memory, sandbox: Sandbox, exp: ExperienceDB | None = None, verifier: Verifier | None = None):
        self.llm, self.memory, self.sandbox, self.exp, self.verifier = llm, memory, sandbox, exp, verifier
        self.n_episodes = 0

    def evaluate(self, harness: Harness, tasks: list[Task], k: int, rep: int = 0, split: str = "evolve", log: bool = True, job: str = "") -> EvalBundle:
        forge = ToolForge(self.llm, self.sandbox)
        agent = Agent(harness, self.llm, self.memory, self.sandbox, forge, self.verifier)
        per_task, evals, eps = {}, {}, []
        for t in tasks:
            rs, ts, mine = [], [], []
            for j in range(k):
                ep = agent.run(t, rep * 1000 + j)
                ep.score = 1.0 if t.check(ep.answer) else 0.0
                rs.append(ep.score); ts.append(ep.tokens or None); mine.append(ep)
            per_task[t.id] = TaskResult(rewards=rs, tokens=ts)
            evals[t.id] = TaskEval(t.id, int(sum(rs)), k)
            eps += mine
        self.n_episodes += len(eps)
        if log and self.exp is not None:
            self.exp.log_episodes(harness.digest(), split, eps)
        vp = sum(e.verifier_ok for e in eps) / max(1, len(eps))
        return EvalBundle(aggregate(job or harness.digest(), k, per_task), evals, eps, vp, dict(forge.rejections), dict(agent.stats))
