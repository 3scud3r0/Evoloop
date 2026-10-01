# Trecho extraído de Raven (EverMind-AI/Raven, evolver/orchestrator/scoring.py, Apache-2.0,
# commit e6c0344): apenas a dataclass TaskEval. MODIFICADO: imports do Raven removidos.
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskEval:
    """Resultado de K tentativas de uma tarefa (moeda universal dos gates)."""
    task_id: str
    passes: int
    attempts: int
    infra_attempts: int = 0

    @property
    def pass_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.passes / self.attempts
