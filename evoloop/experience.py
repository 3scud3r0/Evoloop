"""Experience DB (SQLite): todo episódio, toda decisão de round. Alimenta reflexão, análise de falhas e auditoria."""
from __future__ import annotations
import json
import sqlite3
import time
from dataclasses import dataclass, field


@dataclass
class Episode:
    task_id: str
    family: str
    trial: int
    answer: str
    score: float                  # oráculo estrito (avaliador)
    verifier_ok: bool             # verificador fraco (visto pelo agente)
    attempts: int
    tokens: int
    route: str
    trace: list = field(default_factory=list)
    instruction: str = ""
    input: str = ""
    expected: str = ""


class ExperienceDB:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, ts REAL, harness TEXT, split TEXT, task_id TEXT, family TEXT,
            trial INT, answer TEXT, score REAL, verifier_ok INT, attempts INT, tokens INT, route TEXT, trace TEXT,
            instruction TEXT, input TEXT, expected TEXT);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, ts REAL, round INT, variant TEXT, source TEXT, harness TEXT,
            admissible INT, promoted INT, reason TEXT, detail TEXT);
        CREATE INDEX IF NOT EXISTS ep_h ON episodes(harness, split);""")

    def log_episodes(self, harness: str, split: str, eps: list[Episode]) -> None:
        now = time.time()
        self.db.executemany("INSERT INTO episodes(ts,harness,split,task_id,family,trial,answer,score,verifier_ok,attempts,tokens,route,trace,instruction,input,expected)"
                            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            [(now, harness, split, e.task_id, e.family, e.trial, e.answer, e.score, int(e.verifier_ok), e.attempts,
                              e.tokens, e.route, json.dumps(e.trace), e.instruction, e.input, e.expected) for e in eps])
        self.db.commit()

    def log_decision(self, rnd: int, variant: str, source: str, harness: str, admissible: bool, promoted: bool, reason: str, detail: dict) -> None:
        self.db.execute("INSERT INTO decisions(ts,round,variant,source,harness,admissible,promoted,reason,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                        (time.time(), rnd, variant, source, harness, int(admissible), int(promoted), reason, json.dumps(detail, default=str)))
        self.db.commit()

    def failures(self, harness: str, split: str = "evolve", n: int = 20) -> list[dict]:
        cur = self.db.execute("SELECT task_id,family,answer,instruction,input,expected FROM episodes WHERE harness=? AND split=? AND score<1 "
                              "ORDER BY id DESC LIMIT ?", (harness, split, n))
        return [dict(zip(("task_id", "family", "answer", "instruction", "input", "expected"), r)) for r in cur]

    def family_rates(self, harness: str, split: str = "evolve") -> dict:
        cur = self.db.execute("SELECT family, AVG(score), COUNT(*) FROM episodes WHERE harness=? AND split=? GROUP BY family", (harness, split))
        return {f: (round(a, 3), n) for f, a, n in cur}

    def decisions(self) -> list[dict]:
        cur = self.db.execute("SELECT round,variant,source,harness,admissible,promoted,reason FROM decisions ORDER BY id")
        return [dict(zip(("round", "variant", "source", "harness", "admissible", "promoted", "reason"), r)) for r in cur]

    def count(self, table: str = "episodes") -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
