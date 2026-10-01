"""Registry versionado de harnesses com promoção e ROLLBACK (append-only: nada é apagado, só muda de status)."""
from __future__ import annotations
import sqlite3
import time

from .harness import Harness


class Registry:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS versions(v INTEGER PRIMARY KEY AUTOINCREMENT, digest TEXT, parent INT, json TEXT, note TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, v INT, event TEXT, why TEXT);
        CREATE TABLE IF NOT EXISTS pending(v INTEGER PRIMARY KEY, ts REAL, status TEXT, evidence TEXT);
        """)

    def add(self, h: Harness, parent: int | None, why: str = "") -> int:
        cur = self.db.execute("INSERT INTO versions(digest,parent,json,note,ts) VALUES(?,?,?,?,?)", (h.digest(), parent, h.to_json(), h.note, time.time()))
        v = cur.lastrowid
        self._log(v, "registered", why)
        return v

    def _log(self, v: int, event: str, why: str = "") -> None:
        self.db.execute("INSERT INTO log(ts,v,event,why) VALUES(?,?,?,?)", (time.time(), v, event, why))
        self.db.commit()

    def promote(self, v: int, why: str = "") -> None:
        self._log(v, "promoted", why)

    def reject(self, v: int, why: str = "") -> None:
        self._log(v, "rejected", why)

    def champion(self) -> int | None:
        """Versão vigente = última promoção que não foi revertida."""
        stack: list[int] = []
        for v, ev in self.db.execute("SELECT v,event FROM log WHERE event IN ('promoted','rolled_back') ORDER BY id"):
            if ev == "promoted":
                stack.append(v)
            elif ev == "rolled_back" and v in stack:
                stack.remove(v)
        return stack[-1] if stack else None

    def rollback(self, why: str = "") -> int | None:
        """Reverte a promoção vigente; devolve a nova campeã (anterior)."""
        cur = self.champion()
        if cur is None:
            return None
        self._log(cur, "rolled_back", why)
        return self.champion()

    def get(self, v: int) -> Harness:
        return Harness.from_json(self.db.execute("SELECT json FROM versions WHERE v=?", (v,)).fetchone()[0])

    def history(self) -> list[dict]:
        cur = self.db.execute("SELECT ts,v,event,why FROM log ORDER BY id")
        return [dict(zip(("ts", "v", "event", "why"), r)) for r in cur]

    # ---- candidatos pendentes de aprovação humana (self-modification NÃO é aplicada sozinha por padrão) ----------------
    def add_pending(self, h: Harness, parent: int | None, evidence: dict) -> int:
        import json
        v = self.add(h, parent, "pending")
        self.db.execute("INSERT INTO pending(v,ts,status,evidence) VALUES(?,?,?,?)", (v, time.time(), "pending", json.dumps(evidence, default=str)))
        self.db.commit()
        return v

    def pending(self) -> list[dict]:
        import json
        cur = self.db.execute("SELECT v,ts,status,evidence FROM pending WHERE status='pending' ORDER BY v")
        return [{"v": v, "ts": ts, "status": st, "evidence": json.loads(ev)} for v, ts, st, ev in cur]

    def resolve_pending(self, v: int, status: str) -> dict | None:
        import json
        row = self.db.execute("SELECT evidence FROM pending WHERE v=? AND status='pending'", (v,)).fetchone()
        if not row:
            return None
        self.db.execute("UPDATE pending SET status=? WHERE v=?", (status, v))
        self.db.commit()
        return json.loads(row[0])
