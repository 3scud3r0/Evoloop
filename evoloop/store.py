"""Persistência do modo real: turnos (com feedback do usuário) e snapshots do workspace (desfazer)."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path


def h(s: str) -> str:
    return hashlib.sha256(s.strip().encode()).hexdigest()[:20]


class TurnStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, ts REAL, conv TEXT, harness TEXT, harness_v INT, user_text TEXT, answer TEXT,
            transcript TEXT, steps INT, tokens INT, status TEXT, score REAL, weight REAL, feedback TEXT, fb_source TEXT, snapshot TEXT, history TEXT);
        CREATE TABLE IF NOT EXISTS answers(hash TEXT PRIMARY KEY, turn_id INT);
        CREATE INDEX IF NOT EXISTS t_ts ON turns(ts);""")

    def add_turn(self, conv: str, harness: str, harness_v: int, user_text: str, answer: str, transcript: list, steps: int, tokens: int,
                 status: str, snapshot: str | None, history: list) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO turns(ts,conv,harness,harness_v,user_text,answer,transcript,steps,tokens,status,snapshot,history) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (time.time(), conv, harness, harness_v, user_text, answer, json.dumps(transcript, ensure_ascii=False), steps, tokens, status, snapshot,
                                   json.dumps(history, ensure_ascii=False)))
            tid = cur.lastrowid
            self.db.execute("INSERT OR REPLACE INTO answers(hash,turn_id) VALUES(?,?)", (h(answer), tid))
            self.db.commit()
            return tid

    def turn_for_answer(self, answer: str) -> int | None:
        with self.lock:
            r = self.db.execute("SELECT turn_id FROM answers WHERE hash=?", (h(answer),)).fetchone()
            return r[0] if r else None

    def set_feedback(self, tid: int, score: float | None, text: str, source: str, weight: float) -> None:
        with self.lock:
            self.db.execute("UPDATE turns SET score=?, feedback=?, fb_source=?, weight=? WHERE id=?", (score, text, source, weight, tid))
            self.db.commit()

    def get(self, tid: int) -> dict | None:
        with self.lock:
            cur = self.db.execute("SELECT * FROM turns WHERE id=?", (tid,))
            cols = [c[0] for c in cur.description]
            r = cur.fetchone()
        return dict(zip(cols, r)) if r else None

    def query(self, where: str = "1=1", args: tuple = (), limit: int = 200) -> list[dict]:
        with self.lock:
            cur = self.db.execute(f"SELECT * FROM turns WHERE {where} ORDER BY id DESC LIMIT ?", (*args, limit))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]


class Workspace:
    """Snapshots do diretório de trabalho (o agente tem escrita nele): permite /undo após um `rm -rf` do agente."""
    def __init__(self, root: str | Path, state: str | Path, max_mb: int = 500, keep: int = 20):
        self.root, self.snaps = Path(root).resolve(), Path(state).resolve() / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True); self.snaps.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.keep = max_mb << 20, keep

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file() and not f.is_symlink())

    def snapshot(self, label: str) -> str | None:
        if self.size() > self.max_bytes:
            return None                       # grande demais: sem snapshot (o chamador avisa o usuário)
        sid = f"{int(time.time() * 1000)}-{label}"
        shutil.copytree(self.root, self.snaps / sid, symlinks=True)
        for old in sorted(self.snaps.iterdir())[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)
        return sid

    def restore(self, sid: str) -> bool:
        src = self.snaps / sid
        if not src.is_dir():
            return False
        for c in self.root.iterdir():
            shutil.rmtree(c) if c.is_dir() and not c.is_symlink() else c.unlink()
        for c in src.iterdir():
            shutil.copytree(c, self.root / c.name, symlinks=True) if c.is_dir() else shutil.copy2(c, self.root / c.name)
        return True
