"""Memória persistente (SQLite = fonte da verdade + hash/congelamento) com recuperação LÉXICA (baseline) ou VETORIAL
(embedder + vector store). Congelamento verificável por hash (disciplina do RSIAgent: sem writeback no teste)."""
from __future__ import annotations
import hashlib
import re
import sqlite3
import threading

from .embed import Embedder
from .vstore import make_store


def _toks(s: str) -> set:
    return set(re.findall(r"[a-z]{3,}", s.lower()))


class Memory:
    def __init__(self, path: str = ":memory:", embedder: Embedder | None = None, store: str = "brute", min_score: float = 0.25):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute("CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, kind TEXT, key TEXT UNIQUE, text TEXT)")
        self.frozen_hash: str | None = None
        self.embedder, self.store_kind, self.min_score = embedder, store, min_score
        self.vs = make_store(store) if embedder else None
        if self.vs is not None:
            for _, key, text in self.items():
                self.vs.upsert(key, text, self.embedder.embed([text])[0])

    @property
    def backend(self) -> str:
        return f"vector:{self.embedder.name}/{self.store_kind}" if self.embedder else "lexical"

    def _remember(self, kind: str, key: str, text: str) -> None:
        if self.frozen_hash is not None:
            raise PermissionError("memória congelada: writeback desabilitado")
        self.db.execute("INSERT INTO facts(kind,key,text) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET text=excluded.text",
                        (kind, key, text))
        self.db.commit()
        if self.vs is not None:
            self.vs.upsert(key, text, self.embedder.embed([text])[0])

    def _recall(self, query: str, k: int = 2) -> list[str]:
        if self.vs is not None:
            hits = self.vs.search(self.embedder.embed([query])[0], k)
            return [t for t, s in hits if s >= self.min_score]
        q = _toks(query)
        rows = self.db.execute("SELECT text FROM facts").fetchall()
        sc = sorted(((len(q & _toks(t)), t) for (t,) in rows), key=lambda x: (-x[0], x[1]))
        return [t for s, t in sc[:k] if s > 0]

    def _digest(self) -> str:
        rows = self.db.execute("SELECT kind,key,text FROM facts ORDER BY key").fetchall()
        return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]

    def freeze(self) -> str:
        self.frozen_hash = self.digest()
        return self.frozen_hash

    def snapshot(self) -> "Memory":
        """Cópia congelada (para avaliar candidatos sem contaminar a memória viva)."""
        m = Memory(embedder=self.embedder, store=self.store_kind, min_score=self.min_score)
        for kind, key, text in self.items():
            m.remember(kind, key, text)
        m.freeze()
        return m

    def _len(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def _items(self) -> list[tuple]:
        return self.db.execute("SELECT kind,key,text FROM facts ORDER BY key").fetchall()

    # ---- wrappers thread-safe (servidor multi-thread) ----
    def remember(self, kind, key, text):
        with self.lock:
            return self._remember(kind, key, text)

    def recall(self, query, k=2):
        with self.lock:
            return self._recall(query, k)

    def digest(self):
        with self.lock:
            return self._digest()

    def items(self):
        with self.lock:
            return self._items()

    def __len__(self):
        with self.lock:
            return self._len()
