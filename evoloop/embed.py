"""Embedders. O que é REAL e o que é substituto:
  • HTTPEmbedder / SentenceTransformerEmbedder — embeddings neurais reais (NÃO testados aqui: sem chave/pesos offline).
  • ConceptEmbedder — SUBSTITUTO determinístico para testes: léxico de sinônimos → conceitos. Prova o encanamento
    (paráfrase recuperada onde a interseção léxica falha); NÃO mede a qualidade de um embedder de verdade.
  • HashEmbedder — n-gramas de caracteres com hashing: robusto a flexão/typo, mas sem semântica (sem sinônimos).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import urllib.request
from typing import Protocol

DIM = 256
STOP = set("the of to in an is are and or with it its as by on for from that this each every any into out back keep leave "
           "apply definition literally check edge cases lesson input string".split())
CONCEPTS = {
    "reverse": "reverse reversed flip flipped backwards backward front",
    "capital": "upper uppercase capitalize capital capitals",
    "vowel": "vowel vowels e i o u",
    "digit": "digit digits numeral numerals",
    "sum": "sum add total",
    "sort": "sort sorted order alphabetical alphabetically rearrange",
    "word": "word words",
    "count": "count tally many contain",
    "shift": "shift caesar cipher move replace positions places later",
    "dup": "duplicate duplicates repeated collapse same copy followed itself occurrence",
    "longest": "longest most greatest",
}
LEX = {w: c for c, ws in CONCEPTS.items() for w in ws.split()}


def _h(s: str, salt: str = "") -> tuple[int, float]:
    d = hashlib.blake2b((salt + s).encode(), digest_size=8).digest()
    return int.from_bytes(d[:4], "little") % DIM, 1.0 if d[4] & 1 else -1.0


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ConceptEmbedder:
    name = "concept-sim"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"[a-z]+", t.lower()):
                if w in LEX:
                    i, s = _h(LEX[w], "C")
                    v[i] += 1.0 * s
                elif w not in STOP and len(w) > 2:
                    i, s = _h(w, "W")
                    v[i] += 0.25 * s
            out.append(_norm(v))
        return out


class HashEmbedder:
    name = "char3-hash"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            t = " " + re.sub(r"[^a-z ]", " ", t.lower()) + " "
            for i in range(len(t) - 2):
                j, s = _h(t[i:i + 3], "G")
                v[j] += s
            out.append(_norm(v))
        return out


class HTTPEmbedder:
    """API compatível com OpenAI `/v1/embeddings` (OpenAI, Voyage, vLLM, Ollama...). NÃO testado ao vivo."""
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("EVOLOOP_EMBED_MODEL", "text-embedding-3-small")
        self.url = (base_url or os.environ.get("EVOLOOP_EMBED_URL", "https://api.openai.com/v1")).rstrip("/") + "/embeddings"
        self.key = api_key or os.environ.get("EVOLOOP_EMBED_KEY", "")
        self.name = f"http:{self.model}"

    def embed(self, texts):
        req = urllib.request.Request(self.url, json.dumps({"model": self.model, "input": texts}).encode(),
                                     {"content-type": "application/json", "authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        return [_norm(x["embedding"]) for x in sorted(d["data"], key=lambda x: x["index"])]


class SentenceTransformerEmbedder:
    """Local, opcional (`pip install sentence-transformers`; baixa pesos). NÃO testado aqui (sem acesso a HF)."""
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self.m, self.name = SentenceTransformer(model), f"st:{model}"

    def embed(self, texts):
        return [list(map(float, v)) for v in self.m.encode(texts, normalize_embeddings=True)]


class CachedEmbedder:
    """Cache por texto (o laço reconstrói a memória a cada avaliação; sem cache seria caro com embedder remoto)."""
    def __init__(self, inner: Embedder):
        self.inner, self.name, self.cache, self.calls = inner, inner.name, {}, 0

    def embed(self, texts):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            self.calls += len(miss)
            for t, v in zip(miss, self.inner.embed(miss)):
                self.cache[t] = v
        return [self.cache[t] for t in texts]


def make_embedder(kind: str) -> Embedder | None:
    return {"lexical": None, "concept": ConceptEmbedder, "hash": HashEmbedder, "http": HTTPEmbedder,
            "st": SentenceTransformerEmbedder}[kind]() if kind != "lexical" else None
