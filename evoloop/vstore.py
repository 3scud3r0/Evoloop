"""Vector stores. BruteForceStore (Python puro, exato, zero deps) e ChromaStore (reusa o ChromaVectorStore do GEPA vendorizado:
third_party/gepa/adapters/generic_rag_adapter). Qdrant/LanceDB/Milvus/Weaviate do GEPA seguem a mesma VectorStoreInterface;
para trocá-los basta um wrapper como ChromaStore (upsert é específico de cada cliente)."""
from __future__ import annotations
from typing import Protocol

class VectorStore(Protocol):
    def upsert(self, key: str, text: str, vec: list[float]) -> None: ...
    def search(self, qvec: list[float], k: int) -> list[tuple[str, float]]: ...
    def __len__(self) -> int: ...


class BruteForceStore:
    kind = "brute"

    def __init__(self):
        self.keys: list[str] = []
        self.texts: list[str] = []
        self.mat: list[list[float]] = []

    def upsert(self, key, text, vec):
        v = list(map(float, vec))
        if key in self.keys:
            i = self.keys.index(key)
            self.texts[i], self.mat[i] = text, v
            return
        self.keys.append(key); self.texts.append(text)
        self.mat.append(v)

    def search(self, qvec, k):
        if not self.keys:
            return []
        q = list(map(float, qvec))
        sc = [sum(a * b for a, b in zip(row, q)) for row in self.mat]  # cosseno (vetores normalizados)
        idx = sorted(range(len(sc)), key=lambda i: (-float(sc[i]), self.keys[i]))[:k]
        return [(self.texts[i], float(sc[i])) for i in idx]

    def __len__(self):
        return len(self.keys)


class ChromaStore:
    kind = "chroma"

    def __init__(self, name: str = "evoloop_mem", persist: str | None = None):
        import chromadb  # noqa: PLC0415
        from gepa.adapters.generic_rag_adapter.vector_stores.chroma_store import ChromaVectorStore  # noqa: PLC0415
        import uuid
        client = chromadb.PersistentClient(path=persist) if persist else chromadb.EphemeralClient()
        cname = f"{name}_{uuid.uuid4().hex[:8]}"
        client.create_collection(cname, metadata={"hnsw:space": "cosine"})     # antes do wrapper do GEPA (que faz get_collection)
        self.store = ChromaVectorStore(client, cname, embedding_function=None)
        self.n = 0

    def upsert(self, key, text, vec):
        before = self.store.collection.count()
        self.store.collection.upsert(ids=[key], documents=[text], embeddings=[list(map(float, vec))])
        self.n = self.store.collection.count()

    def search(self, qvec, k):
        if self.n == 0:
            return []
        res = self.store.vector_search(list(map(float, qvec)), k=min(k, self.n))
        return [(r["content"], float(r["score"])) for r in res]

    def __len__(self):
        return self.n


def make_store(kind: str) -> VectorStore:
    return ChromaStore() if kind == "chroma" else BruteForceStore()
