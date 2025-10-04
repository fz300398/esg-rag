from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# <<< wichtig: Store-Schalter >>>
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma").lower()

if VECTOR_STORE == "faiss":
    import faiss  # type: ignore
elif VECTOR_STORE == "chroma":
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore
else:
    raise RuntimeError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")

@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    chunk_id: str


def load_faiss(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return index, meta


def load_chroma(chroma_dir: Path):
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings())
    coll = client.get_or_create_collection(name="esg", metadata={"hnsw:space": "cosine"})
    return coll


def embed_query(q: str, model: SentenceTransformer) -> np.ndarray:
    v = model.encode([q], normalize_embeddings=True)
    return np.array(v, dtype="float32")


def retrieve_faiss(query: str, index, meta, model: SentenceTransformer, k: int = 6) -> List[Retrieved]:
    q = embed_query(query, model)
    scores, ids = index.search(q, k)
    out: List[Retrieved] = []
    for idx in ids[0]:
        m = meta[int(idx)]
        out.append(Retrieved(text=m["text"], source=m["source"], page=m["page"], chunk_id=m["chunk_id"]))
    return out


def retrieve_chroma(query: str, coll, model: SentenceTransformer, k: int = 6) -> List[Retrieved]:
    q = embed_query(query, model)[0].tolist()
    res = coll.query(query_embeddings=[q], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out: List[Retrieved] = []
    for i, doc in enumerate(docs):
        m = metas[i] if i < len(metas) else {}
        out.append(
            Retrieved(
                text=doc or "",
                source=str(m.get("source", "unknown")),
                page=int(m.get("page", 0) or 0),
                chunk_id=str(m.get("chunk_id", f"chroma-{i}")),
            )
        )
    return out


# --------- Reranker (optional) ---------
def use_reranker() -> bool:
    return os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes"}


def rerank_http(query: str, ctx: List[Retrieved]) -> List[Retrieved]:
    url = os.getenv("RERANKER_URL", "http://reranker:9000").rstrip("/")
    payload = {"query": query, "candidates": [{"text": c.text} for c in ctx]}
    r = requests.post(f"{url}/rerank", json=payload, timeout=120)
    r.raise_for_status()
    scores = r.json().get("scores", [])
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    return [ctx[i] for i in order]


# --------- LLM (Ollama) ---------
def call_ollama(prompt: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "top_p": 0.9}}
    r = requests.post(f"{base}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


# --------- Prompting ---------
SYSTEM_PROMPT = (
    "Du bist ein präziser ESG-Assistent. Antworte **auf Deutsch**. "
    "Nutze ausschließlich den bereitgestellten Kontext. "
    "Wenn etwas nicht im Kontext steht, sage ehrlich, dass die Information fehlt. "
    "Formatiere klar und **zitiere** jede wesentliche Aussage mit [Quelle, Seite]."
)


def build_prompt(question: str, contexts: List[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        header = f"[Dok {i}] {c.source} (Seite {c.page}) — {c.chunk_id}"
        blocks.append(header + "\n" + c.text)
    context = "\n\n---\n".join(blocks) if blocks else "(kein Kontext gefunden)"
    user = (
        f"Frage: {question}\n\n"
        f"Kontextauszüge (verwende nur, was relevant ist):\n{context}\n\n"
        "Gib am Ende eine Liste 'Quellen' mit [Dateiname, Seite] an."
    )
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user}\n<|assistant|>"
    return prompt


# --------- End-to-End ---------
def answer(question: str, index_dir: Path, top_k: int = 6) -> Tuple[str, List[Retrieved]]:
    st_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))

    # Vorauswahlgröße
    final_k = int(os.getenv("TOP_K", top_k))
    pre_k = max(int(os.getenv("RERANKER_CANDIDATES", 20)), final_k) if use_reranker() else final_k

    # Retrieve je nach Store
    if VECTOR_STORE == "faiss":
        index, meta = load_faiss(index_dir)
        ctx = retrieve_faiss(question, index, meta, st_model, k=pre_k)
    elif VECTOR_STORE == "chroma":
        coll = load_chroma(Path(os.getenv("CHROMA_DIR", "chroma")))
        ctx = retrieve_chroma(question, coll, st_model, k=pre_k)
    else:
        raise RuntimeError(f"Unsupported VECTOR_STORE: {VECTOR_STORE}")

    # Rerank
    if use_reranker() and ctx:
        ctx = rerank_http(question, ctx)[:final_k]
    else:
        ctx = ctx[:final_k]

    # LLM
    prompt = build_prompt(question, ctx)
    out = call_ollama(prompt)
    return out, ctx
