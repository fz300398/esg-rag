from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import requests
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder

# Environment Configuration
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K = int(os.getenv("TOP_K", "6"))

# Reranker
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes"}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "40"))

# LLM (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# Data Structures
@dataclass
class Retrieved:
    """Ein einzelner Textabschnitt (Chunk), der aus der Datenbank abgerufen wurde."""
    text: str
    source: str
    page: int
    chunk_id: str

# Embeddings & Retrieval
def load_chroma(chroma_dir: Path):
    """Lädt oder erstellt eine persistente Chroma-Collection."""
    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(allow_reset=False)
        )
        return client.get_or_create_collection(
            name="esg_docs",
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden von ChromaDB: {e}")

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Erzeugt ein normalisiertes Embedding für eine Textanfrage."""
    return model.encode([query], normalize_embeddings=True)[0].astype("float32")

def retrieve_chroma(query: str, coll, model: SentenceTransformer, k: int = 6) -> List[Retrieved]:
    """Sucht relevante Textabschnitte (Chunks) in der Chroma-Datenbank."""
    q_emb = embed_query(query, model).tolist()
    res = coll.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    results: List[Retrieved] = []
    for i, doc in enumerate(docs):
        m = metas[i] if i < len(metas) else {}
        results.append(
            Retrieved(
                text=doc or "",
                source=str(m.get("source", "unknown")),
                page=int(m.get("page", 0) or 0),
                chunk_id=str(m.get("chunk_id", f"chroma-{i}")),
            )
        )
    print(f"{len(results)} Chunks aus Chroma geladen.")
    return results

# Reranker
_reranker: CrossEncoder | None = None

def get_reranker() -> CrossEncoder | None:
    """Lädt den CrossEncoder nur einmal (lazy load)."""
    global _reranker
    if _reranker is None and USE_RERANKER:
        print(f"Lade Reranker-Modell: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_local(query: str, ctx: List[Retrieved]) -> List[Retrieved]:
    """Bewertet Dokumenten-Chunks anhand ihrer Relevanz zur Anfrage."""
    ce = get_reranker()
    if ce is None:
        return ctx
    pairs = [(query, c.text) for c in ctx]
    print(f"🔎 Reranke {len(pairs)} Kandidaten …")
    scores = ce.predict(pairs).tolist()
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    ranked = [ctx[i] for i in order]
    print("Reranking abgeschlossen.")
    return ranked


# LLM
def call_ollama(prompt: str) -> str:
    """Sendet den Prompt an das lokale Ollama-LLM."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9},
    }

    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except Exception as e:
        print(f"Fehler beim Ollama-Aufruf: {e}")
        return "Fehler bei der Antwortgenerierung."


# Prompt Construction
SYSTEM_PROMPT = (
    "Du bist ein präziser ESG-Assistent. Antworte **auf Deutsch**. "
    "Nutze ausschließlich den bereitgestellten Kontext. "
    "Wenn etwas nicht im Kontext steht, sage ehrlich, dass die Information fehlt. "
    "Formatiere klar und **zitiere** jede wesentliche Aussage mit [Quelle, Seite]."
)

def build_prompt(question: str, contexts: List[Retrieved]) -> str:
    """Erzeugt den kombinierten Prompt aus Frage und Kontext."""
    blocks = []
    for i, c in enumerate(contexts, start=1):
        header = f"[Dok {i}] {c.source} (Seite {c.page}) — {c.chunk_id}"
        blocks.append(f"{header}\n{c.text}")

    context_text = "\n\n---\n".join(blocks) if blocks else "(kein Kontext gefunden)"
    user = (
        f"Frage: {question}\n\n"
        f"Kontextauszüge (verwende nur, was relevant ist):\n{context_text}\n\n"
        "Gib am Ende eine Liste 'Quellen' mit [Dateiname, Seite] an."
    )

    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user}\n<|assistant|>"

# Full RAG Pipeline
def answer(question: str, top_k: int = TOP_K) -> Tuple[str, List[Retrieved]]:
    """Führt Retrieval, (optional) Reranking und Generierung aus."""
    print(f"Frage: {question}")

    # Embedding Model laden
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    coll = load_chroma(CHROMA_DIR)

    # Kontextpassagen abrufen
    pre_k = RERANKER_CANDIDATES if USE_RERANKER else top_k
    ctx = retrieve_chroma(question, coll, st_model, k=pre_k)

    # Optionales Reranking
    if USE_RERANKER and ctx:
        ctx = rerank_local(question, ctx)[:top_k]
    else:
        ctx = ctx[:top_k]

    # Prompt bauen & LLM aufrufen
    prompt = build_prompt(question, ctx)
    response = call_ollama(prompt)

    # Rückgabe
    print("Antwort generiert.")
    return response, ctx