import os, json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import chromadb
import numpy as np
import requests
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder

# === CONFIGURATION ===
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_STREAM = os.getenv("OLLAMA_STREAM", "false").lower() == "true"

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === MODEL LOADING (only once) ===
logger.info("Loading embedding model...")
st_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Loading reranker model...")
reranker = CrossEncoder(RERANKER_MODEL)

# === CHROMA CLIENT ===
def load_chroma(chroma_dir: Path):
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings())
    return client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

collection = load_chroma(CHROMA_DIR)

# === DATA STRUCTURES ===
@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    chunk_id: str

# === CORE FUNCTIONS ===
def embed_query(query: str) -> np.ndarray:
    """Wandelt die Nutzerfrage in einen semantischen Vektor um."""
    return st_model.encode([query], normalize_embeddings=True)[0].astype("float32")

def retrieve(query: str, k: int = 6, min_results: int = 2) -> List[Retrieved]:
    """Ruft relevante Dokumentpassagen aus ChromaDB ab."""
    q_emb = embed_query(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs or len(docs) < min_results:
        logger.warning("Zu wenige oder keine relevanten Dokumente gefunden – Fallback aktiv.")
        return []

    return [
        Retrieved(
            text=doc,
            source=m.get("source", "?"),
            page=int(m.get("page", 0)),
            chunk_id=m.get("chunk_id", "?")
        )
        for doc, m in zip(docs, metas)
    ]

def query_ollama(prompt: str) -> str:
    """Sendet den Prompt an Ollama und gibt die Antwort zurück."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": OLLAMA_STREAM},
            timeout=120,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        logger.info("Antwort von Ollama erfolgreich empfangen.")
        return response
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return "Beim Abrufen der Antwort vom LLM ist ein Fehler aufgetreten."
        
def answer(question: str, top_k: int = 6, fallback_threshold: int = 2) -> Tuple[str, List[Retrieved]]:
    """Hauptfunktion: kombiniert Retrieval, Reranking und Antwortgenerierung."""
    logger.info(f"Received question: {question}")

    ctx = retrieve(question, k=top_k * 2, min_results=fallback_threshold)

    # === Fallback: Kein Kontext gefunden ===
    if not ctx:
        logger.info("Kein Kontext gefunden. Fallback auf reines LLM aktiviert.")
        fallback_prompt = (
            f"Beantworte die folgende Frage zum Thema ESG so präzise und informativ wie möglich:\n\n"
            f"Frage: {question}\n\nAntwort:"
        )
        return query_ollama(fallback_prompt), []

    # === Reranking ===
    pairs = [(question, c.text) for c in ctx]
    scores = reranker.predict(pairs).tolist()
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    ctx = [ctx[i] for i in order[:top_k]]

    # === Kontextualisierter Prompt ===
    context_text = "\n---\n".join(c.text for c in ctx)
    prompt = (
        f"Frage: {question}\n\n"
        f"Kontext (Ausschnitte aus ESG-Dokumenten):\n{context_text}\n\n"
        f"Formuliere eine strukturierte, gut verständliche Antwort basierend auf diesem Kontext.\n\nAntwort:"
    )

    response = query_ollama(prompt)
    return response, ctx