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
    return st_model.encode([query], normalize_embeddings=True)[0].astype("float32")

def retrieve(query: str, k: int = 6) -> List[Retrieved]:
    q_emb = embed_query(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return [
        Retrieved(
            text=doc,
            source=m.get("source", "?"),
            page=int(m.get("page", 0)),
            chunk_id=m.get("chunk_id", "?")
        )
        for doc, m in zip(docs, metas)
    ]

def answer(question: str, top_k: int = 6) -> Tuple[str, List[Retrieved]]:
    logger.info(f"Received question: {question}")

    ctx = retrieve(question, k=top_k * 2)
    if not ctx:
        return "Ich konnte dazu leider keine Informationen finden.", []

    # Reranking
    pairs = [(question, c.text) for c in ctx]
    scores = reranker.predict(pairs).tolist()
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    ctx = [ctx[i] for i in order[:top_k]]

    # Construct prompt
    context_text = "\n---\n".join(c.text for c in ctx)
    prompt = f"Frage: {question}\n\nKontext:\n{context_text}\n\nAntwort:"

    # Query Ollama
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            timeout=120,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        logger.info("Ollama response received.")
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        response = "Beim Abrufen der Antwort vom LLM ist ein Fehler aufgetreten."

    return response, ctx