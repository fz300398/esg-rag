import os, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import chromadb
import numpy as np
import requests
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    chunk_id: str

def load_chroma(chroma_dir: Path):
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings())
    return client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode([query], normalize_embeddings=True)[0].astype("float32")

def retrieve(query: str, coll, model: SentenceTransformer, k: int = 6) -> List[Retrieved]:
    q_emb = embed_query(query, model).tolist()
    res = coll.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return [Retrieved(text=doc, source=m.get("source","?"), page=int(m.get("page",0)), chunk_id=m.get("chunk_id","?")) for doc,m in zip(docs,metas)]

def answer(question: str, top_k: int = 6) -> Tuple[str, List[Retrieved]]:
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    coll = load_chroma(CHROMA_DIR)
    ctx = retrieve(question, coll, st_model, k=top_k*2)
    pairs = [(question, c.text) for c in ctx]
    scores = reranker.predict(pairs).tolist()
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    ctx = [ctx[i] for i in order[:top_k]]
    prompt = f"Frage: {question}\n\n" + "\n---\n".join(c.text for c in ctx)
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_MODEL,"prompt":prompt}, timeout=600)
    r.raise_for_status()
    return r.json().get("response",""), ctx
