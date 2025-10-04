import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from .rag import answer

from dotenv import load_dotenv, find_dotenv
# 1) .env finden (ausgehend vom CWD)
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)

# 2) .env.local falls vorhanden – überschreibt .env
local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)

# 3) Bequeme Defaults für lokalen Modus
if os.getenv("RUN_ENV") == "local":
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("VECTOR_STORE", "chroma")

class Query(BaseModel):
    question: str


app = FastAPI(title="ESG-RAG (Phi-3 Mini)")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/query")
async def query(q: Query):
    text, ctx = answer(q.question, Path("index"), top_k=int(os.getenv("TOP_K", 6)))
    return {
        "answer": text,
        "contexts": [c.__dict__ for c in ctx],
    }
