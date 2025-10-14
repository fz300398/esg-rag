import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from rag import answer
from ingest import load_pdfs, build_chroma

# Environment Setup
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)

local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)

if os.getenv("RUN_ENV") == "local":
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("VECTOR_STORE", "chroma")

# Wichtige Pfade
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# FastAPI Setup
app = FastAPI(title="ESG Assistant API")

# CORS-Konfiguration für Angular-Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Query(BaseModel):
    question: str

# Healthcheck Endpoint
@app.get("/healthz")
async def healthz():
    """Überprüft, ob Backend, Modelle und Pfade aktiv sind."""
    return {
        "ok": True,
        "msg": "Backend läuft!",
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": os.getenv("VECTOR_STORE", "chroma"),
        "reranker_enabled": os.getenv("USE_RERANKER", "false"),
        "data_dir": str(DATA_DIR),
        "chroma_dir": str(CHROMA_DIR)
    }

# Query Endpoint
@app.post("/query")
async def query(q: Query):
    """
    Hauptendpunkt für das Angular-Frontend.
    Führt RAG + (optional) Reranking + Phi-3 Generierung aus.
    """
    text, ctx = answer(q.question, top_k=int(os.getenv("TOP_K", 6)))
    return {
        "answer": text,
        "contexts": [c.__dict__ for c in ctx],
    }

# Upload Endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Lädt ein ESG-Dokument (PDF) hoch,
    speichert es in /app/data und aktualisiert den Chroma-Index.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DATA_DIR / file.filename

    # Datei speichern
    with open(target_path, "wb") as f:
        f.write(await file.read())

    # Chroma neu aufbauen
    chunks = load_pdfs(DATA_DIR)
    if not chunks:
        return {"ok": False, "msg": "Keine lesbaren Texte in PDF gefunden."}

    build_chroma(chunks, EMBEDDING_MODEL, CHROMA_DIR)

    return {
        "ok": True,
        "msg": f"Datei '{file.filename}' hochgeladen und Index aktualisiert.",
    }

# Root Endpoint
@app.get("/")
async def root():
    """Übersicht der verfügbaren Endpunkte."""
    return {
        "msg": "ESG Assistant API",
        "endpoints": ["/healthz", "/query", "/upload"],
        "status": "bereit",
    }
