import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from rag import answer
from ingest import load_pdfs, build_chroma

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("esg-backend")

# Environment Setup
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)

local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)
    logger.info("Lokale .env.local geladen (überschreibt Standardwerte)")

# Standardwerte setzen
os.environ.setdefault("RUN_ENV", "local")
os.environ.setdefault("VECTOR_STORE", "chroma")
os.environ.setdefault("USE_RERANKER", "true")

# Verzeichnisse & Modelle
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# FastAPI Setup
app = FastAPI(title="ESG Assistant API", version="1.1.0")

# CORS (Frontend-Kommunikation zulassen)
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
# Healthcheck
@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "msg": "Backend läuft!",
        "reranker_enabled": os.getenv("USE_RERANKER", "true"),
        "embedding_model": EMBEDDING_MODEL,
        "chroma_dir": str(CHROMA_DIR),
    }

# RAG Query
@app.post("/query")
async def query(q: Query):
    """Verarbeitet eine Benutzerfrage über das RAG-System."""
    logger.info(f"Neue Anfrage erhalten: {q.question}")
    try:
        answer_text, ctx = answer(q.question, top_k=int(os.getenv("TOP_K", 6)))
        return {
            "answer": answer_text,
            "contexts": [c.__dict__ for c in ctx],
        }
    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung: {e}")
        raise HTTPException(status_code=500, detail="Fehler bei der Anfrageverarbeitung.")

# PDF Upload & Index Update
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Lädt ein ESG-PDF hoch und aktualisiert den Chroma-Index."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DATA_DIR / file.filename

        with open(target_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Datei '{file.filename}' erfolgreich hochgeladen.")

        chunks = load_pdfs(DATA_DIR)
        if not chunks:
            raise HTTPException(status_code=400, detail="Keine Textinhalte im PDF gefunden.")

        success = build_chroma(chunks, EMBEDDING_MODEL, CHROMA_DIR)
        if not success:
            raise HTTPException(status_code=500, detail="Fehler beim Aktualisieren des Index.")

        logger.info("Index erfolgreich aktualisiert.")
        return {"ok": True, "msg": f"Datei '{file.filename}' hochgeladen und Index aktualisiert."}

    except Exception as e:
        logger.error(f"Upload/Indexierung fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")

# Root Endpoint
@app.get("/")
async def root():
    return {
        "msg": "ESG Assistant Backend API",
        "endpoints": ["/healthz", "/query", "/upload"],
        "data_dir": str(DATA_DIR),
        "chroma_dir": str(CHROMA_DIR),
    }