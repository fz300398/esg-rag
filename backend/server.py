import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from rag import answer
from ingest import load_pdfs, build_chroma
from typing import Dict, List
import uuid

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("esg-backend")

# === ENVIRONMENT ===
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)

local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)
    logger.info("Lokale .env.local geladen (überschreibt Standardwerte)")

# === CONFIG ===
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# === FASTAPI ===
app = FastAPI(title="ESG Assistant API", version="1.2.0")

# === Simple in-memory session store ===
session_store: Dict[str, List[Dict[str, str]]] = {}

# === CORS (für Frontend) ===
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
    "http://esg-frontend:4200" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MODELS ===
class Query(BaseModel):
    session_id: str | None = None
    question: str

# === HEALTH ===
@app.get("/healthz")
async def healthz():
    """Healthcheck-Endpunkt."""
    return {
        "ok": True,
        "msg": "Backend läuft!",
        "embedding_model": EMBEDDING_MODEL,
        "chroma_dir": str(CHROMA_DIR),
    }

# === RAG QUERY ===
@app.post("/query")
async def query(q: Query):
    """Empfängt eine Frage, verarbeitet sie kontextsensitiv und liefert Antwort + Quellen."""
    logger.info(f"Neue Anfrage: {q.question}")

    try:
        # Session-ID prüfen oder neu erzeugen
        session_id = q.session_id or str(uuid.uuid4())

        # Verlauf aus Speicher holen
        history = session_store.get(session_id, [])

        # Kontext aus bisherigen Nachrichten extrahieren
        context_text = "\n".join([
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-5:]
        ])

        # RAG-Antwort mit Kontext holen
        answer_text, ctx, confidence = answer(q.question, history=context_text)

        # Verlauf aktualisieren
        history.append({"role": "user", "content": q.question})
        history.append({"role": "assistant", "content": answer_text})
        session_store[session_id] = history

        # Antwort zurückgeben
        return {
            "session_id": session_id,
            "answer": answer_text,
            "confidence": confidence,
            "sources": [
                {
                    "text": c.text[:400] + ("..." if len(c.text) > 400 else ""),
                    "source": c.source,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                }
                for c in ctx
            ],
        }

    except Exception as e:
        logger.exception("Fehler bei der Verarbeitung der Anfrage")
        raise HTTPException(status_code=500, detail=str(e))

# === UPLOAD & INDEX ===
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Lädt eine PDF hoch und aktualisiert den Chroma-Index."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        target_path = DATA_DIR / file.filename

        with open(target_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Datei '{file.filename}' hochgeladen.")

        chunks = load_pdfs(DATA_DIR)
        if not chunks:
            raise HTTPException(status_code=400, detail="Keine Textinhalte im PDF gefunden.")

        build_chroma(chunks, EMBEDDING_MODEL, CHROMA_DIR)
        logger.info("Index erfolgreich aktualisiert.")
        return {"ok": True, "msg": f"Datei '{file.filename}' wurde indexiert."}

    except Exception as e:
        logger.exception("Fehler beim Upload oder Indexaufbau")
        raise HTTPException(status_code=500, detail=f"Fehler: {str(e)}")

# === ROOT ===
@app.get("/")
async def root():
    return {
        "msg": "ESG Assistant Backend API",
        "endpoints": ["/healthz", "/query", "/upload"],
        "data_dir": str(DATA_DIR),
        "chroma_dir": str(CHROMA_DIR),
    }