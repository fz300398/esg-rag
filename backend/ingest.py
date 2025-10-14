import argparse
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import chromadb
from chromadb.config import Settings as ChromaSettings

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Environment Setup
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    logger.warning("⚠️ Keine .env-Datei gefunden – Standardwerte werden verwendet.")

local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)
    logger.info("Lokale .env.local geladen (überschreibt Standardwerte)")

if os.getenv("RUN_ENV") == "local":
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("VECTOR_STORE", "chroma")

# Konfiguration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Datenklasse
@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    title: Optional[str] = None

# Text in überlappende Chunks teilen
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join((text or "").split())
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

# PDF-Dateien laden & zerlegen
def load_pdfs(folder: Path) -> List[Chunk]:
    if not folder.exists():
        logger.error(f"Eingabeordner {folder} existiert nicht.")
        return []

    chunks: List[Chunk] = []
    for pdf_path in sorted(folder.glob("**/*.pdf")):
        try:
            logger.info(f"Lese Datei: {pdf_path.name}")
            reader = PdfReader(str(pdf_path))
            title = None

            md = getattr(reader, "metadata", None)
            if md and isinstance(md, dict) and md.get("title"):
                title = md.get("title")
            elif hasattr(md, "title"):
                title = md.title

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Fehler beim Lesen von Seite {page_idx}: {e}")
                    text = ""

                if not text.strip():
                    continue

                for i, c in enumerate(chunk_text(text)):
                    chunks.append(
                        Chunk(
                            text=c,
                            source=pdf_path.name,
                            page=page_idx,
                            chunk_id=f"{pdf_path.stem}-p{page_idx}-c{i}",
                            title=title,
                        )
                    )

        except Exception as e:
            logger.error(f"Konnte {pdf_path} nicht verarbeiten: {e}")

    logger.info(f"{len(chunks)} Text-Chunks extrahiert.")
    return chunks

# Chroma Index erstellen
def build_chroma(chunks: List[Chunk], model_name: str, chroma_dir: Path) -> bool:
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        st = SentenceTransformer(model_name)

        client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings(allow_reset=True))
        client.reset()
        coll = client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metas = [{"source": c.source, "page": c.page, "title": c.title, "chunk_id": c.chunk_id} for c in chunks]

        embeddings = st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).tolist()
        coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

        logger.info(f"Chroma-Index erfolgreich in {chroma_dir} erstellt.")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Chroma-Index: {e}")
        return False

# Hauptfunktion
def main() -> bool:
    parser = argparse.ArgumentParser(description="Ingest ESG-Dokumente in Chroma-Index.")
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--chroma_dir", type=Path, default=CHROMA_DIR)
    args = parser.parse_args()

    chunks = load_pdfs(args.input)
    if not chunks:
        logger.warning("Keine Texte gefunden.")
        return False

    return build_chroma(chunks, args.embedding_model, args.chroma_dir)

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Ingest fehlgeschlagen.")
        raise SystemExit(1)
    logger.info("Ingest erfolgreich abgeschlossen.")