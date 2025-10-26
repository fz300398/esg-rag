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

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ENVIRONMENT
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    logger.warning("Keine .env-Datei gefunden – Standardwerte werden verwendet.")

local_path = Path(".env.local")
if local_path.exists():
    load_dotenv(local_path, override=True)
    logger.info("Lokale .env.local geladen (überschreibt Standardwerte)")

# CONFIGURATION
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# DATA CLASS
@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    title: Optional[str] = None

logger.info(f"Lade Embedding-Modell: {EMBEDDING_MODEL}")
st_model = SentenceTransformer(EMBEDDING_MODEL)

# UTILS
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Teilt Text in überlappende Chunks."""
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

def load_pdfs(folder: Path) -> List[Chunk]:
    """Lädt alle PDFs im Ordner und zerlegt sie in Text-Chunks."""
    if not folder.exists():
        logger.error(f"Eingabeordner {folder} existiert nicht.")
        return []

    chunks: List[Chunk] = []
    pdf_files = sorted(folder.glob("**/*.pdf"))

    for pdf_path in pdf_files:
        logger.info(f"Verarbeite Datei: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            title = getattr(reader.metadata, "title", None) if reader.metadata else None

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Fehler beim Lesen von Seite {page_idx} in {pdf_path.name}: {e}")
                    continue

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
            logger.error(f"Fehler bei {pdf_path.name}: {e}")

    logger.info(f"{len(chunks)} Text-Chunks aus {len(pdf_files)} PDFs extrahiert.")
    return chunks

def build_chroma(chunks: List[Chunk], model_name: str, chroma_dir: Path, reset: bool = False) -> bool:
    """Erstellt oder erweitert den Chroma-Index mit Duplikatsschutz."""
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings())

        if reset:
            logger.warning("Chroma-Speicher wird komplett zurückgesetzt!")
            client.reset()

        coll = client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

        existing_ids = set(coll.get()["ids"]) if coll.count() > 0 else set()
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("Keine neuen Chunks zum Hinzufügen (alles bereits indexiert).")
            return True

        texts = [c.text for c in new_chunks]
        ids = [c.chunk_id for c in new_chunks]
        metas = [
            {"source": c.source, "page": c.page, "title": c.title, "chunk_id": c.chunk_id}
            for c in new_chunks
        ]

        logger.info(f"Berechne Embeddings für {len(new_chunks)} neue Chunks...")
        embeddings = st_model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

        coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

        logger.info(f"Chroma-Index in {chroma_dir} um {len(new_chunks)} neue Chunks erweitert.")
        return True
    except Exception as e:
        logger.exception("Fehler beim Aufbau des Chroma-Index")
        return False

# MAIN
def main() -> bool:
    parser = argparse.ArgumentParser(description="Ingest ESG-Dokumente in Chroma-Index.")
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--chroma_dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--reset", action="store_true", help="Bestehenden Index löschen")
    args = parser.parse_args()

    chunks = load_pdfs(args.input)
    if not chunks:
        logger.warning("Keine Textinhalte gefunden.")
        return False

    return build_chroma(chunks, args.embedding_model, args.chroma_dir, reset=args.reset)

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Ingest fehlgeschlagen.")
        raise SystemExit(1)
    logger.info("Ingest erfolgreich abgeschlossen.")