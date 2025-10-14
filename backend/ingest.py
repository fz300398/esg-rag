import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings

# Environment Setup
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)

# Lokale Default-Werte
if os.getenv("RUN_ENV") == "local":
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# Konfiguration aus .env
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Datenmodell
@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    title: str | None = None

# Chunking
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Teilt Text in überlappende Chunks."""
    text = " ".join((text or "").split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

# PDF-Verarbeitung
def load_pdfs(folder: Path) -> List[Chunk]:
    """Liest alle PDFs im angegebenen Ordner ein und erstellt Text-Chunks."""
    chunks: List[Chunk] = []
    pdf_files = sorted(folder.glob("**/*.pdf"))

    if not pdf_files:
        print(f"Keine PDF-Dateien in {folder} gefunden.")
        return []

    for pdf_path in pdf_files:
        print(f"Verarbeite Datei: {pdf_path.name}")
        try:
            reader = PdfReader(str(pdf_path))
            title = getattr(getattr(reader, "metadata", None), "title", None)

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

                if not text.strip():
                    continue  # Leere Seiten überspringen

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
            print(f"Fehler beim Lesen von {pdf_path}: {e}")

    print(f"Insgesamt {len(chunks)} Text-Chunks extrahiert.")
    return chunks

# Chroma Index-Erstellung
def build_chroma(chunks: List[Chunk], model_name: str, chroma_dir: Path):
    """Erstellt eine persistente Chroma-Collection mit Embeddings."""
    print(f"Erstelle ChromaDB-Collection in {chroma_dir} …")
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Chroma-Client initialisieren
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings(allow_reset=True))
    client.reset()  # löscht alte Collections
    collection = client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

    # Embedding-Modell laden
    st = SentenceTransformer(model_name)
    print(f"Lade Embedding-Modell: {model_name}")

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": c.source, "page": c.page, "title": c.title, "chunk_id": c.chunk_id} for c in chunks]

    print(f"Berechne Embeddings für {len(chunks)} Chunks …")
    embeddings = st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).tolist()

    # In Chroma speichern
    collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

    print(f"Chroma-Index erfolgreich erstellt: {chroma_dir}")

# Main Entry Point
def main():
    print("Starte Ingestion-Prozess …")
    print(f"Eingabeordner: {DATA_DIR}")
    print(f"Ziel: {CHROMA_DIR}")
    print(f"Embedding-Modell: {EMBEDDING_MODEL}")

    if not DATA_DIR.exists():
        raise SystemExit(f"Eingabeordner {DATA_DIR} existiert nicht. Lege Dokumente in 'data/' ab.")

    chunks = load_pdfs(DATA_DIR)
    if not chunks:
        raise SystemExit("Keine Textinhalte gefunden. Lege PDF-Dateien in den Ordner 'data/'.")

    build_chroma(chunks, EMBEDDING_MODEL, CHROMA_DIR)
    print("Ingestion abgeschlossen!")

if __name__ == "__main__":
    main()
