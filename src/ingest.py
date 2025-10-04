import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

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

VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss").lower()

# FAISS nur importieren, wenn wir ihn brauchen (vermeidet Fehler lokal)
if VECTOR_STORE == "faiss":
    import faiss  # type: ignore

# Chroma nur importieren, wenn wir ihn brauchen
if VECTOR_STORE == "chroma":
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore

CHUNK_SIZE = 1000  # Zeichen
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    title: str | None = None


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


def load_pdfs(folder: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for pdf_path in sorted(folder.glob("**/*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            title = None
            md = getattr(reader, "metadata", None)
            if md and getattr(md, "title", None):
                title = md.title
            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if not text.strip():
                    continue  # <<< neu: leere Seiten überspringen
                for i, c in enumerate(chunk_text(text)):
                    chunks.append(
                        Chunk(
                            text=c,
                            source=str(pdf_path.name),
                            page=page_idx,
                            chunk_id=f"{pdf_path.stem}-p{page_idx}-c{i}",
                            title=title,
                        )
                    )
        except Exception as e:
            print(f"Warnung: Konnte {pdf_path} nicht lesen: {e}")
    return chunks


def build_faiss(chunks: List[Chunk], model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    embeddings = st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine auf normalisierten Vektoren
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    meta: List[Dict] = [asdict(c) for c in chunks]
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FAISS-Index erstellt: {out_dir}")


def build_chroma(chunks: List[Chunk], model_name: str, chroma_dir: Path):
    chroma_dir.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings(allow_reset=True))
    client.reset()  # löscht alle Collections im Pfad
    coll = client.get_or_create_collection(name="esg", metadata={"hnsw:space": "cosine"})

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": c.source, "page": c.page, "title": c.title, "chunk_id": c.chunk_id} for c in chunks]

    # Embeddings selbst berechnen (robust)
    embs = st.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).tolist()

    # Bestehende Collection zurücksetzen? – hier aus Sicherheitsgründen ja:
    coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    print(f"Chroma-Index erstellt in: {chroma_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data"))
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--embedding_model", type=str, default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    parser.add_argument("--chroma_dir", type=Path, default=Path("chroma"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Eingabeordner {args.input} existiert nicht. Lege PDFs in 'data/' ab.")
    chunks = load_pdfs(args.input)
    if not chunks:
        raise SystemExit("Keine Texte gefunden. Lege PDF-Dateien in den Ordner 'data/'.")

    if VECTOR_STORE == "faiss":
        build_faiss(chunks, args.embedding_model, args.index)
    elif VECTOR_STORE == "chroma":
        build_chroma(chunks, args.embedding_model, args.chroma_dir)
    else:
        raise SystemExit(f"Unbekannter VECTOR_STORE: {VECTOR_STORE} (erwarte 'faiss' oder 'chroma')")


if __name__ == "__main__":
    main()