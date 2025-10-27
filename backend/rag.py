import os, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import chromadb
import numpy as np
import requests
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer, CrossEncoder
from deep_translator import GoogleTranslator

# LOGGING INITIALISIEREN
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SMALLTALK ERKENNUNG
SMALLTALK_PATTERNS = [
    # Begrüßungen
    "hi", "hallo", "hey", "servus", "moin", "guten tag",
    "guten morgen", "guten abend", "grüß dich", "na", "yo",

    # Höflichkeiten & Verabschiedungen
    "danke", "vielen dank", "dankeschön", "ok danke",
    "tschüss", "ciao", "bis bald", "mach’s gut", "auf wiedersehen",

    # Smalltalk & Befindlichkeiten
    "wie geht", "wie läuft", "alles gut", "was geht",
    "wie war dein tag", "was machst du", "mir geht", "dir geht",

    # Meta-/Selbstbezug
    "was kannst du", "wer bist du", "was bist du", "was machst du",
    "was ist dein name", "bist du echt", "bist du ein bot",
    "woher kommst du", "was kannst du tun", "erzähl was über dich"
]

def is_smalltalk(text: str) -> bool:
    """Einfache Erkennung von Smalltalk-Eingaben."""
    t = text.lower().strip()
    if not t or len(t.split()) < 2:
        return True
    if any(p in t for p in SMALLTALK_PATTERNS) and not any(
            kw in t for kw in ["esg", "nachhalt", "umwelt", "bericht", "unternehmen",
                               "sozial", "governance", "klima", "ziel"]
    ):
        if len(t.split()) <= 8:
            return True
    return False

# ÜBERSETZUNG
_translation_cache = {}

def translate_to_english(text: str) -> str:
    """Übersetzt deutschsprachige Eingaben ins Englische für präzisere Embeddings."""
    if text in _translation_cache:
        return _translation_cache[text]
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        if translated != text:
            logger.info(f"Frage automatisch übersetzt: '{text}' → '{translated}'")
        _translation_cache[text] = translated
        return translated
    except Exception as e:
        logger.warning(f"Übersetzung fehlgeschlagen: {e}")
        return text

# KONFIGURATION
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/app/chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_STREAM = os.getenv("OLLAMA_STREAM", "false").lower() == "true"

# MODELLE LADEN
logger.info("Lade Embedding-Modell ...")
st_model = SentenceTransformer(EMBEDDING_MODEL)

logger.info("Lade Reranker-Modell ...")
reranker = CrossEncoder(RERANKER_MODEL)

# CHROMA CLIENT INITIALISIEREN
def load_chroma(chroma_dir: Path):
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=ChromaSettings())
    return client.get_or_create_collection(name="esg_docs", metadata={"hnsw:space": "cosine"})

collection = load_chroma(CHROMA_DIR)

# DATA CLASS FÜR RETRIEVAL
@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    chunk_id: str

# KERNFUNKTIONEN
def embed_query(query: str) -> np.ndarray:
    """Wandelt die Nutzerfrage in einen semantischen Vektor um."""
    return st_model.encode([query], normalize_embeddings=True)[0].astype("float32")

def retrieve(query: str, k: int = 6, min_results: int = 2) -> List[Retrieved]:
    """Ruft relevante Dokumentpassagen aus ChromaDB ab."""
    q_emb = embed_query(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs or len(docs) < min_results:
        logger.warning("Zu wenige oder keine relevanten Dokumente gefunden – Fallback aktiv.")
        return []

    return [
        Retrieved(
            text=doc,
            source=m.get("source", "?"),
            page=int(m.get("page", 0)),
            chunk_id=m.get("chunk_id", "?")
        )
        for doc, m in zip(docs, metas)
    ]

def query_ollama(prompt: str) -> str:
    """Sendet den Prompt an Ollama und gibt die Antwort zurück."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": OLLAMA_STREAM},
            timeout=600,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        logger.info("Antwort von Ollama erfolgreich empfangen.")
        return response
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        return "Beim Abrufen der Antwort vom LLM ist ein Fehler aufgetreten."

# ANTWORTGENERIERUNG
def answer(question: str, top_k: int = 6, fallback_threshold: int = 2, history: Optional[str] = None) -> Tuple[
    str, List[Retrieved], float]:
    logger.info(f"Neue Frage: {question}")

    # Automatische Übersetzung (Deutsch → Englisch für Embeddings)
    translated_question = translate_to_english(question)

    # Smalltalk Detection
    if is_smalltalk(question):
        logger.info("Smalltalk erkannt – keine Dokumentensuche.")
        prompt = (
            "Du bist ein freundlicher, natürlicher ESG-Chatbot. "
            "Wenn der Nutzer Smalltalk macht, antworte locker, aber respektvoll – "
            "ohne Fachbegriffe oder Quellenangaben. "
            f"Hier ist die Nachricht des Nutzers:\n\n{question}\n\nAntwort:"
        )
        response = query_ollama(prompt)
        return response, [], 1.0  # Keine Quellen / volle Confidence

    # Dokumente abrufen
    ctx = retrieve(translated_question, k=top_k * 2, min_results=fallback_threshold)

    # Fallback, falls kein Kontext gefunden
    if not ctx:
        logger.info("Kein Kontext gefunden. Fallback auf reines LLM aktiviert.")
        fallback_prompt = (
            "Du bist ein ESG-Assistent. "
            "Wenn du keine Quellen findest, gib dein bestes Wissen an "
            "und bitte den Nutzer um Präzisierung, falls nötig.\n\n"
        )
        if history:
            fallback_prompt += "Vorheriger Verlauf:\n" + history + "\n\n"
        fallback_prompt += f"Frage: {question}\n\nAntwort:"
        response = query_ollama(fallback_prompt)
        return response, [], 0.0

    # Relevanzbewertung & Reranking
    pairs = [(translated_question, c.text) for c in ctx]
    scores = reranker.predict(pairs).tolist()
    order = sorted(range(len(ctx)), key=lambda i: scores[i], reverse=True)
    ctx = [ctx[i] for i in order[:top_k]]

    # Confidence berechnen
    if scores:
        confidence = float(np.clip(np.mean(scores[:top_k]), 0.0, 1.0))
    else:
        confidence = 0.0

    # Doppelte Quellen entfernen
    unique_sources = []
    for src in ctx:
        entry = (src.source, src.page)
        if entry not in unique_sources:
            unique_sources.append(entry)

    # Quellenliste formatieren
    source_list = [
        f"- {c.source} (Seite {c.page})"
        for c in ctx if c.source != "?"
    ]

    # Prompt zusammenbauen
    prompt_parts = [
        "Du bist ein ESG-Assistent. Antworte immer auf Deutsch, auch wenn die ursprüngliche Frage auf Englisch gestellt wurde.",
        "Nutze die folgenden Dokumentenausschnitte, um eine sachliche und vollständige Antwort zu geben.",
        "Wenn du Informationen nutzt, gib bitte am Ende deiner Antwort an, aus welchen Dokumenten (Quelle + Seite) sie stammen.",
        ""
    ]

    if history:
        prompt_parts.append("Vorheriger Verlauf:\n" + history)
        prompt_parts.append("")

    prompt_parts.append(f"Frage: {question}\n")
    prompt_parts.append("Relevante ESG-Kontexte:")
    for i, c in enumerate(ctx, 1):
        prompt_parts.append(f"\n--- [Dokument {i}] Quelle: {c.source}, Seite {c.page} ---\n{c.text}")
    prompt_parts.append("\nAntwort:")

    prompt = "\n".join(prompt_parts)

    # Anfrage an LLM
    response = query_ollama(prompt)

    return response, ctx, confidence