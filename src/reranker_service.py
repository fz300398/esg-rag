import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder


class Candidate(BaseModel):
    text: str


class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]


class RerankResponse(BaseModel):
    scores: List[float]


app = FastAPI(title="Reranker Service")

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
ce: Optional[CrossEncoder] = None


@app.on_event("startup")
async def load():
    global ce
    ce = CrossEncoder(MODEL_NAME)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": MODEL_NAME, "loaded": ce is not None}


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    if ce is None:
        # Sollte nach Startup nicht passieren, aber gibt eine klare Fehlermeldung zurück
        return RerankResponse(scores=[])
    pairs = [(req.query, c.text) for c in req.candidates]
    scores = ce.predict(pairs).tolist()  # höhere Scores = relevanter
    return RerankResponse(scores=scores)
