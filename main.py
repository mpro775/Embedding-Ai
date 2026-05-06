from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

app = FastAPI(title="Embedding API", version="1.0.0")

model = SentenceTransformer(MODEL_NAME)


class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = req.texts if isinstance(req.texts, list) else [req.texts]

    vectors = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    return {
        "model": MODEL_NAME,
        "dimension": len(vectors[0]) if vectors else 0,
        "vectors": vectors
    }