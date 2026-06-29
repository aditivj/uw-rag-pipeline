import time
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil, os

from app.ingest import ingest_pdf
from app.retriever import retrieve
from app.llm import generate_answer
from app.logger import logger

app = FastAPI(title="Insurance RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), 
                 strategy: str = "semantic"):
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_pdf(tmp_path, strategy)
    os.remove(tmp_path)
    return result

@app.post("/query")
async def query(req: QueryRequest):
    start = time.time()

    # Retrieve + rerank
    chunks, cache_hit = retrieve(req.query, req.top_k)

    # Generate answer
    answer, input_tokens, output_tokens = generate_answer(
        req.query, chunks, cache_hit
    )

    latency_ms = (time.time() - start) * 1000

    # Log everything
    log = logger.log(
        query=req.query,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=cache_hit
    )

    return {
        "answer": answer,
        "latency_ms": log["latency_ms"],
        "cost_usd": log["estimated_cost_usd"],
        "cache_hit": cache_hit,
        "chunks_used": len(chunks)
    }

@app.get("/metrics")
def metrics():
    return logger.summary()