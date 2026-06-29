from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "insurance_docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

cache = []
CACHE_THRESHOLD = 0.92

def get_cached(query_embedding):
    for entry in cache:
        sim = float(
            sum(a * b for a, b in zip(query_embedding, entry["embedding"])) /
            (sum(a**2 for a in query_embedding) ** 0.5 *
             sum(b**2 for b in entry["embedding"]) ** 0.5)
        )
        if sim >= CACHE_THRESHOLD:
            return entry["chunks"], True
    return None, False

def retrieve(query: str, top_k: int = 10):
    query_embedding = embedder.encode(query).tolist()

    cached_chunks, cache_hit = get_cached(query_embedding)
    if cache_hit:
        return cached_chunks, True

    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1)
    )

    chunks = results["documents"][0] if results["documents"] else []

    if not chunks:
        return [], False

    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), reverse=True)
    top_chunks = [chunk for _, chunk in ranked[:3]]

    cache.append({"embedding": query_embedding, "chunks": top_chunks})

    return top_chunks, False
