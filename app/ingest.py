from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import os

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "insurance_docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

def chunk_document(file_path: str, strategy: str = "semantic"):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if strategy == "fixed":
        # Fixed-size: baseline, simpler but worse for long clauses
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
    else:
        # Semantic: respects paragraph + sentence boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    chunks = splitter.split_documents(pages)
    return chunks

def ingest_pdf(file_path: str, strategy: str = "semantic"):
    chunks = chunk_document(file_path, strategy)
    collection = get_collection()

    texts = [c.page_content for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [f"{os.path.basename(file_path)}_chunk_{i}" 
        for i in range(len(texts))]
    metadatas = [{"source": file_path, 
                "page": c.metadata.get("page", 0),
                "strategy": strategy} 
                for c in chunks]

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return {"chunks_ingested": len(chunks), "strategy": strategy}