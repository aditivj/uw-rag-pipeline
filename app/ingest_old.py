from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber
import os

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "insurance_docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)

# def chunk_document(file_path: str, strategy: str = "semantic"):
#     loader = PyPDFLoader(file_path)
#     pages = loader.load()

#     if strategy == "fixed":
#         # Fixed-size: baseline, simpler but worse for long clauses
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500, chunk_overlap=50
#         )
#     else:
#         # Semantic: respects paragraph + sentence boundaries
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=150,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )

#     chunks = splitter.split_documents(pages)
#     return chunks

def extract_table_chunks(file_path: str):
    """
    Layout-aware extraction: pulls tables separately using pdfplumber
    and converts each row into its own semantically coherent chunk.
    This avoids tables being flattened into prose and diluted by
    fixed-size text splitting.
    """
    table_chunks = []
    table_pages = set()

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                # Only treat as a real table if header looks like column names
                if not header or all(c is None for c in header):
                    continue

                table_pages.add(page_num)

                for row in table[1:]:
                    if not row or all(c is None for c in row):
                        continue
                    # Build one coherent sentence per row:
                    # "Cyber Liability: Loss arising from electronic data..."
                    cells = [str(c).strip() for c in row if c]
                    if len(cells) >= 2:
                        row_text = f"{cells[0]}: {' '.join(cells[1:])}"
                    else:
                        row_text = " ".join(cells)

                    if len(row_text.strip()) > 5:
                        table_chunks.append({
                            "text": row_text,
                            "page": page_num,
                            "type": "table_row"
                        })

    return table_chunks, table_pages




# def ingest_pdf(file_path: str, strategy: str = "semantic"):
#     chunks = chunk_document(file_path, strategy)
#     collection = get_collection()

#     texts = [c.page_content for c in chunks]
#     embeddings = embedder.encode(texts).tolist()
#     ids = [f"{os.path.basename(file_path)}_chunk_{i}" 
#         for i in range(len(texts))]
#     metadatas = [{"source": file_path, 
#                 "page": c.metadata.get("page", 0),
#                 "strategy": strategy} 
#                 for c in chunks]

#     collection.upsert(
#         documents=texts,
#         embeddings=embeddings,
#         ids=ids,
#         metadatas=metadatas
#     )

#     return {"chunks_ingested": len(chunks), "strategy": strategy}

def chunk_prose(file_path: str, skip_pages: set, strategy: str = "semantic"):
    """
    Standard prose chunking, but skips pages that were already
    fully handled as tables to avoid duplicate/diluted content.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    chunks = splitter.split_documents(pages)
    return chunks


def ingest_pdf(file_path: str, strategy: str = "semantic"):
    # Step 1: extract tables as standalone, row-level chunks
    table_chunks, table_pages = extract_table_chunks(file_path)

    # Step 2: extract prose chunks as before
    prose_chunks = chunk_prose(file_path, table_pages, strategy)

    collection = get_collection()

    texts = []
    metadatas = []
    ids = []

    base_name = os.path.basename(file_path)

    # Add table-row chunks
    for i, tc in enumerate(table_chunks):
        texts.append(tc["text"])
        metadatas.append({
            "source": file_path,
            "page": tc["page"],
            "strategy": strategy,
            "chunk_type": "table_row"
        })
        ids.append(f"{base_name}_table_{i}")

    # Add prose chunks
    for i, c in enumerate(prose_chunks):
        texts.append(c.page_content)
        metadatas.append({
            "source": file_path,
            "page": c.metadata.get("page", 0),
            "strategy": strategy,
            "chunk_type": "prose"
        })
        ids.append(f"{base_name}_prose_{i}")

    if not texts:
        return {"chunks_ingested": 0, "strategy": strategy,
                "table_chunks": 0, "prose_chunks": 0}

    embeddings = embedder.encode(texts).tolist()

    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "chunks_ingested": len(texts),
        "strategy": strategy,
        "table_chunks": len(table_chunks),
        "prose_chunks": len(prose_chunks)
    }
