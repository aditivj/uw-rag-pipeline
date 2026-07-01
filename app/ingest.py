from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber
import re
import os

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "insurance_docs"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)


HEADING_PATTERN = re.compile(r"SECTION|EXCLUSION|COVERAGE|ENDORSEMENT|LIMIT", re.I)

def find_preceding_heading(table_top_y: float, words: list, max_lines_up: int = 6) -> str:
    lines = {}
    for w in words:
        if w["top"] < table_top_y:
            key = round(w["top"])
            lines.setdefault(key, []).append(w["text"])

    if not lines:
        return ""

    sorted_ys = sorted(lines.keys(), reverse=True)[:max_lines_up]

    for y in sorted_ys:
        line_text = " ".join(lines[y]).strip()
        if len(line_text) < 80 and HEADING_PATTERN.search(line_text):
            return line_text

    return ""


def extract_table_chunks(file_path: str):
    """
    Layout-aware extraction with section-context tagging AND column-label
    tagging. Each row's cells are explicitly labelled with their column
    header (e.g. 'Limit: 85,00,000, Deductible: 10,000') so multi-column
    tables don't lose meaning when flattened to text.
    """
    table_chunks = []
    table_pages = set()

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()
            tables = page.find_tables()

            for table in tables:
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue
                header = rows[0]
                if not header or all(c is None for c in header):
                    continue

                table_pages.add(page_num)

                table_top_y = table.bbox[1]
                heading = find_preceding_heading(table_top_y, words)

                clean_header = [str(h).strip() if h else f"Col{i}"
                                 for i, h in enumerate(header)]

                for row in rows[1:]:
                    if not row or all(c is None for c in row):
                        continue

                    cells = [str(c).strip() if c else "" for c in row]

                    # First column is usually the row label/name
                    row_name = cells[0] if cells else ""

                    # Remaining columns get explicit "ColumnName: value" labels
                    labeled_parts = []
                    for col_name, val in zip(clean_header[1:], cells[1:]):
                        if val:
                            labeled_parts.append(f"{col_name}: {val}")

                    if not row_name and not labeled_parts:
                        continue

                    row_text = f"{row_name} — " + ", ".join(labeled_parts) if labeled_parts else row_name

                    if len(row_text.strip()) <= 5:
                        continue

                    if heading:
                        full_text = f"[{heading}] {row_text}"
                    else:
                        full_text = row_text

                    table_chunks.append({
                        "text": full_text,
                        "page": page_num,
                        "type": "table_row",
                        "heading": heading
                    })

    return table_chunks, table_pages


def chunk_prose(file_path: str, skip_pages: set, strategy: str = "semantic"):
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
    table_chunks, table_pages = extract_table_chunks(file_path)
    prose_chunks = chunk_prose(file_path, table_pages, strategy)

    collection = get_collection()

    texts = []
    metadatas = []
    ids = []

    base_name = os.path.basename(file_path)

    for i, tc in enumerate(table_chunks):
        texts.append(tc["text"])
        metadatas.append({
            "source": file_path,
            "page": tc["page"],
            "strategy": strategy,
            "chunk_type": "table_row",
            "heading": tc["heading"]
        })
        ids.append(f"{base_name}_table_{i}")

    for i, c in enumerate(prose_chunks):
        texts.append(c.page_content)
        metadatas.append({
            "source": file_path,
            "page": c.metadata.get("page", 0),
            "strategy": strategy,
            "chunk_type": "prose",
            "heading": ""
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
