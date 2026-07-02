import gradio as gr
import os
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import tempfile

# ── Models ────────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("insurance_docs")

# ── Cache ─────────────────────────────────────────────────────────────────────
cache = []
CACHE_THRESHOLD = 0.92

HEADING_PATTERN = re.compile(r"SECTION|EXCLUSION|COVERAGE|ENDORSEMENT|LIMIT", re.I)

def find_heading(table_top_y, words, max_lines=6):
    lines = {}
    for w in words:
        if w["top"] < table_top_y:
            lines.setdefault(round(w["top"]), []).append(w["text"])
    for y in sorted(lines.keys(), reverse=True)[:max_lines]:
        text = " ".join(lines[y]).strip()
        if len(text) < 80 and HEADING_PATTERN.search(text):
            return text
    return ""

def ingest_pdf(file):
    if file is None:
        return "No file uploaded."
    try:
        path = file.name
        table_chunks = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words()
                for table in page.find_tables():
                    rows = table.extract()
                    if not rows or len(rows) < 2:
                        continue
                    header = [str(h).strip() if h else f"Col{i}" for i, h in enumerate(rows[0])]
                    heading = find_heading(table.bbox[1], words)
                    for row in rows[1:]:
                        if not row or all(c is None for c in row):
                            continue
                        cells = [str(c).strip() if c else "" for c in row]
                        row_name = cells[0] if cells else ""
                        labeled = [f"{header[i+1]}: {v}" for i, v in enumerate(cells[1:]) if v]
                        row_text = f"{row_name} — " + ", ".join(labeled) if labeled else row_name
                        if len(row_text.strip()) > 5:
                            full = f"[{heading}] {row_text}" if heading else row_text
                            table_chunks.append(full)

        loader = PyPDFLoader(path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        prose_chunks = [c.page_content for c in splitter.split_documents(pages)]

        all_chunks = table_chunks + prose_chunks
        embeddings = embedder.encode(all_chunks).tolist()
        import os as _os
        base = _os.path.basename(path)
        ids = [f"{base}_chunk_{i}" for i in range(len(all_chunks))]
        collection.upsert(documents=all_chunks, embeddings=embeddings, ids=ids)

        return f"✅ Ingested {len(all_chunks)} chunks ({len(table_chunks)} table rows, {len(prose_chunks)} prose)"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ask(question, history):
    if not question.strip():
        return history, ""
    try:
        q_emb = embedder.encode(question).tolist()

        # Check cache
        for entry in cache:
            sim = sum(a*b for a,b in zip(q_emb, entry["emb"])) / (
                sum(a**2 for a in q_emb)**0.5 * sum(b**2 for b in entry["emb"])**0.5)
            if sim >= CACHE_THRESHOLD:
                history.append((question, entry["answer"] + "\n\n⚡ Cache hit"))
                return history, ""

        results = collection.query(query_embeddings=[q_emb], n_results=min(10, collection.count() or 1))
        chunks = results["documents"][0] if results["documents"] else []
        if not chunks:
            history.append((question, "No documents ingested yet. Please upload a PDF first."))
            return history, ""

        scores = reranker.predict([[question, c] for c in chunks])
        top3 = [c for _, c in sorted(zip(scores, chunks), reverse=True)[:3]]

        context = "\n\n".join(top3)
        prompt = f"""You are an insurance document assistant.
Answer the question using only the context provided.

Rules:
- If context is tagged [SECTION III - EXCLUSIONS], the item is NOT covered.
- If context shows a coverage table, state the limit directly.
- Give a direct confident answer. Only say "I could not find this" if topic is truly absent.

Context:
{context}

Question: {question}

Answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        answer = response.choices[0].message.content
        cache.append({"emb": q_emb, "answer": answer})
        history.append((question, answer))
        return history, ""
    except Exception as e:
        history.append((question, f"❌ Error: {str(e)}"))
        return history, ""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Insurance Document Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Insurance Document Q&A")
    gr.Markdown("Upload insurance PDFs and ask questions in plain English.")

    with gr.Tabs():
        with gr.Tab("💬 Ask Questions"):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                q_input = gr.Textbox(placeholder="e.g. Is flood damage covered?", scale=5)
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    "Is flood damage covered under the homeowners policy?",
                    "What is the deductible for Coverage C?",
                    "Is cyber liability covered under the commercial policy?",
                    "What was the cause of loss in the claims report?",
                    "What is the net settlement amount recommended?",
                ],
                inputs=q_input
            )
            ask_btn.click(ask, [q_input, chatbot], [chatbot, q_input])
            q_input.submit(ask, [q_input, chatbot], [chatbot, q_input])

        with gr.Tab("📁 Upload Documents"):
            gr.Markdown("Upload insurance policy or claims PDFs.")
            file_input = gr.File(file_types=[".pdf"])
            upload_btn = gr.Button("Ingest Document", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(ingest_pdf, file_input, status)

    gr.Markdown("Built with FastAPI · ChromaDB · sentence-transformers · Groq LLaMA 3.1 8B · pdfplumber")

if __name__ == "__main__":
    demo.launch()
