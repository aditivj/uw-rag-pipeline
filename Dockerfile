FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed by pdfplumber and sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so container startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code
COPY app/ ./app/
COPY evals/ ./evals/

# Create runtime directories
RUN mkdir -p logs chroma_db data/docs

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
