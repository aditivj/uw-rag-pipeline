FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ libpoppler-cpp-dev poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app_hf.py .
COPY app/ ./app/

RUN mkdir -p logs data/docs

CMD ["python", "app_hf.py"]
