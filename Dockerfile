FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers doesn't pull the much
# larger CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY db ./db
COPY data ./data
COPY scripts ./scripts
COPY eval ./eval

RUN useradd --create-home appuser && mkdir -p /models && chown appuser /models
USER appuser

EXPOSE 8000

# Build the database on first start (schema + seed + ingest), then serve.
CMD ["sh", "-c", "python -m scripts.setup_db --if-empty && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
