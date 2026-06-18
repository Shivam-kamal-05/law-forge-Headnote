# ─────────────────────────────────────────────────────────────
# Law Lens — Headnote Forge · production image
# Build context is the REPO ROOT (so backend/ is visible).
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System binaries for the OCR fallback:
#   poppler-utils  → pdf2image (rasterise scanned PDFs)
#   tesseract-ocr  → pytesseract (read the rasterised pages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app/backend

# Install deps first for better layer caching.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (src/ + static/).
COPY backend/ .

EXPOSE 8000

# Render injects $PORT; fall back to 8000 for local `docker run`.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
