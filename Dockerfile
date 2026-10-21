FROM python:3.11-slim

# Tesseract is the OCR engine that reads the plate text.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
