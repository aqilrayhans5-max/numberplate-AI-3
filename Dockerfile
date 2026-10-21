FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch is much smaller than the default build.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-rcnn.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download the trained plate model from the GitHub Release.
ARG MODEL_URL=https://huggingface.co/AQILRAYHAN/plate-rcnn/resolve/main/plate_rcnn.pt
RUN curl -fL "$MODEL_URL" -o /app/plate_rcnn.pt

COPY . .

ENV PORT=10000
ENV PLATE_RCNN_PATH=/app/plate_rcnn.pt
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
