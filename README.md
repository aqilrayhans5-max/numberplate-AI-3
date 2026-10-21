# PlateVision AI

FastAPI-powered number plate recognition with image, video and browser camera capture.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## Deployment

Render: Docker runtime, Dockerfile in repository root. Tesseract is installed by the Dockerfile. SQLite storage on free instances is ephemeral; configure durable Postgres/storage for production. Supply PLATE_RCNN_PATH with fine-tuned Faster R-CNN weights to enable the R-CNN detector; OpenCV fallback is less accurate. The application cannot determine vehicle ownership or make/model merely from license plate text.

## Faster R-CNN plate detector

Plate detection uses torchvision's Faster R-CNN (ResNet-50 FPN). It is off by default and the app falls back to OpenCV when no weights are set.

1. Prepare a YOLO-format dataset: `images/*.jpg` and `labels/*.txt` with lines `<class> <cx> <cy> <w> <h>` (normalised).
2. Train (needs torch; a GPU is much faster):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
   pip install -r requirements-rcnn.txt
   python tools/train_rcnn.py --data dataset --out plate_rcnn.pt --epochs 20
   ```
3. Run with the weights:
   ```bash
   PLATE_RCNN_PATH=plate_rcnn.pt PLATE_RCNN_CONF=0.35 python -m uvicorn app.main:app
   ```

`/api/health` reports `"model": "faster-rcnn"` when the weights load. Optional `PLATERECOGNIZER_API_TOKEN` still takes precedence over the local detector.
