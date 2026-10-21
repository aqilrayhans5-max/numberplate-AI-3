"""Fine-tune a Faster R-CNN licence-plate detector.

Expected dataset layout (YOLO format, as exported by Roboflow and most plate datasets):

    dataset/
      images/  *.jpg | *.jpeg | *.png
      labels/  same basename .txt, one line per plate: "<class> <cx> <cy> <w> <h>"
               (coordinates normalised to 0..1; class id is ignored, every box is a plate)

Usage:
    pip install -r requirements-rcnn.txt
    python tools/train_rcnn.py --data dataset --out plate_rcnn.pt --epochs 20

Then run the app with:
    PLATE_RCNN_PATH=plate_rcnn.pt uvicorn app.main:app
"""
import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF

PLATE_LABEL = 1  # 0 is background in torchvision detection models


class PlateDataset(Dataset):
    def __init__(self, root, train=True):
        self.images = sorted(p for p in (Path(root) / "images").iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.labels_dir = Path(root) / "labels"
        self.train = train
        if not self.images:
            raise SystemExit(f"No images found in {root}/images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes = []
        label_file = self.labels_dir / (img_path.stem + ".txt")
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

        tensor = TF.to_tensor(image)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and random.random() < 0.5:
            tensor = tensor.flip(-1)
            if len(boxes_t):
                boxes_t[:, [0, 2]] = w - boxes_t[:, [2, 0]]
        target = {
            "boxes": boxes_t,
            "labels": torch.full((len(boxes_t),), PLATE_LABEL, dtype=torch.int64),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def build_model():
    # COCO-pretrained backbone and RPN; only the final predictor is replaced for 2 classes.
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="plate_rcnn.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PlateDataset(args.data, train=True)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / max(1, len(loader)):.4f}")

    # Save only the weights; app/recognition.py rebuilds the 2-class model and loads this.
    torch.save(model.cpu().state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
