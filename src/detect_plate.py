"""Detect license plates with the fine-tuned YOLOv8 model.

Usage:
    python src/detect_plate.py [image_or_dir ...] [--conf 0.25] [--out results/predictions]
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(weights=WEIGHTS):
    return YOLO(str(weights))


def detect_plates(model, image, conf=0.25):
    """Return a list of (x1, y1, x2, y2, confidence) for each plate in a BGR image."""
    result = model.predict(image, conf=conf, verbose=False)[0]
    return [(*map(int, b.xyxy[0].tolist()), float(b.conf[0])) for b in result.boxes]


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[str(ROOT / "test_images")])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "predictions"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()

    for f in collect_images(args.inputs):
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name}: cannot read")
            continue
        plates = detect_plates(model, img, args.conf)
        for x1, y1, x2, y2, c in plates:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{c:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(out / f.name), img)
        print(f"{f.name}: {len(plates)} plate(s)", [round(p[4], 2) for p in plates])


if __name__ == "__main__":
    main()
