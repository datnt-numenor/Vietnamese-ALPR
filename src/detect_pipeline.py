"""Two-stage plate detection: COCO vehicle detector -> crop -> plate detector.

The plate model was trained on close-up motorbike photos, so it misses small plates
in street scenes. Cropping each vehicle first makes the plate large enough to find.

Usage:
    python src/detect_pipeline.py [image_or_dir ...] [--vehicle-conf 0.25] [--plate-conf 0.25]
"""
import argparse
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from detect_plate import IMAGE_EXTS, ROOT, collect_images

VEHICLE_WEIGHTS = ROOT / "models" / "yolov8s.pt"  # COCO pretrained, auto-downloaded on first run
PLATE_WEIGHTS = ROOT / "models" / "best.pt"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CROP_PAD = 0.05  # fraction of the vehicle box added on each side so edge plates are not cut
PLATE_IOU = 0.5  # duplicates across overlapping vehicle crops are merged above this IoU


def detect_vehicles(model, img, conf):
    result = model.predict(img, conf=conf, classes=list(VEHICLE_CLASSES), verbose=False)[0]
    return [(*map(int, b.xyxy[0].tolist()), VEHICLE_CLASSES[int(b.cls[0])], float(b.conf[0])) for b in result.boxes]


def detect_plates_in_vehicles(vehicle_model, plate_model, img, vehicle_conf=0.25, plate_conf=0.25):
    """Return (vehicles, plates). Each plate is (x1, y1, x2, y2, conf, vehicle_label) in full-image coords."""
    h, w = img.shape[:2]
    vehicles = detect_vehicles(vehicle_model, img, vehicle_conf)
    plates = []
    for x1, y1, x2, y2, label, _ in vehicles:
        px, py = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
        cx1, cy1, cx2, cy2 = max(x1 - px, 0), max(y1 - py, 0), min(x2 + px, w), min(y2 + py, h)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        for b in plate_model.predict(crop, conf=plate_conf, verbose=False)[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            plates.append((int(bx1) + cx1, int(by1) + cy1, int(bx2) + cx1, int(by2) + cy1, float(b.conf[0]), label))

    if plates:
        keep = nms(
            torch.tensor([p[:4] for p in plates], dtype=torch.float32),
            torch.tensor([p[4] for p in plates]),
            PLATE_IOU,
        )
        plates = [plates[i] for i in keep.tolist()]
    return vehicles, plates


def draw(img, vehicles, plates):
    for x1, y1, x2, y2, label, _ in vehicles:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
    for x1, y1, x2, y2, conf, _ in plates:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{conf:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[str(ROOT / "test_images")])
    ap.add_argument("--vehicle-conf", type=float, default=0.25)
    ap.add_argument("--plate-conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "pipeline"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vehicle_model, plate_model = YOLO(str(VEHICLE_WEIGHTS)), YOLO(str(PLATE_WEIGHTS))

    for f in collect_images(args.inputs):
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name}: cannot read")
            continue
        vehicles, plates = detect_plates_in_vehicles(vehicle_model, plate_model, img, args.vehicle_conf, args.plate_conf)
        draw(img, vehicles, plates)
        cv2.imwrite(str(out / f.name), img)
        print(f"{f.name}: {len(vehicles)} vehicle(s), {len(plates)} plate(s)", [round(p[4], 2) for p in plates])


if __name__ == "__main__":
    main()
