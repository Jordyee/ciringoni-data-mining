"""Evaluate a native-source checkpoint against the fixed M1 val coordinate system."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from pycocotools.coco import COCO
import torch
from ultralytics import YOLO

from evaluasi_mosaic import GT_PATH, THREADS_CPU, benchmark_cpu, m1_ap_small, precision_recall_fixed, save_metrics


def to_m1_coordinates(boxes, source, reference):
    mapped = boxes.clone()
    for axis_xy in (0, 1):
        coordinates = mapped[:, axis_xy::2]
        coordinates.sub_(source[axis_xy]).mul_(reference[axis_xy + 2] / source[axis_xy + 2])
        coordinates.add_(reference[axis_xy]).clamp_(0, 640)
    return mapped


class M1CoordinatePredictor:
    def __init__(self, model, geometry, size):
        self.model = model
        self.geometry = geometry
        self.native_key = f"content_at_{size}"

    def predict(self, **options):
        for result in self.model.predict(**options):
            start = perf_counter()
            row = self.geometry[Path(result.path).stem]
            with torch.inference_mode():
                boxes = to_m1_coordinates(result.boxes.xyxy, row[self.native_key], row["content_at_640"])
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                result.boxes.data[:, :4] = boxes
                result.boxes = result.boxes[keep]
            result.speed["projection"] = (perf_counter() - start) * 1000
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_yaml", type=Path)
    parser.add_argument("geometry_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True, choices=(640, 1280))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--cpu-benchmark", action="store_true", help="Add CPU latency to completed validation")
    args = parser.parse_args()

    metrics_path = args.output / "metrik.json"
    if args.cpu_benchmark:
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        if row["imgsz"] != args.size or Path(row["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Checkpoint or input size differs from the evaluated one.")
        latency = benchmark_cpu(args.checkpoint, args.output / "coco_val_inputs.txt", row["gambar"], args.size)
        row.update(latency, cpu_torch_threads=THREADS_CPU)
        save_metrics(metrics_path, row)
        print(json.dumps(latency), flush=True)
        return

    gt = COCO(str(GT_PATH))
    ids = sorted(gt.imgs)
    if ids != list(range(1, 357)) or len(gt.anns) != 2205:
        raise ValueError("Ground truth differs from the M1 validation reference.")
    val_dir = args.dataset_yaml.resolve().parent / "images/val"
    images = [val_dir / gt.imgs[image_id]["file_name"] for image_id in ids]
    expected = {image.resolve() for image in images}
    if expected != {path.resolve() for path in val_dir.rglob("*.jpg")}:
        raise ValueError("Native dataset must contain all 356 M1 validation inputs with the same IDs.")
    manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and not manifest.get("ready"):
        raise ValueError("Native dataset geometry manifest has not been marked ready.")
    rows = manifest["files"] if isinstance(manifest, dict) else manifest
    rows = [row for row in rows if row["split"] == "val"]
    geometry = {Path(row["filename"]).stem: row for row in rows}
    if len(rows) != len(images) or set(geometry) != {image.stem for image in images}:
        raise ValueError("Geometry manifest must cover each M1 validation ID exactly once.")
    for row in geometry.values():
        for size in (640, 1280):
            left, top, width, height = row[f"content_at_{size}"]
            if not (0 <= left < left + width <= size and 0 <= top < top + height <= size):
                raise ValueError(f"Invalid content rectangle for {row['filename']} at size {size}.")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.checkpoint))
    metrics = model.val(
        data=str(args.dataset_yaml.resolve()), split="val", imgsz=args.size, device=args.device, batch=args.batch,
        project=str(args.output.resolve()), name="ultralytics", max_det=100, workers=0,
        plots=False, exist_ok=True,
    )
    predictor = M1CoordinatePredictor(model, geometry, args.size)
    ap_small = m1_ap_small()(predictor, args.size, args.device, GT_PATH, images, args.output)
    predictions = json.loads((args.output / "coco_val_predictions.json").read_text(encoding="utf-8"))
    precision, recall = precision_recall_fixed(gt, predictions)
    row = {
        "id": args.output.name, "split": "val", "gambar": len(images), "imgsz": args.size,
        "val_ultralytics_batch": args.batch, "checkpoint": str(args.checkpoint),
        "presisi": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map), "AP-small": ap_small,
        "presisi_conf_0.05_iou_0.5": precision, "recall_conf_0.05_iou_0.5": recall,
    }
    save_metrics(metrics_path, row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
