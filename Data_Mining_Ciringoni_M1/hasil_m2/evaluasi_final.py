"""Score full-image validation, or lock and evaluate the single held-out model."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
from itertools import islice
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import psutil
from pycocotools.coco import COCO
import torch
from ultralytics import YOLO

from bangun_test_native import sha256
from evaluasi_e2 import M1CoordinatePredictor
from evaluasi_mosaic import CONF_FIXED, GT_PATH, THREADS_CPU, precision_recall_fixed, save_metrics
from uji_tiling_640 import score


PROTOCOL = {
    "batch": 1, "conf_ap": 0.001, "conf_precision_recall": CONF_FIXED, "iou_match": 0.5,
    "iou_nms_if_used": 0.7, "max_det": 100, "torch_threads": THREADS_CPU, "warmup_images": 10,
    "inference": "one full square image, YOLO26 end-to-end, native-to-M1 projection",
    "latency": "CPU preprocessing, inference, postprocessing and coordinate projection, excluding JPEG read",
}
VERSIONS = ("torch", "torchvision", "ultralytics", "openvino", "nncf", "pycocotools", "numpy",
            "opencv-python", "psutil")


def environment():
    return {**{name: importlib.metadata.version(name) for name in VERSIONS}, "torch": str(torch.__version__)}


def model_files(model):
    if model.suffix == ".pt":
        files = [model]
    else:
        files = sorted(model.glob("*"))
        if sorted(path.suffix for path in files) != [".bin", ".xml", ".yaml"]:
            raise ValueError("Expected one OpenVINO XML, BIN and metadata YAML")
    return {path.name: sha256(path) for path in files}


def inputs(native, split, size):
    is_test = split == "test"
    gt_path = native / "coco_test_gt.json" if is_test else GT_PATH
    audit_path = native / ("audit_test_native.json" if is_test else "audit_pasangan_e2.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["ready"]:
        raise ValueError(f"Incomplete native {split} audit")
    gt = COCO(str(gt_path))
    ids = sorted(gt.imgs)
    count, boxes = (178, 1159) if is_test else (356, 2205)
    if ids != list(range(1, count + 1)) or len(gt.anns) != boxes or set(gt.cats) != {1}:
        raise ValueError(f"Unexpected M1 ground truth for {split}")
    folder = native / str(size) / "images" / split
    images = [folder / gt.imgs[image_id]["file_name"] for image_id in ids]
    rows = [row for row in audit["files"] if row["split"] == split and row["status"] == "paired"]
    geometry = {Path(row["filename"]).stem: row for row in rows}
    if (len(rows) != count or set(geometry) != {path.stem for path in images}
            or set(folder.glob("*.jpg")) != set(images)):
        raise ValueError(f"Image or geometry IDs differ from ground truth for {split}")
    for image_id, path in zip(ids, images, strict=True):
        row = geometry[path.stem]
        for resolution in {640, size}:
            x, y, w, h = row[f"content_at_{resolution}"]
            if not (0 <= x < x + w <= resolution and 0 <= y < y + h <= resolution):
                raise ValueError("Invalid native-to-M1 content rectangle")
        if is_test:
            label = native / str(size) / "labels/test" / (path.stem + ".txt")
            if (row["image_id"] != image_id or
                    row["sha256"][str(size)] != {"image": sha256(path), "label": sha256(label)}):
                raise ValueError("Native test image or label differs from audited file")
    ids_sha256 = hashlib.sha256(
        json.dumps([[i, gt.imgs[i]["file_name"]] for i in ids], ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    fingerprints = {
        "ids_sha256": ids_sha256, "ground_truth_sha256": sha256(gt_path), "audit_sha256": sha256(audit_path),
        "images": count, "boxes": boxes,
    }
    if is_test:
        if fingerprints["ground_truth_sha256"] != audit["ground_truth_sha256"]:
            raise ValueError("Test ground truth differs from audited file")
        fingerprints["archive_sha256"] = audit["archive_sha256"]
    return gt, images, geometry, fingerprints


def evaluate(model_path, size, dataset, output):
    gt, images, geometry, fingerprints = dataset
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    affinity = process.cpu_affinity()[:THREADS_CPU]
    if len(affinity) != THREADS_CPU:
        raise RuntimeError(f"Expected {THREADS_CPU} available logical CPUs")
    process.cpu_affinity(affinity)
    torch.set_num_threads(THREADS_CPU)
    model = YOLO(str(model_path), task="detect")
    predictor = M1CoordinatePredictor(model, geometry, size)
    manifest = output / "inputs.txt"
    manifest.write_text("".join(path.as_posix() + "\n" for path in images), encoding="utf-8")
    options = dict(source=str(manifest), imgsz=size, device="cpu", batch=PROTOCOL["batch"],
                   conf=PROTOCOL["conf_ap"], iou=PROTOCOL["iou_nms_if_used"], max_det=PROTOCOL["max_det"],
                   stream=True, verbose=False, nms=False)
    warm = predictor.predict(**options)
    list(islice(warm, PROTOCOL["warmup_images"]))
    warm.close()
    lookup = {path: image_id for image_id, path in zip(sorted(gt.imgs), images, strict=True)}
    seen, detections, times = set(), [], []
    for result in predictor.predict(**options):
        image_id = lookup[Path(result.path).resolve()]
        if image_id in seen:
            raise ValueError("Duplicate prediction image")
        seen.add(image_id)
        times.append({"image_id": image_id, "ms": sum(result.speed.values())})
        for x1, y1, x2, y2, conf, cls in result.boxes.data.tolist():
            if cls != 0 or x2 <= x1 or y2 <= y1 or not 0 <= conf <= 1:
                raise ValueError("Unexpected box in M1 coordinates")
            detections.append({"image_id": image_id, "category_id": 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf})
    if seen != set(gt.imgs):
        raise ValueError("Predictions do not cover the audited split")
    metrics = score(gt, detections, sorted(gt.imgs))
    precision, recall = precision_recall_fixed(gt, detections)
    backend = model.predictor.model.backend
    runtime = {"cpu_affinity": affinity, "end2end": model.predictor.model.end2end}
    if hasattr(backend, "ov_compiled_model"):
        runtime["openvino_threads"] = backend.ov_compiled_model.get_property("INFERENCE_NUM_THREADS")
        runtime["openvino_precision_hint"] = str(backend.ov_compiled_model.get_property("INFERENCE_PRECISION_HINT"))
    if not runtime["end2end"]:
        raise ValueError("YOLO26 exported model did not use the locked end-to-end head")
    row = {
        "size": size, "dataset": fingerprints, "protocol": PROTOCOL, "runtime": runtime,
        "versions": environment(), "model_sha256": model_files(model_path), **metrics,
        "precision": precision, "recall": recall,
        "cpu_median_ms": median(item["ms"] for item in times),
        "cpu_p95_ms": float(np.percentile([item["ms"] for item in times], 95)),
    }
    save_metrics(output / "predictions.json", detections)
    save_metrics(output / "latency.json", times)
    save_metrics(output / "metrics.json", row)
    print(json.dumps(row, indent=2), flush=True)


def lock_config(model, checkpoint, dataset, validation_path, size, precision):
    files = model_files(model)
    val = json.loads(validation_path.read_text(encoding="utf-8"))
    checkpoint_digest = sha256(checkpoint)
    if (model.suffix == ".pt" or val["model_sha256"] != files or val["size"] != size
            or val["protocol"] != PROTOCOL or val["dataset"]["images"] != 356
            or val["versions"] != environment()
            or val["export"] != {"precision": precision, "checkpoint_sha256": checkpoint_digest}):
        raise ValueError("OpenVINO files must match their 356-image validation result")
    return {
        "format": "OpenVINO", "precision": precision, "size": size, "protocol": PROTOCOL,
        "checkpoint_sha256": checkpoint_digest, "model_sha256": files, "test": dataset[-1],
        "validation_metrics_sha256": sha256(validation_path), "versions": environment(),
        "evaluator_files_sha256_normalized_lf": {
            file: hashlib.sha256(Path(__file__).with_name(file).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for file in ("evaluasi_final.py", "evaluasi_e2.py", "evaluasi_mosaic.py", "uji_tiling_640.py")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("val", "lock", "test"))
    parser.add_argument("model", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path, help="New score directory, or new JSON file in lock mode")
    parser.add_argument("--size", type=int, choices=(640, 1280), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-metrics", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    dataset = inputs(args.native.resolve(), "val" if args.mode == "val" else "test", args.size)
    if args.mode != "val":
        if not (args.precision and args.checkpoint and args.validation_metrics):
            parser.error("lock/test require --precision, --checkpoint and --validation-metrics")
        lock = lock_config(model, args.checkpoint, dataset, args.validation_metrics, args.size, args.precision)
    if args.mode == "lock":
        lock["locked_at_wib"] = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(lock, stream, indent=2)
            stream.write("\n")
        print("Lock SHA-256:", sha256(args.output), flush=True)
    else:
        if args.mode == "test":
            if not args.lock:
                parser.error("test requires the previously created --lock")
            saved = json.loads(args.lock.read_text(encoding="utf-8"))
            locked_at = saved.pop("locked_at_wib", None)
            if not locked_at or saved != lock:
                raise ValueError("Test configuration or file hashes differ from the pre-test lock")
            print(f"Using test lock {sha256(args.lock)} from {locked_at}", flush=True)
        evaluate(model, args.size, dataset, args.output.resolve())
