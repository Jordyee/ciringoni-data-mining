"""Compare full-frame inference and SAHI on the native E2 validation set."""

import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.slicing import get_slice_bboxes
import torch

from evaluasi_e2 import to_m1_coordinates
from evaluasi_mosaic import CONF_FIXED, GT_PATH, precision_recall_fixed
from uji_tiling_640 import draw_boxes, montage, score

TILE = 640
OVERLAP = 0.25
MERGE_IOU = 0.50
MIN_CONFIDENCE = 0.001
MAX_DETECTIONS = 100
CPU_THREADS = 4
SLICE_BATCH = 4

def predict(model, rgb, sliced):
    model.image_size = TILE if sliced else rgb.shape[0]
    if sliced:
        return get_sliced_prediction(
            rgb, model, slice_height=TILE, slice_width=TILE,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            perform_standard_pred=False, postprocess_type="NMS",
            postprocess_match_metric="IOU", postprocess_match_threshold=MERGE_IOU,
            postprocess_class_agnostic=False, verbose=0, batch_size=SLICE_BATCH,
        )
    return get_prediction(rgb, model)


def rows_for_coco(result, image_id, geometry):
    objects = sorted(result.object_prediction_list, key=lambda obj: obj.score.value, reverse=True)
    if not objects:
        return []
    boxes = torch.tensor([obj.bbox.to_xyxy() for obj in objects], dtype=torch.float32)
    boxes = to_m1_coordinates(boxes, geometry["content_at_1280"], geometry["content_at_640"])
    return [
        {"image_id": image_id, "category_id": int(obj.category.id) + 1,
         "bbox": [left, top, right - left, bottom - top], "score": float(obj.score.value)}
        for obj, (left, top, right, bottom) in zip(objects, boxes.tolist(), strict=True)
        if right > left and bottom > top
    ][:MAX_DETECTIONS]


def read_rgb(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape != (1280, 1280, 3):
        raise ValueError(f"Expected a 1280 × 1280 native export: {path.name}")
    return rgb


def write_evidence(ground_truth, arms, reference, output):
    small_by_arm = {}
    predicted = {}
    for arm, (rows, _) in arms.items():
        by_image = defaultdict(list)
        for row in rows:
            if row["score"] >= CONF_FIXED:
                by_image[row["image_id"]].append(row)
        predicted[arm] = by_image
        detections = [row for rows in by_image.values() for row in rows]
        if not detections:
            small_by_arm[arm] = {}
            continue
        matches = COCOeval(ground_truth, ground_truth.loadRes(detections), "bbox")
        matches.params.iouThrs = np.array([0.5])
        matches.params.areaRng = matches.params.areaRng[:1]
        matches.params.areaRngLbl = ["all"]
        matches.evaluate()
        small_by_arm[arm] = {
            item["image_id"]: {
                int(ann) for ann, ignored in zip(item["dtMatches"][0], item["dtIgnore"][0], strict=True)
                if ann and not ignored and ground_truth.anns[int(ann)]["area"] < 1024
            }
            for item in matches.evalImgs if item is not None
        }

    def differences(image_id):
        full = small_by_arm["full"].get(image_id, set())
        sliced = small_by_arm["sahi"].get(image_id, set())
        clutter = sum(len(by_image[image_id]) for by_image in predicted.values())
        return (len(full ^ sliced)/(clutter+1), len(full ^ sliced), -clutter, -image_id)

    image_id = max(sorted(ground_truth.imgs), key=differences)
    path = reference / ground_truth.imgs[image_id]["file_name"]
    with Image.open(path) as image:
        ref = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    if ref.shape != (640, 640, 3):
        raise ValueError(f"Expected the 640 reference frame for validation image {image_id}")
    gt_rows = ground_truth.imgToAnns[image_id]
    panels = []
    labels = []
    for label, rows, color in (
        ("Anotasi acuan", gt_rows, (40, 230, 65)),
        ("Penuh 1280", predicted["full"][image_id], (20, 120, 255)),
        ("SAHI tile 640", predicted["sahi"][image_id], (20, 120, 255)),
    ):
        panels.append(draw_boxes(ref.copy(), [(x, y, x+w, y+h) for x, y, w, h in (row["bbox"] for row in rows)], color))
        labels.append(f"{label} | {len(rows)} kotak")
    montage(output / "bukti_e4_sahi.png", f"E4 | Val {image_id} | Proyeksi 640 | Confidence 0,05", panels, labels)
    return {
        "image_id": image_id,
        "full_matched_small": len(small_by_arm["full"].get(image_id, set())),
        "sahi_matched_small": len(small_by_arm["sahi"].get(image_id, set())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("val_directory", type=Path)
    parser.add_argument("geometry_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--smoke", action="store_true", help="Check two images without recording scores.")
    parser.add_argument("--evidence-reference", type=Path, help="Optional M1 validation images for the example figure.")
    args = parser.parse_args()
    if args.output.exists() and not args.smoke:
        raise FileExistsError(args.output)

    ground_truth = COCO(str(GT_PATH))
    image_ids = sorted(ground_truth.imgs)
    image_paths = {key: args.val_directory / ground_truth.imgs[key]["file_name"] for key in image_ids}
    manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    entries = [row for row in manifest["files"] if row["split"] == "val"]
    geometry = {Path(row["filename"]).stem: row for row in entries}
    if (not manifest["ready"] or len(entries) != 356 or len(image_ids) != 356 or len(ground_truth.anns) != 2205
            or set(geometry) != {path.stem for path in image_paths.values()}
            or not all(path.is_file() for path in image_paths.values())):
        raise ValueError("Expected the complete audited E2 validation set and M1 validation IDs.")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(args.checkpoint),
        confidence_threshold=MIN_CONFIDENCE, device=args.device,
    )
    benchmark = args.device == "cpu"
    for image_id in image_ids[:10 if benchmark and not args.smoke else 2]:
        path = image_paths[image_id]
        rgb = read_rgb(path)
        for sliced in (False, True):
            result = predict(model, rgb, sliced)
            if args.smoke:
                json.dumps(rows_for_coco(result, image_id, geometry[path.stem]))
    if args.smoke:
        print("SAHI full and sliced prediction passed on two E2 validation images.")
        return

    # Predictor initialization can reset the PyTorch CPU thread count.
    if benchmark:
        torch.set_num_threads(CPU_THREADS)
    arms = {"full": ([], []), "sahi": ([], [])}
    for count, image_id in enumerate(image_ids, 1):
        path = image_paths[image_id]
        rgb = read_rgb(path)
        order = ("full", "sahi") if count % 2 else ("sahi", "full")
        for arm in order:
            detections, latency = arms[arm]
            start = perf_counter()
            result = predict(model, rgb, arm == "sahi")
            detections.extend(rows_for_coco(result, image_id, geometry[path.stem]))
            if benchmark:
                latency.append((perf_counter() - start) * 1000)
        if count % 50 == 0 or count == len(image_ids):
            print(f"{count}/{len(image_ids)} validation images", flush=True)
    if benchmark and torch.get_num_threads() != CPU_THREADS:
        raise RuntimeError("The configured CPU thread count changed during timing.")

    args.output.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open("rb") as checkpoint:
        checkpoint_hash = file_digest(checkpoint, "sha256").hexdigest()
    slice_count = len(get_slice_bboxes(
        image_height=1280, image_width=1280, slice_height=TILE, slice_width=TILE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
    ))
    metrics = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_hash,
        "images": len(image_ids), "gt_boxes": len(ground_truth.anns), "sahi_version": sahi.__version__,
        "full_input": 1280, "tile": TILE, "overlap": OVERLAP, "slices_per_image": slice_count,
        "slice_batch": SLICE_BATCH, "min_confidence": MIN_CONFIDENCE, "merge_iou": MERGE_IOU,
        "max_detections_per_image": MAX_DETECTIONS, "device": args.device,
    }
    if benchmark:
        metrics.update(
            torch_cpu_threads=CPU_THREADS,
            timing_scope="Per image after JPEG decode, including preprocessing, inference, merging, and coordinate mapping",
        )
    for arm, (detections, latency) in arms.items():
        (args.output / f"prediksi_{arm}.json").write_text(json.dumps(detections), encoding="utf-8")
        precision, recall = precision_recall_fixed(ground_truth, detections)
        metrics[arm] = {
            **score(ground_truth, detections, image_ids),
            "precision_conf_0.05_iou_0.5": precision, "recall_conf_0.05_iou_0.5": recall,
        }
        if benchmark:
            metrics[arm].update(cpu_median_ms=float(np.median(latency)), cpu_p95_ms=float(np.percentile(latency, 95)))
    if args.evidence_reference:
        metrics["evidence"] = write_evidence(ground_truth, arms, args.evidence_reference, args.output)
    (args.output / "metrik_e4_sahi.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
