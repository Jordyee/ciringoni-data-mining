import argparse
import ast
import csv
from itertools import islice
import json
from pathlib import Path
import statistics

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from ultralytics import YOLO


OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parent
GT_PATH = PROJECT / "hasil_m1/coco_val_gt.json"
CHECKPOINTS = {
    "E0_640": "runs/E0_640/weights/best.pt",
    "E6_640_no_mosaic": "runs/E6_640_no_mosaic/weights/best.pt",
    "E5_640_p2": "runs/E5_640_p2/weights/best.pt",
    "E3_control_full_640": "eksperimen_e3_640_verified/runs/E3_control_full_640/weights/last.pt",
    "E3_tile384_640": "eksperimen_e3_640_verified/runs/E3_tile384_640/weights/last.pt",
}
CONF_FIXED = 0.05
THREADS_CPU = 4


def m1_ap_small():
    notebook = json.loads((PROJECT / "FP_Data_Mining_M1_Eksekusi_Lokal.ipynb").read_text(encoding="utf-8"))
    training = next(cell for cell in notebook["cells"] if cell.get("id") == "m1-11")
    function = next(
        node for node in ast.parse("".join(training["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name == "ap_small_for_model"
    )
    scope = {"Path": Path, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PROJECT / "FP_Data_Mining_M1_Eksekusi_Lokal.ipynb"), "exec"), scope)
    return scope[function.name]


def precision_recall_fixed(ground_truth, predictions):
    predictions = [prediction for prediction in predictions if prediction["score"] >= CONF_FIXED]
    if not predictions:
        return 0.0, 0.0
    scorer = COCOeval(ground_truth, ground_truth.loadRes(predictions), "bbox")
    scorer.params.iouThrs = np.array([0.5])
    scorer.params.areaRng = [scorer.params.areaRng[0]]
    scorer.params.areaRngLbl = ["all"]
    scorer.evaluate()
    matched = sum(
        int(np.count_nonzero((item["dtMatches"][0] != 0) & ~item["dtIgnore"][0]))
        for item in scorer.evalImgs if item is not None
    )
    return matched / len(predictions), matched / len(ground_truth.anns)


def benchmark_cpu(checkpoint, manifest, expected_images, imgsz=640):
    torch.set_num_threads(THREADS_CPU)
    model = YOLO(str(checkpoint))
    options = {
        "source": str(manifest), "imgsz": imgsz, "device": "cpu", "batch": 1,
        "conf": CONF_FIXED, "max_det": 100, "stream": True, "verbose": False,
    }
    warmup = model.predict(**options)
    list(islice(warmup, 10))
    warmup.close()
    torch.set_num_threads(THREADS_CPU)
    times = [sum(result.speed.values()) for result in model.predict(**options)]
    if len(times) != expected_images:
        raise ValueError("Jumlah gambar benchmark CPU tidak sesuai dengan manifest val.")
    return {"cpu_median_ms": statistics.median(times), "cpu_p95_ms": float(np.percentile(times, 95))}


def save_metrics(path, row):
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def write_csv(validation):
    paths = [validation / run / "metrik.json" for run in CHECKPOINTS]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths if path.is_file()]
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (validation / "eksperimen_mosaic_val.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_yaml", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--runs", nargs="+", choices=tuple(CHECKPOINTS), default=("E0_640", "E6_640_no_mosaic"))
    parser.add_argument("--cpu-benchmark", action="store_true", help="Add CPU latency to completed validation runs")
    args = parser.parse_args()

    validation = OUTPUT / "evaluasi"
    if args.cpu_benchmark:
        for run in args.runs:
            run_dir = validation / run
            metrics_path = run_dir / "metrik.json"
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = OUTPUT / CHECKPOINTS[run]
            latency = benchmark_cpu(checkpoint, run_dir / "coco_val_inputs.txt", row["gambar"])
            latency["cpu_torch_threads"] = THREADS_CPU
            row.update(latency)
            save_metrics(metrics_path, row)
            write_csv(validation)
            print(run, json.dumps(latency), flush=True)
        return

    gt = COCO(str(GT_PATH))
    image_ids = sorted(gt.imgs)
    if image_ids != list(range(1, 357)) or len(gt.anns) != 2205:
        raise ValueError("Ground truth val berbeda dari referensi M1.")
    val_dir = args.dataset_yaml.resolve().parent / "images/val"
    images = [val_dir / gt.imgs[image_id]["file_name"] for image_id in image_ids]
    if {path.resolve() for path in val_dir.rglob("*.jpg")} != {path.resolve() for path in images}:
        raise ValueError("Daftar gambar val berbeda dari ground truth M1.")
    ap_small_for_model = m1_ap_small()

    for run in args.runs:
        run_dir = validation / run
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = OUTPUT / CHECKPOINTS[run]
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(args.dataset_yaml), split="val", imgsz=640, device=args.device,
            project=str(run_dir), name="ultralytics", max_det=100, workers=0,
            plots=False, exist_ok=True,
        )
        ap_small = ap_small_for_model(model, 640, args.device, GT_PATH, images, run_dir)
        predictions = json.loads((run_dir / "coco_val_predictions.json").read_text(encoding="utf-8"))
        precision_fixed, recall_fixed = precision_recall_fixed(gt, predictions)
        row = {
            "id": run, "split": "val", "gambar": len(images), "presisi": float(metrics.box.mp),
            "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map), "AP-small": ap_small,
            "presisi_conf_0.05_iou_0.5": precision_fixed,
            "recall_conf_0.05_iou_0.5": recall_fixed,
            "checkpoint": CHECKPOINTS[run],
        }
        save_metrics(run_dir / "metrik.json", row)
        write_csv(validation)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
