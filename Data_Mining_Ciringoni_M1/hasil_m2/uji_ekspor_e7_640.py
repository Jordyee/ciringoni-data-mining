"""Export and compare FP32 and INT8 for the provisional processed-640 control."""

import argparse
import json
from pathlib import Path
from random import Random
import shutil

import nncf
import openvino
from pycocotools.coco import COCO
import torch
from ultralytics import YOLO

from evaluasi_mosaic import GT_PATH, benchmark_cpu, m1_ap_small, precision_recall_fixed, save_metrics
from uji_tiling_640 import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("processed_dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.processed_dataset.resolve()
    out = args.output.resolve()

    gt = COCO(str(GT_PATH))
    ids = sorted(gt.imgs)
    val = [root / "images" / "val" / gt.imgs[image_id]["file_name"] for image_id in ids]
    train = sorted((root / "images" / "train").glob("*.jpg"))
    if len(train) != 1244 or len(val) != 356 or not all(path.is_file() for path in val):
        raise ValueError("Expected the complete processed 640 train and M1 validation splits.")
    calibration = sorted(Random(42).sample(train, 320))
    if set(calibration) & set(val):
        raise ValueError("Calibration must use the training split.")

    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "e0_640.pt"
    shutil.copy2(args.checkpoint, checkpoint)
    inputs = out / "calibration_train.txt"
    inputs.write_text("".join(path.as_posix() + "\n" for path in calibration), encoding="utf-8")
    calibration_yaml = out / "calibration.yaml"
    save_metrics(calibration_yaml, {
        "path": out.as_posix(), "train": inputs.as_posix(),
        "val": (root / "images" / "val").as_posix(), "names": ["infected"],
    })
    torch.set_num_threads(4)
    common = {"format": "openvino", "imgsz": 640, "batch": 1, "device": "cpu", "workers": 0}
    exports = {
        "fp32": Path(YOLO(str(checkpoint)).export(**common)),
        "int8": Path(YOLO(str(checkpoint)).export(
            **common, quantize=8, data=str(calibration_yaml), split="train",
        )),
    }
    metrics = {
        "checkpoint": str(args.checkpoint), "selection": "provisional control, processed 640",
        "openvino_version": openvino.__version__, "nncf_version": nncf.__version__,
        "calibration_split": "train", "calibration_images": len(calibration),
        "validation_images": len(val),
    }
    for name, folder in exports.items():
        result_dir = out / f"validasi_{name}"
        result_dir.mkdir()
        model = YOLO(str(folder), task="detect")
        m1_ap_small()(model, 640, "cpu", GT_PATH, val, result_dir)
        predictions = json.loads((result_dir / "coco_val_predictions.json").read_text(encoding="utf-8"))
        precision, recall = precision_recall_fixed(gt, predictions)
        metrics[name] = {
            **score(gt, predictions, ids),
            "precision_conf_0.05_iou_0.5": precision, "recall_conf_0.05_iou_0.5": recall,
            "model_bytes": sum(path.stat().st_size for path in folder.iterdir() if path.suffix in {".xml", ".bin"}),
            **benchmark_cpu(folder, result_dir / "coco_val_inputs.txt", len(val)),
        }
    save_metrics(out / "metrik_ekspor_e7.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
