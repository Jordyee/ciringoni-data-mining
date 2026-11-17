"""Export the selected native checkpoint and compare OpenVINO variants on validation."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import shutil

import openvino
import torch
from ultralytics import YOLO

from evaluasi_final import evaluate, inputs, sha256
from evaluasi_mosaic import save_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, choices=(640, 1280), required=True)
    args = parser.parse_args()
    size, native, out = args.size, args.native.resolve(), args.output.resolve()
    dataset = inputs(native, "val", size)
    root = native / str(size) / "images"
    train = sorted((root / "train").glob("*.jpg"))
    audit = json.loads((native / "audit_pasangan_e2.json").read_text(encoding="utf-8"))
    train_ids = {Path(row["filename"]).stem for row in audit["files"] if row["split"] == "train"}
    if (len(train) != 1244 or {path.stem for path in train} != train_ids
            or train_ids & {path.stem for path in dataset[1]} or (root / "test").exists()):
        raise ValueError("INT8 calibration requires the audited, separate training split")
    calibration = sorted(Random(42).sample(train, 320))
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "candidate.pt"
    shutil.copy2(args.checkpoint, checkpoint)
    manifest = out / "calibration_train.txt"
    manifest.write_text("".join(path.as_posix() + "\n" for path in calibration), encoding="utf-8")
    calibration_yaml = out / "calibration.yaml"
    save_metrics(calibration_yaml, {
        "path": out.as_posix(), "train": manifest.as_posix(),
        "val": (root / "val").as_posix(), "names": ["infected"],
    })
    common = dict(format="openvino", imgsz=size, batch=1, device="cpu", workers=0,
                  dynamic=False, nms=False, max_det=100)
    torch.set_num_threads(4)
    exports = {
        "fp32": Path(YOLO(str(checkpoint)).export(**common, quantize=32)),
        "int8": Path(YOLO(str(checkpoint)).export(**common, quantize=8, data=str(calibration_yaml), split="train")),
    }
    metrics = {
        "checkpoint_sha256": sha256(checkpoint), "calibration_split": "train", "calibration_seed": 42,
        "calibration_images": len(calibration), "calibration_ids_sha256": hashlib.sha256(
            "\n".join(path.stem for path in calibration).encode("utf-8")
        ).hexdigest(),
        "exports": {},
    }
    core = openvino.Core()
    for precision, folder in exports.items():
        ir = core.read_model(next(folder.glob("*.xml")))
        op_types = [node.get_type_name() for node in ir.get_ops()]
        quantized = op_types.count("FakeQuantize")
        if list(ir.input().shape) != [1, 3, size, size] or bool(quantized) != (precision == "int8"):
            raise ValueError(f"Unexpected OpenVINO graph shape or quantization for {precision}")
        result_dir = out / precision
        evaluate(folder.resolve(), size, dataset, result_dir)
        path = result_dir / "metrics.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["export"] = {"precision": precision, "checkpoint_sha256": metrics["checkpoint_sha256"]}
        row["model_bytes"] = sum(file.stat().st_size for file in folder.iterdir() if file.suffix in {".xml", ".bin"})
        row["fake_quantize_ops"] = quantized
        save_metrics(path, row)
        metrics["exports"][precision] = row
    if metrics["exports"]["fp32"]["runtime"] != metrics["exports"]["int8"]["runtime"]:
        raise ValueError("OpenVINO variants ran with different CPU settings")
    save_metrics(out / "comparison.json", metrics)


if __name__ == "__main__":
    main()
