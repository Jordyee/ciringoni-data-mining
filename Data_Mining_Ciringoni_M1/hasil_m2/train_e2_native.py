import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import torch
from ultralytics import YOLO


WIB = timezone(timedelta(hours=7))
EXPECTED = {"train": 1244, "val": 356}
EPOCHS = 5


def verify_pair(native):
    audit = json.loads((native / "audit_pasangan_e2.json").read_text(encoding="utf-8"))
    if not audit["ready"] or audit["counts"] != {"paired": 1600}:
        raise ValueError("The complete pair of native exports is pending.")
    for split, count in EXPECTED.items():
        sets = [{path.stem for path in (native / str(size) / f"images/{split}").glob("*.jpg")} for size in (640, 1280)]
        if sets[0] != sets[1] or len(sets[0]) != count:
            raise ValueError(f"Expected matching ID sets with {count} {split} images.")
    if not all((native / str(size) / "data.yaml").is_file() for size in (640, 1280)):
        raise ValueError("Both native-export dataset YAML files are required.")


def train(weights, data, work, name, size, batch, epochs, smoke=False):
    torch.cuda.reset_peak_memory_stats()
    print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | training started | {name}", flush=True)
    model = YOLO(weights)
    model.train(
        data=str(data), project=str(work), name=name, imgsz=size, epochs=epochs,
        batch=batch, seed=42, deterministic=True, device=0, workers=0,
        mosaic=1.0, close_mosaic=0, optimizer="AdamW", lr0=0.002, momentum=0.9,
        save=not smoke, plots=not smoke,
    )
    actual = model.trainer.batch_size
    print(f"{name} | actual_batch={actual} | peak_reserved_mib={torch.cuda.max_memory_reserved() / 1024**2:.1f}", flush=True)
    if actual != batch:
        raise RuntimeError(f"{name} reduced batch from {batch} to {actual}. Retry the pair at a smaller shared batch.")
    if not (work / name / "weights/best.pt").is_file():
        raise RuntimeError(f"Missing checkpoint for {name}.")
    del model
    torch.cuda.empty_cache()
    print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | training finished | {name}", flush=True)


def smoke_yaml(native, work, batch):
    data = native / "1280"
    largest = sorted((data / "labels/train").glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True)[:batch]
    images = [data / "images/train" / f"{label.stem}.jpg" for label in largest]
    val = sorted((data / "images/val").glob("*.jpg"))[:2 * batch]
    for split, paths in (("train", images), ("val", val)):
        (work / f"smoke_{split}.txt").write_text("".join(f"{path.as_posix()}\n" for path in paths), encoding="utf-8")
    config = {"path": work.as_posix(), "train": "smoke_train.txt", "val": "smoke_val.txt", "names": ["infected"]}
    yaml = work / "smoke.yaml"
    yaml.write_text(json.dumps(config), encoding="utf-8")
    return yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train native-source YOLO26n at 640 and 1280 under matching settings.")
    parser.add_argument("native", type=Path)
    parser.add_argument("--batch", type=int, choices=(2, 4), default=4)
    parser.add_argument("--smoke-only", action="store_true", help="Check 1280 GPU memory with one batch only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    native = args.native.resolve()
    verify_pair(native)
    output = Path(__file__).resolve().parent / "eksperimen_e2_native"
    weights = output.parent.parent / "data/yolo26n.pt"
    if output.exists() and not args.smoke_only:
        raise FileExistsError(output)
    with TemporaryDirectory(prefix="ciringoni-e2-train-", delete=args.smoke_only) as folder:
        work = Path(folder)
        print(f"Local scratch for checkpoint recovery | {work}", flush=True)
        train(weights, smoke_yaml(native, work, args.batch), work, "E2_smoke_1280", 1280, args.batch, 1, smoke=True)
        if not args.smoke_only:
            runs = work / "runs"
            train(weights, native / "1280/data.yaml", runs, "E2_native_1280", 1280, args.batch, EPOCHS)
            train(weights, native / "640/data.yaml", runs, "E2_control_native_640", 640, args.batch, EPOCHS)
            shutil.copytree(runs, output / "runs")
            print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | verified checkpoints published | {output}", flush=True)
