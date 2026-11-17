import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile

import torch
from ultralytics import YOLO


WIB = timezone(timedelta(hours=7))
STEPS = 620
BATCH = 8
LR = 0.002
RUNS = {"E3_control_full_640": 4, "E3_tile384_640": 1}


def prepare_data(dataset, tiled_dataset, output):
    images = sorted((dataset / "images/train").glob("*.jpg"))[:1240]
    val = dataset / "images/val"
    if len(images) != 1240 or len(list(val.glob("*.jpg"))) != 356:
        raise ValueError("Expected 1,240 selected train images and 356 validation images.")

    tiles = [
        tiled_dataset / "images/train" / f"{image.stem}_x{x}_y{y}.jpg"
        for image in images for y in (0, 256) for x in (0, 256)
    ]
    if len(set(tiles)) != 4960 or any(not tile.is_file() for tile in tiles):
        raise ValueError("Expected four distinct 384-pixel tiles per selected source image.")
    output.mkdir(exist_ok=False)
    for name, paths in zip(RUNS, (images, tiles)):
        if len(paths) * RUNS[name] != BATCH * STEPS:
            raise ValueError(f"{name} has a different batch budget.")
        manifest = output / f"{name}.txt"
        manifest.write_text("".join(f"{path.as_posix()}\n" for path in paths), encoding="utf-8")
        config = {"path": output.as_posix(), "train": manifest.as_posix(), "val": val.as_posix(), "names": ["infected"]}
        (output / f"{name}.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def train_one(output, weights, name, epochs):
    counts = {"minibatches": 0, "optimizer_steps": 0}

    def count_step(*_):
        counts["optimizer_steps"] += 1

    def on_start(trainer):
        if len(trainer.train_loader) * epochs != STEPS or trainer.accumulate != 1:
            raise RuntimeError(f"{name}: unexpected train loader length or optimizer accumulation.")
        # The default end-to-end loss decay runs at epoch boundaries. Hold it fixed for both arms.
        trainer.model.criterion = trainer.model.init_criterion()
        trainer.model.criterion.final_o2m = trainer.model.criterion.o2m_copy
        trainer.scaler = torch.amp.GradScaler("cuda", init_scale=256.0, growth_interval=STEPS + 1)
        trainer.optimizer.register_step_post_hook(count_step)

    def count_batch(trainer):
        counts["minibatches"] += 1

    print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | training started | {name}", flush=True)
    model = YOLO(weights)
    model.add_callback("on_train_start", on_start)
    model.add_callback("on_train_batch_end", count_batch)
    model.train(
        data=str(output / f"{name}.yaml"), project=str(output / "runs"), name=name,
        epochs=epochs, imgsz=640, batch=BATCH, nbs=BATCH, device=0, workers=0,
        seed=42, deterministic=True, mosaic=0.0, close_mosaic=0, val=False, save=False,
        optimizer="AdamW", lr0=LR, lrf=1.0, warmup_epochs=0.0, patience=0,
    )
    checkpoint = output / "runs" / name / "weights/last.pt"
    if not checkpoint.is_file() or set(counts.values()) != {STEPS}:
        raise RuntimeError(f"{name} did not match the step budget: {counts}.")
    verified = {"name": name, "epochs": epochs, **counts, "checkpoint": checkpoint.relative_to(output).as_posix()}
    (output / f"{name}_steps.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | training finished | {name} | {counts}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the equal-step E3 processed-640 pair.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("tiled_dataset", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    output = Path(__file__).resolve().parent / "eksperimen_e3_640_verified"
    if output.exists():
        raise FileExistsError(output)
    weights = output.parent.parent / "data/yolo26n.pt"
    with tempfile.TemporaryDirectory(prefix="ciringoni-e3-") as folder:
        scratch = Path(folder) / "work"
        prepare_data(args.dataset.resolve(), args.tiled_dataset.resolve(), scratch)
        for name, epochs in RUNS.items():
            train_one(scratch, weights, name, epochs)
        shutil.copytree(scratch, output)
        for name in RUNS:
            config_path = output / f"{name}.yaml"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(path=output.as_posix(), train=f"{name}.txt")
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"{datetime.now(WIB).isoformat(timespec='seconds')} | verified checkpoints published | {output}", flush=True)
