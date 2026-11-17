import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


MOSAIC_PAIR = ("E0_640", "E6_640_no_mosaic")
P2_RUN = "E5_640_p2"


def train(data, epochs, names):
    output = Path(__file__).resolve().parent
    weights = output.parent / "data" / "yolo26n.pt"
    runs = output / "runs"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment.")
    if any((runs / name).exists() for name in names):
        raise SystemExit("An experiment directory already exists. Preserve those results before rerunning.")

    wib = timezone(timedelta(hours=7))
    print(f"PyTorch {torch.__version__} | Ultralytics {ultralytics.__version__} | {torch.cuda.get_device_name(0)}")
    for name in names:
        print(f"{datetime.now(wib).isoformat(timespec='seconds')} | training started | {name}")
        model = YOLO("yolo26n-p2.yaml" if name == P2_RUN else weights)
        pretrained = str(weights) if name == P2_RUN else True
        model.train(
            data=str(data.resolve()), imgsz=640, epochs=epochs, batch=8, seed=42,
            deterministic=True, device=0, workers=0, project=str(runs), name=name,
            pretrained=pretrained, mosaic=0.0 if name == "E6_640_no_mosaic" else 1.0,
            close_mosaic=0,
        )
        del model
        torch.cuda.empty_cache()
        print(f"{datetime.now(wib).isoformat(timespec='seconds')} | training finished | {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the YOLO26n control and 640-pixel ablations.")
    parser.add_argument("--data", type=Path, required=True, help="Existing 640-pixel YOLO dataset YAML")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--run", choices=(*MOSAIC_PAIR, P2_RUN), help="Run one experiment instead of the mosaic pair")
    args = parser.parse_args()
    train(args.data, args.epochs, (args.run,) if args.run else MOSAIC_PAIR)
