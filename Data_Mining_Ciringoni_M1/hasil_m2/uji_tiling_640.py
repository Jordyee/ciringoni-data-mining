import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO


OUTPUT = Path(__file__).resolve().parent / "uji_tiling_640"
M1 = OUTPUT.parent.parent / "hasil_m1"
SIZE = 640
TILE = 384
OFFSETS = [(x, y) for y in (0, SIZE - TILE) for x in (0, SIZE - TILE)]
MIN_VISIBILITY = 0.25
CONF = 0.001
MERGE_IOU = 0.50
MAX_DET = 100
CPU_THREADS = 3


def train_boxes(label_path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        category, cx, cy, w, h = map(float, line.split())
        boxes.append((int(category), (cx-w/2)*SIZE, (cy-h/2)*SIZE, (cx+w/2)*SIZE, (cy+h/2)*SIZE))
    return boxes


def clip_box(box, tile_x, tile_y):
    category, x0, y0, x1, y1 = box
    left = max(0.0, x0 - tile_x)
    top = max(0.0, y0 - tile_y)
    right = min(TILE, x1 - tile_x)
    bottom = min(TILE, y1 - tile_y)
    width, height = right - left, bottom - top
    if min(width, height) < 1 or width * height < MIN_VISIBILITY * (x1-x0) * (y1-y0):
        return None
    return (category, left, top, right, bottom)


def export_train(dataset, destination):
    if destination.exists():
        raise FileExistsError(f"Pilih folder baru agar hasil ekspor sebelumnya tetap utuh: {destination}")
    image_dir = destination / "images/train"
    label_dir = destination / "labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    counts = Counter()
    for path in sorted((dataset / "images/train").glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (SIZE, SIZE):
            raise ValueError(f"Citra train tidak terbaca atau berbeda ukuran: {path.name}")
        boxes = train_boxes(dataset / "labels/train" / (path.stem + ".txt"))
        for x, y in OFFSETS:
            stem = f"{path.stem}_x{x}_y{y}"
            if not cv2.imwrite(str(image_dir / (stem + ".jpg")), image[y:y+TILE, x:x+TILE]):
                raise OSError(f"Gagal menulis {stem}.jpg")
            labels = []
            for box in boxes:
                clipped = clip_box(box, x, y)
                if clipped is None:
                    continue
                category, left, top, right, bottom = clipped
                cx, cy = (left+right)/(2*TILE), (top+bottom)/(2*TILE)
                width, height = (right-left)/TILE, (bottom-top)/TILE
                labels.append(f"{category} {cx:.9f} {cy:.9f} {width:.9f} {height:.9f}\n")
            (label_dir / (stem + ".txt")).write_text("".join(labels), encoding="utf-8")
            counts.update(tiles=1, tile_boxes=len(labels), empty_tiles=not labels)
    expected = json.loads((OUTPUT / "audit_geometri_train.json").read_text(encoding="utf-8"))
    if any(expected[name] != value for name, value in counts.items()):
        raise ValueError(f"Hasil ekspor berbeda dari audit: {dict(counts)}")
    val = (dataset.resolve() / "images/val").as_posix()
    config = {"path": destination.resolve().as_posix(), "train": "images/train", "val": val, "names": ["infected"]}
    (destination / "data.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save("ekspor_train_e3.json", dict(dataset_yaml=str(destination / "data.yaml"), **counts))
    print(f"Ekspor E3: {dict(counts)}. Konfigurasi: {destination / 'data.yaml'}", flush=True)


def draw_boxes(image, boxes, color):
    for left, top, right, bottom in boxes:
        p0 = (round(left), round(top))
        p1 = (round(right), round(bottom))
        cv2.rectangle(image, p0, p1, color, 2)
    return image


def montage(path, title, images, labels):
    width, header, footer = 310, 55, 36
    canvas = np.full((width + header + footer, width * len(images), 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (12, 34), font, 0.73, (18, 18, 18), 2, cv2.LINE_AA)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = width * i
        canvas[header:header+width, x:x+width] = cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, label, (x + 9, header + width + 23), font, 0.48, (18, 18, 18), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Gagal menulis {path.name}")


def geometry(dataset):
    assert clip_box((0, 350, 300, 410, 340), 0, 0) == (0, 350, 300, 384, 340)
    assert clip_box((0, 380, 300, 410, 340), 0, 0) is None
    assert clip_box((0, 260, 264, 300, 315), 256, 256) == (0, 4, 8, 44, 59)
    counts = Counter()
    evidence = None
    for path in sorted((dataset / "images/train").glob("*.jpg")):
        boxes = train_boxes(dataset / "labels/train" / (path.stem + ".txt"))
        retained = set()
        counts.update(images=1, input_boxes=len(boxes), tiles=len(OFFSETS))
        for tile_x, tile_y in OFFSETS:
            inside = []
            cut = 0
            for index, box in enumerate(boxes):
                clipped = clip_box(box, tile_x, tile_y)
                if clipped is None:
                    continue
                assert 0 <= clipped[1] < clipped[3] <= TILE and 0 <= clipped[2] < clipped[4] <= TILE
                inside.append(clipped)
                retained.add(index)
                cut += not (0 <= box[1]-tile_x and box[3]-tile_x <= TILE and 0 <= box[2]-tile_y and box[4]-tile_y <= TILE)
            counts.update(tile_boxes=len(inside), empty_tiles=not inside, clipped_boxes=cut)
            if cut and not evidence:
                evidence = (path, boxes, inside, tile_x, tile_y)
        counts.update(original_boxes_retained=len(retained))
    if counts["images"] != 1244 or evidence is None:
        raise ValueError(f"Pemeriksaan geometri tidak lengkap: {dict(counts)}")
    path, boxes, clipped, tile_x, tile_y = evidence
    image = cv2.imread(str(path))
    original = draw_boxes(image.copy(), [row[1:] for row in boxes], (40, 230, 65))
    cv2.rectangle(original, (tile_x, tile_y), (tile_x+TILE, tile_y+TILE), (255, 180, 20), 2)
    tile = draw_boxes(image[tile_y:tile_y+TILE, tile_x:tile_x+TILE].copy(), [row[1:] for row in clipped], (40, 230, 65))
    montage(OUTPUT / "bukti_geometri_e3.png", "Uji geometri E3 | Kotak anotasi train", [original, tile], ["Citra 640 dan posisi tile", "Tile 384 dan label setelah potong"])
    return dict(counts, original_boxes_lost=counts["input_boxes"]-counts["original_boxes_retained"], evidence_image=path.name)


def infer(model, image, tiled):
    start = perf_counter()
    offsets = OFFSETS if tiled else [(0, 0)]
    crops = [image[y:y+TILE, x:x+TILE].copy() for x, y in offsets] if tiled else [image]
    predictions = model.predict(crops, imgsz=SIZE, device="cpu", conf=CONF, max_det=MAX_DET, verbose=False)
    boxes = torch.cat([
        result.boxes.xyxy.cpu() + torch.tensor([x, y, x, y])
        for result, (x, y) in zip(predictions, offsets, strict=True)
    ]).clamp_(0, SIZE)
    scores = torch.cat([result.boxes.conf.cpu() for result in predictions])
    classes = torch.cat([result.boxes.cls.cpu() for result in predictions]).long()
    valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    before_merge = len(boxes)
    if tiled:
        keep = batched_nms(boxes, scores, classes, MERGE_IOU)[:MAX_DET]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    rows = [([x0, y0, x1-x0, y1-y0], score, category+1) for (x0, y0, x1, y1), score, category in zip(boxes.tolist(), scores.tolist(), classes.tolist(), strict=True)]
    return rows, 1000*(perf_counter()-start), before_merge


def score(gt, predictions, ids):
    if predictions:
        detections = gt.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = dict(images=gt.dataset["images"], categories=gt.dataset["categories"], annotations=[])
        detections.createIndex()
    evaluation = COCOeval(gt, detections, "bbox")
    evaluation.params.imgIds = ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return {"ap_50_95": float(evaluation.stats[0]), "ap_50": float(evaluation.stats[1]), "ap_small": float(evaluation.stats[3])}


def evaluate(dataset, limit):
    gt = COCO(str(M1 / "coco_val_gt.json"))
    ids = sorted(gt.imgs)
    if len(ids) != 356 or len(gt.anns) != 2205:
        raise ValueError("Ground truth val berbeda dari M1")
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(1)
    model = YOLO(str(M1 / "runs/E0/weights/best.pt"))
    ids = ids[:limit]
    modes = ("full", "tiled")
    results = {mode: [] for mode in modes}
    timing = {mode: [] for mode in modes}
    before = Counter()
    evidence = None
    for index, image_id in enumerate(ids):
        name = gt.imgs[image_id]["file_name"]
        image = cv2.imread(str(dataset / "images/val" / name))
        if image is None or image.shape[:2] != (SIZE, SIZE):
            raise ValueError(f"Citra val tidak terbaca atau berbeda ukuran: {name}")
        if index == 0:
            for _ in range(5):
                infer(model, image, False)
                infer(model, image, True)
            torch.set_num_threads(CPU_THREADS)
        by_mode = {}
        # Alternate order so the same mode does not always run first.
        for mode in (modes if index % 2 == 0 else modes[::-1]):
            rows, elapsed, count = infer(model, image, mode == "tiled")
            by_mode[mode] = rows
            timing[mode].append(elapsed)
            before[mode] += count
            results[mode].extend(dict(image_id=image_id, bbox=bbox, score=confidence, category_id=category) for bbox, confidence, category in rows)
        if evidence is None and all(any(row[1] >= 0.10 for row in by_mode[mode]) for mode in modes):
            labels = gt.loadAnns(gt.getAnnIds(imgIds=[image_id]))
            evidence = (name, image, labels, by_mode)
        if (index + 1) % 25 == 0 or index + 1 == len(ids):
            print(f"val {index + 1}/{len(ids)}", flush=True)
    if torch.get_num_threads() != CPU_THREADS:
        raise RuntimeError("The configured CPU thread count changed during timing.")
    if evidence:
        name, image, labels, by_mode = evidence
        ground = draw_boxes(image.copy(), [(x,y,x+w,y+h) for x,y,w,h in (ann["bbox"] for ann in labels)], (40, 230, 65))
        panels = [ground]
        for mode in modes:
            panels.append(draw_boxes(image.copy(), [(x,y,x+w,y+h) for (x,y,w,h), conf, _ in by_mode[mode] if conf >= 0.10], (20, 120, 255)))
        montage(OUTPUT / "bukti_inferensi_tiling.png", "Uji inferensi tiling | Citra validation yang sama", panels, ["Anotasi acuan", "Penuh 640, conf >= 0,10", "Tile 384 + NMS, conf >= 0,10"])
        print(f"Contoh gambar: {name}", flush=True)
    print("COCOeval:", flush=True)
    metrics = {}
    for mode in modes:
        (OUTPUT / f"prediksi_{mode}.json").write_text(json.dumps(results[mode]), encoding="utf-8")
        metrics[mode] = dict(
            score(gt, results[mode], ids),
            cpu_median_ms=median(timing[mode]), cpu_p95_ms=float(np.percentile(timing[mode], 95)),
            candidates_before_merge=before[mode], candidates_after_merge=len(results[mode]),
        )
    return dict(images=len(ids), image_example=evidence[0] if evidence else None, **metrics)


def save(name, data):
    (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Existing processed 640 YOLO dataset root")
    parser.add_argument("--limit", type=int, default=356)
    parser.add_argument("--export-train", type=Path, help="Write train tiles to a new directory using the finished audit")
    args = parser.parse_args()
    if not 1 <= args.limit <= 356:
        parser.error("--limit harus antara 1 dan 356")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.export_train:
        export_train(args.dataset, args.export_train)
        raise SystemExit(0)
    print(f"Mulai {datetime.now().astimezone().isoformat(timespec='seconds')}", flush=True)
    audit = geometry(args.dataset)
    save("audit_geometri_train.json", dict(tile=TILE, overlap_px=2*TILE-SIZE, min_visibility=MIN_VISIBILITY, **audit))
    print(json.dumps(audit), flush=True)
    metrics = evaluate(args.dataset, args.limit)
    save("metrik_tiling_val.json", dict(
        checkpoint="hasil_m1/runs/E0/weights/best.pt", input_imgsz=SIZE, tile=TILE, overlap_px=2*TILE-SIZE,
        conf=CONF, merge_iou=MERGE_IOU, max_det=MAX_DET, cpu_threads=CPU_THREADS, **metrics,
    ))
    print(json.dumps(metrics), flush=True)
    print(f"Selesai {datetime.now().astimezone().isoformat(timespec='seconds')}", flush=True)
