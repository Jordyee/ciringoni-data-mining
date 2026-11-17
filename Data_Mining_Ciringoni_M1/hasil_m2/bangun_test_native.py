"""Audit and build the held-out native 1280 set without running inference."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from bangun_pasangan_e2 import build_one
from pulihkan_sumber_e2 import SPLIT_COUNTS, source_rows, table


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_splits(archive, sources):
    assignments = {field: {} for field in ("stem", "id_gambar", "md5", "source_id", "grup")}
    for row in sources:
        for field, splits in assignments.items():
            value, split = row[field], row["split"]
            if not value or (value in splits and splits[value] != split):
                raise ValueError(f"Missing or cross-split key in {field}")
            splits[value] = split
    for field in ("stem", "id_gambar", "md5"):
        if len(assignments[field]) != len(sources):
            raise ValueError(f"Duplicate {field} among retained images")
    by_stem = assignments["stem"]
    retained = [
        row for row in table(archive, "pasangan_hampir_kembar.csv")
        if row["stem_a"] in by_stem and row["stem_b"] in by_stem
    ]
    if any(by_stem[row["stem_a"]] != by_stem[row["stem_b"]] for row in retained):
        raise ValueError("The UTS near-duplicate pairs cross split boundaries")
    return {"unique_keys": {field: len(values) for field, values in assignments.items()},
            "known_near_duplicate_pairs": len(retained)}


def build_test(archive_path, cache, output):
    if output.exists():
        raise FileExistsError(output)
    with ZipFile(archive_path) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError("UTS archive failed CRC verification")
        sources = source_rows(archive, tuple(SPLIT_COUNTS))
        split_audit = audit_splits(archive, sources)
        test = sorted((row for row in sources if row["split"] == "test"), key=lambda row: row["stem"])
        for size in (640, 1280):
            for artifact in ("images", "labels"):
                (output / str(size) / artifact / "test").mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda row: build_one(archive, row, cache, output), test))
        counts = Counter(row["status"] for row in results)
        report = {"ready": False, "counts": dict(counts), "archive_sha256": sha256(archive_path),
                  "splits": SPLIT_COUNTS, **split_audit, "files": results}
        audit_path = output / "audit_test_native.json"
        audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if counts != {"paired": SPLIT_COUNTS["test"]}:
            raise ValueError(f"Incomplete native test export: {dict(counts)}")
        images, annotations = [], []
        clipped = 0
        for image_id, (source, row) in enumerate(zip(test, results, strict=True), 1):
            stem = source["stem"]
            filename = f"{stem}.jpg"
            images.append({"id": image_id, "file_name": filename, "width": 640, "height": 640})
            row.update(source_md5=source["md5"], image_id=image_id, sha256={})
            originals = archive.read(f"dataset/labels/test/{stem}.txt").decode("utf-8-sig").split()
            if len(originals) != 5 * row["boxes"]:
                raise ValueError("UTS test box count differs from the source audit")
            old_x, old_y, old_w, old_h = row["content_at_640"]
            boxes = [tuple(map(float, originals[i:i + 5])) for i in range(0, len(originals), 5)]
            for size in (640, 1280):
                image_path = output / str(size) / "images/test" / filename
                label_path = output / str(size) / "labels/test" / f"{stem}.txt"
                with Image.open(image_path) as image:
                    if image.format != "JPEG" or image.size != (size, size):
                        raise ValueError("Invalid native test export")
                    image.verify()
                row["sha256"][str(size)] = {"image": sha256(image_path), "label": sha256(label_path)}
                new_x, new_y, new_w, new_h = row[f"content_at_{size}"]
                labels = label_path.read_text(encoding="utf-8").splitlines()
                for (_, x, y, w, h), label in zip(boxes, labels, strict=True):
                    cls, nx, ny, nw, nh = map(float, label.split())
                    restored = ((nx * size - new_x) * old_w / new_w + old_x,
                                (ny * size - new_y) * old_h / new_h + old_y,
                                nw * size * old_w / new_w, nh * size * old_h / new_h)
                    original = (x * 640, y * 640, w * 640, h * 640)
                    if cls != 0 or any(abs(a - b) > 0.0001 for a, b in zip(original, restored, strict=True)):
                        raise ValueError("Native test box failed inverse geometry check")
            for _, x, y, w, h in boxes:
                relative = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                x0, y0, x1, y1 = bounded = (
                    max(0.0, relative[0]), max(0.0, relative[1]),
                    min(1.0, relative[2]), min(1.0, relative[3]),
                )
                clipped += bounded != relative
                w, h = (x1 - x0) * 640, (y1 - y0) * 640
                if min(w, h) <= 0:
                    raise ValueError("Empty M1 reference box after clipping")
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                    "bbox": [x0 * 640, y0 * 640, w, h], "area": w * h, "iscrowd": 0})
        if len(annotations) != 1159:
            raise ValueError("UTS test ground truth must contain 1,159 boxes")
    for size in (640, 1280):
        if {path.name for path in (output / str(size) / "images/test").iterdir()} != {row["file_name"] for row in images}:
            raise ValueError("Native test image IDs must match ground truth exactly")
    gt = output / "coco_test_gt.json"
    gt.write_text(json.dumps({"info": {"description": "Held-out UTS test in M1 coordinates"},
                              "images": images, "annotations": annotations,
                              "categories": [{"id": 1, "name": "infected"}]}) + "\n", encoding="utf-8")
    report.update(ready=True, images=len(images), boxes=len(annotations),
                  reference_clipped_boxes=clipped, ground_truth_sha256=sha256(gt))
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_test(args.archive, args.cache, args.output.resolve())
