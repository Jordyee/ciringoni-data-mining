"""Verify source geometry and build matched native E2 640 and 1280 datasets."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
from statistics import mean
import subprocess
import time
from zipfile import ZipFile

from PIL import Image, ImageChops, ImageOps, ImageStat

from pulihkan_sumber_e2 import get_digest, source_rows


def open_heic(raw, decoded):
    if not decoded.is_file():
        env = dict(os.environ, CIR_E2_HEIC_INPUT=str(raw), CIR_E2_HEIC_OUTPUT=str(decoded))
        command = """
        $ErrorActionPreference = 'Stop'
        Add-Type -AssemblyName PresentationCore
        $source = [IO.File]::OpenRead($env:CIR_E2_HEIC_INPUT)
        try {
            $decoder = [System.Windows.Media.Imaging.BitmapDecoder]::Create(
                $source, [System.Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
                [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad)
            $encoder = [System.Windows.Media.Imaging.PngBitmapEncoder]::new()
            $encoder.Frames.Add($decoder.Frames[0])
            $stream = [IO.File]::Create($env:CIR_E2_HEIC_OUTPUT)
            try { $encoder.Save($stream) } finally { $stream.Dispose() }
        } finally { $source.Dispose() }
        """
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            env=env, check=True, capture_output=True, timeout=30,
        )
    return Image.open(decoded)


def layout(image, target, dim_round=round, offset_round=round):
    scale = target / max(image.size)
    width = min(target, max(1, dim_round(image.width * scale)))
    height = min(target, max(1, dim_round(image.height * scale)))
    return width, height, offset_round((target - width) / 2), offset_round((target - height) / 2)


def difference(image, baseline, color, geometry, resample):
    width, height, left, top = geometry
    candidate = Image.new("RGB", baseline.size, color)
    candidate.paste(image.resize((width, height), resample), (left, top))
    return mean(ImageStat.Stat(ImageChops.difference(candidate, baseline)).mean)


def match_geometry(source, baseline, inventory_size):
    orientation = source.getexif().get(274, 1)
    images = [("raw", source.convert("RGB"))]
    if orientation in range(2, 9):
        images.append(("exif", ImageOps.exif_transpose(source).convert("RGB")))
    image_sizes = {image.size for _, image in images}
    if inventory_size not in image_sizes:
        raise ValueError(f"Inventory dimensions {inventory_size} disagree with decoded sizes {sorted(image_sizes)}")
    color = baseline.getpixel((0, 0))
    matches = []
    for name, image in images:
        geometry = layout(image, 640)
        score = difference(image, baseline, color, geometry, Image.Resampling.BICUBIC)
        matches.append((score, name, image, geometry, round, round))
    best = min(matches, key=lambda match: match[0])
    if best[0] <= 4.5:
        return best

    if orientation == 1:
        image = images[0][1]
        images += [
            ("rot90", image.transpose(Image.Transpose.ROTATE_90)),
            ("rot180", image.transpose(Image.Transpose.ROTATE_180)),
            ("rot270", image.transpose(Image.Transpose.ROTATE_270)),
        ]
        for name, image in images[1:]:
            geometry = layout(image, 640)
            score = difference(image, baseline, color, geometry, Image.Resampling.BICUBIC)
            matches.append((score, name, image, geometry, round, round))
    best = min(matches, key=lambda match: match[0])
    _, name, image, _, _, _ = best
    modes = (round, math.floor, math.ceil)
    checked = set()
    for dim_round in modes:
        for offset_round in modes:
            geometry = layout(image, 640, dim_round, offset_round)
            if geometry in checked:
                continue
            checked.add(geometry)
            for resample in (Image.Resampling.BICUBIC, Image.Resampling.BILINEAR):
                score = difference(image, baseline, color, geometry, resample)
                matches.append((score, name, image, geometry, dim_round, offset_round))
    return min(matches, key=lambda match: match[0])


def remap(boxes, old, target_geometry, target):
    w640, h640, x640, y640 = old
    width, height, left, top = target_geometry
    scale_x, scale_y = width / w640, height / h640
    output = []
    for klass, x, y, w, h in boxes:
        if klass != 0 or not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
            raise ValueError("Malformed original UTS box")
        if (
            x * 640 - w * 320 < x640 - 1.05 or x * 640 + w * 320 > x640 + w640 + 1.05
            or y * 640 - h * 320 < y640 - 1.05 or y * 640 + h * 320 > y640 + h640 + 1.05
        ):
            raise ValueError("UTS box overlaps padding by more than one pixel")
        translated_x = ((x * 640 - x640) * scale_x + left) / target
        translated_y = ((y * 640 - y640) * scale_y + top) / target
        new_w, new_h = w * 640 * scale_x / target, h * 640 * scale_y / target
        if (
            min(translated_x - new_w / 2, translated_y - new_h / 2) < -1e-5
            or max(translated_x + new_w / 2, translated_y + new_h / 2) > 1 + 1e-5
        ):
            raise ValueError("Transformed box exceeds output frame")
        assert abs((((translated_x * target - left) / scale_x + x640) / 640) - x) < 1e-9
        assert abs((((translated_y * target - top) / scale_y + y640) / 640) - y) < 1e-9
        output.append(f"{int(klass)} {translated_x:.8f} {translated_y:.8f} {new_w:.8f} {new_h:.8f}\n")
    return "".join(output)


def build_one(archive, row, cache, output):
    stem, split, digest = row["stem"], row["split"], row["md5"]
    result = {"split": split, "filename": row["file"]}
    raw = cache / f"{digest}.raw"
    if not raw.is_file():
        return dict(result, status="source_missing")
    if get_digest(raw) != digest:
        return dict(result, status="source_hash_failed")
    try:
        with Image.open(BytesIO(archive.read(f"dataset/images/{split}/{stem}.jpg"))) as image:
            baseline = image.convert("RGB")
        source_image = (
            open_heic(raw, cache / f"{digest}.png") if Path(row["file"]).suffix.lower() == ".heic"
            else Image.open(raw)
        )
        with source_image as source:
            inventory_size = (int(float(row["lebar"])), int(float(row["tinggi"])))
            score, orientation, image, old, dim_round, offset_round = match_geometry(source, baseline, inventory_size)
        result.update(mae=round(score, 4), orientation=orientation, source_size=image.size, geometry_640=old)
        if score > 8.0:
            return dict(result, status="geometry_review")
        text = archive.read(f"dataset/labels/{split}/{stem}.txt").decode("utf-8-sig")
        boxes = [tuple(map(float, line.split())) for line in text.splitlines() if line.strip()]
        color = baseline.getpixel((0, 0))
        exports = []
        for target in (640, 1280):
            new = old if target == 640 else layout(image, target, dim_round, offset_round)
            width, height, left, top = new
            padded = Image.new("RGB", (target, target), color)
            padded.paste(image.resize((width, height), Image.Resampling.BICUBIC), (left, top))
            exports.append((target, padded, remap(boxes, old, new, target)))
            result[f"content_at_{target}"] = [left, top, width, height]
        for target, padded, label in exports:
            (output / str(target) / "labels" / split / f"{stem}.txt").write_text(label, encoding="utf-8")
            padded.save(output / str(target) / "images" / split / f"{stem}.jpg", format="JPEG", quality=95)
        return dict(result, status="paired", boxes=len(boxes))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(result, status="review", reason=type(error).__name__, detail=str(error)[:220])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    for size in ("640", "1280"):
        for artifact in ("images", "labels"):
            for split in ("train", "val"):
                (output / size / artifact / split).mkdir(parents=True, exist_ok=True)
    def persist(results):
        counts = dict(Counter(row["status"] for row in results))
        boxes = Counter()
        for row in results:
            if row["status"] == "paired":
                boxes[row["split"]] += row["boxes"]
        ready = counts == {"paired": 1600} and boxes == {"train": 10540, "val": 2205}
        report = {"ready": ready, "counts": counts, "boxes": boxes, "files": results}
        stage = output / "audit_pasangan_e2.part"
        stage.write_text(json.dumps(report, indent=2), encoding="utf-8")
        stage.replace(output / "audit_pasangan_e2.json")
        return ready, counts, boxes

    for size in ("640", "1280"):
        (output / size / "data.yaml").unlink(missing_ok=True)
    with ZipFile(args.archive) as archive:
        rows = source_rows(archive)
        if args.sample_count:
            available = [row for row in rows if (args.cache / f"{row['md5']}.raw").is_file()]
            rows = random.Random(42).sample(available, min(args.sample_count, len(available)))
        pool_size = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            observed = {}
            deadline = time.monotonic() + args.wait_seconds
            while True:
                available = [
                    row for row in rows if row["stem"] not in observed
                    and (args.cache / f"{row['md5']}.raw").is_file()
                ]
                for source, result in zip(available, pool.map(
                    lambda row: build_one(archive, row, args.cache, output), available
                )):
                    observed[source["stem"]] = result
                results = [
                    observed.get(row["stem"], {"split": row["split"], "filename": row["file"], "status": "source_missing"})
                    for row in rows
                ]
                ready, counts, boxes = persist(results)
                if len(observed) == len(rows) or time.monotonic() >= deadline:
                    break
                time.sleep(min(10, deadline - time.monotonic()))
    for size in ("640", "1280"):
        if ready:
            (output / size / "data.yaml").write_text(
                f"path: {json.dumps((output / size).as_posix())}\n"
                "train: images/train\nval: images/val\nnames:\n  0: infected\n",
                encoding="utf-8",
            )
    print(json.dumps({"ready": ready, "counts": counts, "boxes": boxes}))


if __name__ == "__main__":
    main()
