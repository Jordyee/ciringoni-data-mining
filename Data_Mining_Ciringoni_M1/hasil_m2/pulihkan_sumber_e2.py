"""Download verified E2 train and validation source images into a local cache."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from http.client import HTTPException
from io import TextIOWrapper
import json
from pathlib import Path
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile


def table(archive, name):
    with TextIOWrapper(archive.open(name), encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def source_rows(archive):
    inventory = defaultdict(list)
    for row in table(archive, "inventaris.csv"):
        if row["punya_gambar"] == "True":
            inventory[(row["stem"], row["source_id"])].append(row)
    sources = []
    for entry in table(archive, "pembagian_split.csv"):
        if entry["split"] not in {"train", "val"}:
            continue
        stem = Path(entry["file"]).stem
        (original,) = inventory[(stem, entry["source_id"])]
        digest = original["md5"]
        if len(digest) != 32 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid inventory MD5 for {stem}")
        sources.append(dict(original, stem=stem, split=entry["split"]))
    if Counter(row["split"] for row in sources) != {"train": 1244, "val": 356}:
        raise ValueError("Unexpected UTS train / validation population")
    return sources


def get_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "md5").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--seed-cache", type=Path)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        sources = source_rows(archive)
    random.Random(42).shuffle(sources)
    args.cache.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + args.seconds
    cooldown = 0.0
    recent_transient = 0
    lock = threading.Lock()
    stop = threading.Event()

    def fetch(row):
        nonlocal cooldown, recent_transient
        digest = row["md5"]
        target = args.cache / f"{digest}.raw"
        if target.is_file() and get_digest(target) == digest:
            return row, "cached", target.stat().st_size
        if args.seed_cache is not None:
            existing = args.seed_cache / f"{row['split']}-{Path(row['file']).name}"
            if existing.is_file() and get_digest(existing) == digest:
                target.write_bytes(existing.read_bytes())
                return row, "cached", existing.stat().st_size
        query = urlencode({"id": row["id_gambar"], "export": "download", "confirm": "t"})
        request = Request("https://drive.usercontent.google.com/download?" + query, headers={"User-Agent": "Mozilla/5.0"})
        last_error = "unattempted"
        for attempt in range(3):
            with lock:
                pause = max(0, cooldown - time.monotonic())
            if time.monotonic() + pause >= deadline or stop.wait(pause):
                return row, "budget_or_throttle", 0
            try:
                with urlopen(request, timeout=14) as stream:
                    raw = stream.read(25_000_001)
            except HTTPError as error:
                last_error = f"http_{error.code}"
                if error.code not in {429, 500, 502, 503, 504}:
                    return row, last_error, 0
                retry_after = error.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else 3 * 2**attempt
                with lock:
                    cooldown = max(cooldown, time.monotonic() + delay)
                    recent_transient += 1
                    if recent_transient >= 12:
                        stop.set()
            except (OSError, URLError, HTTPException) as error:
                last_error = type(error).__name__
                if time.monotonic() + 2**attempt >= deadline or stop.wait(2**attempt):
                    break
            else:
                if hashlib.md5(raw).hexdigest() != digest:
                    return row, "hash_or_access_failure", 0
                temp = target.with_suffix(".part")
                temp.write_bytes(raw)
                temp.replace(target)
                with lock:
                    recent_transient = max(0, recent_transient - 1)
                return row, "downloaded", len(raw)
        return row, last_error, 0

    progress = args.cache / "progres_unduh_e2.json"
    outcomes = []
    elapsed = 0

    def persist():
        counts = dict(Counter(row["status"] for row in outcomes))
        payload = {"expected": len(sources), "seconds": round(time.monotonic() - start), "counts": counts, "files": outcomes}
        staging = progress.with_suffix(".part")
        staging.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        staging.replace(progress)
        print(payload["seconds"], counts, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, row) for row in sources]
        for future in as_completed(futures):
            row, status, size = future.result()
            outcomes.append({"split": row["split"], "filename": row["file"], "status": status, "bytes": size})
            if time.monotonic() - start - elapsed >= 15:
                persist()
                elapsed = time.monotonic() - start
    persist()
    assert len(outcomes) == len(sources)


if __name__ == "__main__":
    main()
