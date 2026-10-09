#!/usr/bin/env python3
"""
Download 10 TSPLIB TSP instances and package them into one ZIP file.

Output:
    data/
        bays29.tsp
        berlin52.tsp
        bier127.tsp
        a280.tsp
        lin318.tsp
        rd400.tsp
        u574.tsp
        d657.tsp
        rat783.tsp
        pr1002.tsp
    APGGS_10_TSP_Datasets.zip

Usage:
    python download_tsp_datasets.py

Requirements:
    Python 3.8+ (standard library only)
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


# Public TSPLIB files.
BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master"

DATASETS = [
    "bays29.tsp",
    "berlin52.tsp",
    "bier127.tsp",
    "a280.tsp",
    "lin318.tsp",
    "rd400.tsp",
    "u574.tsp",
    "d657.tsp",
    "rat783.tsp",
    "pr1002.tsp",
]

DATA_DIR = Path("data")
ZIP_NAME = "APGGS_10_TSP_Datasets.zip"

# Download settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
TIMEOUT_SECONDS = 30


def download_file(filename: str, destination: Path) -> bool:
    """Download one file with a few retries."""
    url = f"{BASE_URL}/{filename}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{attempt}/{MAX_RETRIES}] Downloading {filename} ...")

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "APGGS-TSPLIB-Dataset-Downloader/1.0"
                },
            )

            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                content = response.read()

            if not content:
                raise ValueError("Downloaded file is empty.")

            # Basic TSPLIB sanity check.
            sample = content[:4096].decode("utf-8", errors="replace")
            if "NAME:" not in sample or "DIMENSION:" not in sample:
                raise ValueError(
                    "Downloaded content does not look like a TSPLIB .tsp file."
                )

            destination.write_bytes(content)
            print(f"      Saved: {destination}")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"      Failed: {exc}")

            if attempt < MAX_RETRIES:
                print(f"      Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)

    return False


def create_zip(data_dir: Path, zip_path: Path, filenames: list[str]) -> None:
    """Create a ZIP containing the downloaded TSP files under data/."""
    print(f"\nCreating ZIP: {zip_path}")

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for filename in filenames:
            file_path = data_dir / filename
            # Store files under data/ inside the ZIP.
            zf.write(file_path, arcname=f"data/{filename}")

    print(f"ZIP created successfully: {zip_path}")


def main() -> None:
    """Download all datasets and create the ZIP archive."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    print("=" * 72)
    print("APGGS - TSPLIB 10-Dataset Downloader")
    print("=" * 72)
    print(f"Download directory: {DATA_DIR.resolve()}")
    print(f"ZIP output:         {Path(ZIP_NAME).resolve()}")
    print()

    for filename in DATASETS:
        destination = DATA_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            # Reuse an existing valid-looking file.
            try:
                sample = destination.read_bytes()[:4096].decode(
                    "utf-8", errors="replace"
                )
                if "NAME:" in sample and "DIMENSION:" in sample:
                    print(f"[SKIP] {filename} already exists.")
                    successful.append(filename)
                    continue
            except OSError:
                pass

        if download_file(filename, destination):
            successful.append(filename)
        else:
            failed.append(filename)

    print("\n" + "-" * 72)
    print(f"Successful downloads: {len(successful)}/{len(DATASETS)}")

    if failed:
        print("Failed downloads:")
        for filename in failed:
            print(f"  - {filename}")

        print("\nZIP was not created because one or more files are missing.")
        raise SystemExit(1)

    create_zip(DATA_DIR, Path(ZIP_NAME), successful)

    # Verify ZIP contents.
    print("\nVerifying ZIP contents...")
    with zipfile.ZipFile(ZIP_NAME, mode="r") as zf:
        expected = {f"data/{name}" for name in DATASETS}
        actual = set(zf.namelist())

        missing = expected - actual
        if missing:
            raise RuntimeError(f"ZIP verification failed. Missing: {sorted(missing)}")

        bad_files = zf.testzip()
        if bad_files is not None:
            raise RuntimeError(f"ZIP integrity check failed for: {bad_files}")

    print("ZIP integrity check passed.")
    print("\nDone!")
    print(f"Datasets folder: {DATA_DIR.resolve()}")
    print(f"ZIP archive:     {Path(ZIP_NAME).resolve()}")


if __name__ == "__main__":
    main()
