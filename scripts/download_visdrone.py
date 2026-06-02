"""Download VisDrone2019-DET dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import RAW_DATA_DIR


# Ultralytics mirror for VisDrone2019-DET
BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
VISDRONE_FILES = [
    "VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev.zip",
]


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


def download_visdrone(output_dir: Path):
    """Download and extract VisDrone."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in VISDRONE_FILES:
        url = f"{BASE_URL}/{filename}"
        zip_path = output_dir / filename

        if zip_path.exists():
            print(f"{filename} already exists, skipping download.")
        else:
            print(f"Downloading {filename}...")
            try:
                download_file(url, zip_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue

        # Extract if not already extracted
        extract_dir = output_dir / filename.replace(".zip", "")
        if extract_dir.exists():
            print(f"{extract_dir} already exists, skipping extraction.")
        else:
            print(f"Extracting {filename}...")
            extract_zip(zip_path, output_dir)

    print(f"\nVisDrone dataset ready at: {output_dir}")
    print("  Splits: train, val, test-dev")
    print("  Next step: run scripts/build_yolo_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Download VisDrone2019-DET dataset")
    parser.add_argument("--output", default=str(RAW_DATA_DIR / "visdrone"),
                        help="Output directory for dataset")
    args = parser.parse_args()

    download_visdrone(Path(args.output))


if __name__ == "__main__":
    main()
