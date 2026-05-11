"""
prepare_dataset.py
==================
Download the FULL NIH Chest X-ray dataset (all 12 zip files), extract all
images into a common images/ folder, and build a manifest CSV for the MIA
pipeline.

CRITICAL: The 50/50 member/non-member split is performed on the COMPLETE
~112,120-image dataset, NOT on a subset.  This reflects the real-world MIA
scenario where the attacker has access to all images but does not know which
50% were used to train the victim model.

Outputs
-------
  images/           ← extracted PNGs from all 12 zip files
  manifest.csv      ← columns: path, label, label_idx, split

Usage
-----
  python prepare_dataset.py [--data_dir DATA_DIR] [--images_dir IMAGES_DIR]

The script is idempotent: already-downloaded zips and already-extracted
images are skipped automatically.
"""

import os
import sys
import zipfile
import shutil
import argparse
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# Place data artefacts next to the script in Victim_Model/
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
# Data_Entry_2017.csv is expected at the project root (one level up)
LABEL_CSV   = os.path.join(os.path.dirname(BASE_DIR), "Data_Entry_2017.csv")
MANIFEST    = os.path.join(BASE_DIR, "manifest.csv")

# HuggingFace mirror for the NIH Chest X-ray dataset
HF_BASE = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset"
    "/resolve/main/data"
)

# All 12 zip files (images_001 … images_012)
ZIP_NAMES = [f"images_{i:03d}.zip" for i in range(1, 13)]
ZIP_URLS  = [f"{HF_BASE}/images/{name}" for name in ZIP_NAMES]

# 15 disease classes (order matches the training scripts)
DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]
NUM_CLASSES  = len(DISEASE_CLASSES)   # 15

MEMBER_RATIO = 0.50   # 50 % members, 50 % non-members
RANDOM_SEED  = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: str):
    """Download *url* to *dest* with a simple progress bar. Skips if present."""
    if os.path.exists(dest):
        mb = os.path.getsize(dest) / 1_048_576
        print(f"  Already present: {os.path.basename(dest)} ({mb:.1f} MB) — skipping.")
        return

    print(f"  Downloading: {url}")
    print(f"          ->: {dest}")

    last_pct = [-1]

    def _hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct != last_pct[0] and pct % 5 == 0:
            mb_done  = downloaded   / 1_048_576
            mb_total = total_size   / 1_048_576
            print(f"    {pct:3d}%  ({mb_done:7.1f} / {mb_total:7.1f} MB)", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print(f"  Done: {os.path.getsize(dest) / 1_048_576:.1f} MB", flush=True)


def _extract_zip(zip_path: str, images_dir: str):
    """Extract PNG files from *zip_path* into *images_dir* (flat layout)."""
    os.makedirs(images_dir, exist_ok=True)
    print(f"  Extracting {os.path.basename(zip_path)} -> {images_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        png_members = [m for m in zf.namelist() if m.lower().endswith(".png")]
        print(f"    {len(png_members)} PNG files in archive.")

        for i, member in enumerate(png_members):
            target_name = os.path.basename(member)
            if not target_name:
                continue
            target_path = os.path.join(images_dir, target_name)
            if os.path.exists(target_path):
                continue          # already extracted
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if (i + 1) % 2000 == 0:
                print(f"    extracted {i + 1}/{len(png_members)}", flush=True)

    total = len([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
    print(f"  Done. Total PNGs in {images_dir}: {total}", flush=True)


def _multi_hot(finding_str: str, label_to_idx: dict) -> list:
    """Convert a pipe-separated finding string into a 15-dim multi-hot list."""
    vec = [0] * NUM_CLASSES
    for tag in str(finding_str).split("|"):
        tag = tag.strip()
        if tag in label_to_idx:
            vec[label_to_idx[tag]] = 1
    return vec


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download NIH Chest X-ray dataset and build manifest.csv"
    )
    parser.add_argument("--data_dir",   default=DATA_DIR,   help="Directory for zip files")
    parser.add_argument("--images_dir", default=IMAGES_DIR, help="Directory for extracted images")
    parser.add_argument("--label_csv",  default=LABEL_CSV,  help="Path to Data_Entry_2017.csv")
    parser.add_argument("--manifest",   default=MANIFEST,   help="Output manifest CSV path")
    parser.add_argument(
        "--zips", type=int, default=12,
        help="Number of zip files to download (1-12). Default: 12 (full dataset)"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # ── 1. Download zip files ─────────────────────────────────────────────────
    n_zips = min(args.zips, 12)
    print(f"\n[1/4] Downloading {n_zips} zip file(s) from HuggingFace …")
    zip_paths = []
    for name, url in zip(ZIP_NAMES[:n_zips], ZIP_URLS[:n_zips]):
        dest = os.path.join(args.data_dir, name)
        _download(url, dest)
        zip_paths.append(dest)

    # ── 2. Extract images ─────────────────────────────────────────────────────
    existing_count = (
        len([f for f in os.listdir(args.images_dir) if f.lower().endswith(".png")])
        if os.path.exists(args.images_dir)
        else 0
    )

    # Estimate expected count (each zip has ~9,000-10,000 images)
    expected_min = n_zips * 8_000
    print(f"\n[2/4] Extracting images (already have {existing_count}) …")
    if existing_count >= expected_min:
        print(f"  Looks like images already extracted ({existing_count} PNGs) — skipping.")
    else:
        for zp in zip_paths:
            _extract_zip(zp, args.images_dir)

    # ── 3. Build multi-label manifest ─────────────────────────────────────────
    print(f"\n[3/4] Building manifest from {args.label_csv} …")

    if not os.path.exists(args.label_csv):
        print(f"  ERROR: Label CSV not found at {args.label_csv}")
        print("  Please place Data_Entry_2017.csv in the project root directory.")
        sys.exit(1)

    df = pd.read_csv(args.label_csv)
    print(f"  Label CSV rows: {len(df)}")

    # Filter to images that were actually extracted
    available = set(os.listdir(args.images_dir))
    df = df[df["Image Index"].isin(available)].copy()
    print(f"  Rows matching extracted images: {len(df)}")

    if len(df) == 0:
        print("  ERROR: No matching images found.")
        sys.exit(1)

    # Build multi-hot label vectors
    label_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}

    df["label_idx"] = df["Finding Labels"].apply(
        lambda s: str(_multi_hot(s, label_to_idx))
    )
    df["label"] = df["Finding Labels"]

    # Class distribution
    label_counter = Counter()
    for finding in df["Finding Labels"]:
        for tag in str(finding).split("|"):
            tag = tag.strip()
            if tag in label_to_idx:
                label_counter[tag] += 1

    print(f"\n  Disease distribution ({NUM_CLASSES} classes):")
    for name in DISEASE_CLASSES:
        print(f"    {name:25s}  {label_counter.get(name, 0):7d}")

    # ── 4. 50/50 member / non-member split ───────────────────────────────────
    print(f"\n[4/4] Splitting into member / non-member (seed={RANDOM_SEED}) …")
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    n_total   = len(df)
    n_members = int(n_total * MEMBER_RATIO)

    member_df    = df.iloc[:n_members].copy()
    nonmember_df = df.iloc[n_members:].copy()

    member_df["split"]    = "member"
    nonmember_df["split"] = "nonmember"

    print(f"  Total images matched: {n_total}")
    print(f"  Members:              {len(member_df)}")
    print(f"  Non-members:          {len(nonmember_df)}")

    # Build manifest
    manifest = pd.concat([member_df, nonmember_df], ignore_index=True)
    manifest["path"] = manifest["Image Index"].apply(
        lambda fn: os.path.join(args.images_dir, fn)
    )
    manifest = manifest[["path", "label", "label_idx", "split"]]
    manifest.to_csv(args.manifest, index=False)

    print(f"\n[DONE] Manifest written to: {args.manifest}")
    print(f"  Total rows: {len(manifest)}")
    print(f"  Columns:    {list(manifest.columns)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
