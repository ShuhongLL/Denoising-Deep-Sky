#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_fits_to_npy.py

Non-recursively iterate subdirectories under --input. In each subdir, look into its
'data' folder and find files whose names END WITH --suffix (e.g., "_synth.fits.fz" or "synth.fits.fz").
For each match:
  - Read HDU[1] first; if absent/no data, fall back to HDU[0].
  - Convert to float32 NumPy array.
  - Save to --output/<subdir>/<same_basename>.npy

Rules:
  * The corresponding output subdir (--output/<subdir>) must already exist; otherwise error.
  * --overwrite:
      - False: if ANY target .npy in that subdir already exists, skip the entire subdir.
      - True: overwrite files unconditionally.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits


def load_fits_primary_or_first(fp: Path) -> np.ndarray:
    """Return data from HDU[1] if available; otherwise HDU[0]. Preserves NaNs, float32."""
    with fits.open(fp, memmap=True) as hdul:
        arr = None
        if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None:
            arr = hdul[1].data
        elif getattr(hdul[0], "data", None) is not None:
            arr = hdul[0].data
        if arr is None:
            raise ValueError(f"No image data found in {fp}")
        return np.asarray(arr, dtype=np.float32)


def find_subdirs(root: Path) -> List[Path]:
    """List immediate subdirectories of root (non-recursive)."""
    return [p for p in root.iterdir() if p.is_dir()]


def strip_fits_like_suffix(filename: str) -> str:
    """Strip common FITS endings to build output basename."""
    if filename.endswith(".fits.fz"):
        return filename[: -len(".fits.fz")]
    if filename.endswith(".fits"):
        return filename[: -len(".fits")]
    if filename.endswith(".fz"):
        return filename[: -len(".fz")]
    # generic: remove last extension if any
    return filename.rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser(description="Convert FITS images to .npy in mirrored subdirs.")
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="Root path containing subdirectories (non-recursive).")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="Output root where subdirs must already exist.")
    ap.add_argument("--suffix", type=str, required=True,
                    help='Literal filename suffix to match (no wildcards), e.g. "_synth.fits.fz" or "synth.fits.fz".')
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, overwrite existing .npy files. If not set, skip subdir if any target exists.")
    args = ap.parse_args()

    in_root: Path = args.input.resolve()
    out_root: Path = args.output.resolve()
    suffix: str = args.suffix
    overwrite: bool = args.overwrite

    if not in_root.is_dir():
        print(f"[ERROR] --input not a directory: {in_root}", file=sys.stderr)
        sys.exit(2)
    if not out_root.is_dir():
        print(f"[ERROR] --output not a directory: {out_root}", file=sys.stderr)
        sys.exit(2)

    subdirs = find_subdirs(in_root)
    if not subdirs:
        print(f"[WARN] No subdirectories found under: {in_root}")

    missing_out_subdirs: List[Path] = []
    processed = 0
    skipped_exist = 0
    skipped_nomatch = 0

    for sub in sorted(subdirs):
        data_dir = sub / "data"
        if not data_dir.is_dir():
            print(f"[WARN] Missing 'data' folder: {data_dir}")
            continue

        out_subdir = out_root / sub.name
        if not out_subdir.is_dir():
            missing_out_subdirs.append(out_subdir)
            continue

        # Collect matches: literal tail match, case-sensitive
        matches = sorted(p for p in data_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
        if not matches:
            print(f"[INFO] No files ending with '{suffix}' in {data_dir}")
            skipped_nomatch += 1
            continue

        intended_outputs = [out_subdir / "data" /f"{strip_fits_like_suffix(fp.name)}.npy" for fp in matches]

        if not overwrite and any(p.exists() for p in intended_outputs):
            print(f"[SKIP] Existing outputs found in {out_subdir}; skipping subdir (overwrite disabled).")
            skipped_exist += 1
            continue

        for src_fp, dst_fp in zip(matches, intended_outputs):
            try:
                arr = load_fits_primary_or_first(src_fp)
            except Exception as e:
                print(f"[ERROR] Failed to read {src_fp}: {e}", file=sys.stderr)
                continue

            if not dst_fp.parent.is_dir():
                print(f"[ERROR] Output subdir missing (should exist): {dst_fp.parent}", file=sys.stderr)
                continue

            if dst_fp.exists() and not overwrite:
                print(f"[SKIP] Exists: {dst_fp}")
                continue

            try:
                np.save(dst_fp, arr)
                print(f"[OK] {src_fp.name} -> {dst_fp}")
                processed += 1
            except Exception as e:
                print(f"[ERROR] Failed to save {dst_fp}: {e}", file=sys.stderr)

    if missing_out_subdirs:
        print("\n[ERROR] These output subdirectories do NOT exist (must be created beforehand):", file=sys.stderr)
        for p in missing_out_subdirs:
            print(f"  - {p}", file=sys.stderr)
        print(f"\n[STATS] processed={processed}, skipped_exist={skipped_exist}, skipped_nomatch={skipped_nomatch}")
        sys.exit(1)

    print(f"\n[STATS] processed={processed}, skipped_exist={skipped_exist}, skipped_nomatch={skipped_nomatch}")


if __name__ == "__main__":
    main()
