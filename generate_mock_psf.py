#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import hashlib


def stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def moffat_patch(radius, peak, fwhm, beta, q=1.0, theta=0.0):
    """
    Generate an elliptical Moffat PSF patch (center peak = peak).

    Args:
        radius: Patch radius; output shape is (2r+1, 2r+1)
        q: Axis ratio (0 < q <= 1), q=1 means circular symmetry
        theta: Rotation angle in radians
    """
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")

    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * xx + st * yy
    yr = -st * xx + ct * yy

    # Relationship between Moffat alpha and FWHM:
    # FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
    denom = 2.0 * np.sqrt(np.power(2.0, 1.0 / beta) - 1.0) + 1e-12
    alpha = fwhm / denom

    rr2 = (xr * xr + (yr / max(q, 1e-6)) ** 2) / (alpha * alpha + 1e-12)
    patch = peak * np.power(1.0 + rr2, -beta)
    return patch.astype(np.float32)


def disk_mask(radius):
    ax = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    return (xx * xx + yy * yy) <= (radius * radius)


def place_mock_psfs(
    os_img,
    bg_mask,
    calib_img,
    rng,
    n_psf_min=3,
    n_psf_max=8,
    edge_margin=64,
    min_center_dist=48,
    peak_min=500.0,
    peak_max=5000.0,
    max_candidates_scan=200000,
    tail_frac_for_mask=0.01,
):
    """
    Returns:
      src_only_img  : Mock sources only (no background)
      mock_gt_img   : Mock GT (write source-only PSF only in mock regions; zero elsewhere)
      mock_mask_img : Mock source region mask (uint8)
      centers_info  : List of placed source info
    """
    H, W = os_img.shape[:2]
    assert os_img.ndim == 2, f"os_img must be 2D, got shape={os_img.shape}"
    assert bg_mask.shape == os_img.shape
    assert calib_img.shape == os_img.shape

    bg_mask_bool = (bg_mask > 0)

    # Candidate centers: background region + away from edges + finite values
    valid = bg_mask_bool.copy()
    valid &= np.isfinite(os_img)
    valid &= np.isfinite(calib_img)

    valid[:edge_margin, :] = False
    valid[-edge_margin:, :] = False
    valid[:, :edge_margin] = False
    valid[:, -edge_margin:] = False

    ys, xs = np.where(valid)
    if len(ys) == 0:
        raise RuntimeError("No valid background candidate positions (edge_margin may be too large, or mask is unexpected).")

    # Shuffle candidate order
    perm = rng.permutation(len(ys))
    if len(perm) > max_candidates_scan:
        perm = perm[:max_candidates_scan]

    # Output arrays
    src_only = np.zeros_like(os_img, dtype=np.float32)
    mock_mask_img = np.zeros_like(os_img, dtype=np.uint8)
    mock_gt = np.zeros_like(os_img, dtype=np.float32)

    centers = []  # [(y, x, exclusion_radius), ...]

    # Random target number of PSFs
    if n_psf_max < n_psf_min:
        n_psf_max = n_psf_min
    target_n = int(rng.integers(n_psf_min, n_psf_max + 1))

    for idx in perm:
        if len(centers) >= target_n:
            break

        cy = int(ys[idx])
        cx = int(xs[idx])

        # Random PSF shape parameters (tunable)
        r_support = int(rng.integers(8, 16))          # support radius (pixels)
        fwhm = float(rng.uniform(2.0, 5.0))           # FWHM (pixels)
        beta = float(rng.uniform(2.5, 4.5))           # Moffat beta
        q = float(rng.uniform(0.85, 1.0))             # near-circular
        theta = float(rng.uniform(0.0, np.pi))

        # Dynamic minimum distance: at least min_center_dist and consider source size
        excl_r = max(min_center_dist, 2 * r_support + 10)

        # Keep distance from previously placed centers
        ok_dist = True
        for py, px, pr in centers:
            d2 = (cy - py) ** 2 + (cx - px) ** 2
            min_d = max(excl_r, pr)
            if d2 < (min_d ** 2):
                ok_dist = False
                break
        if not ok_dist:
            continue

        y0, y1 = cy - r_support, cy + r_support + 1
        x0, x1 = cx - r_support, cx + r_support + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue

        patch_bgmask = bg_mask_bool[y0:y1, x0:x1]
        patch_calib = calib_img[y0:y1, x0:x1]

        dmask = disk_mask(r_support)
        # Require the entire support disk to be inside the background region
        if patch_bgmask.shape != dmask.shape:
            continue
        if not np.all(patch_bgmask[dmask]):
            continue

        # Estimate local background mean from calib image (inside support disk)
        local_bg = float(np.mean(patch_calib[dmask]))

        # Random target center peak after adding background
        target_peak_total = float(rng.uniform(peak_min, peak_max))
        src_peak = target_peak_total - local_bg

        # Skip if background is too high and source peak becomes non-positive
        if src_peak <= 5.0:
            continue

        # Generate source patch (source-only)
        patch_src = moffat_patch(
            radius=r_support,
            peak=src_peak,
            fwhm=fwhm,
            beta=beta,
            q=q,
            theta=theta,
        )

        # Mock-PSF mask region (threshold truncation + limited to support disk)
        thr = max(1.0, tail_frac_for_mask * src_peak)
        use = (patch_src >= thr) & dmask

        # Ensure selected region is still fully in background
        if not np.all(patch_bgmask[use]):
            continue

        # Avoid overlap with previously placed mock regions
        existed = mock_mask_img[y0:y1, x0:x1] > 0
        if np.any(existed & use):
            continue

        # Write source-only (used to create os_mock)
        region_src = src_only[y0:y1, x0:x1]
        region_src[use] += patch_src[use]

        # Write GT (source-only PSF, no background added)
        region_gt = mock_gt[y0:y1, x0:x1]
        region_gt[use] = patch_src[use]

        # Write mask
        region_m = mock_mask_img[y0:y1, x0:x1]
        region_m[use] = 1

        centers.append((cy, cx, excl_r))

    return src_only, mock_gt, mock_mask_img, centers


def process_one_os_file(
    os_path: Path,
    seed: int,
    n_psf_min: int,
    n_psf_max: int,
    edge_margin: int,
    min_center_dist: int,
    peak_min: float,
    peak_max: float,
    overwrite: bool = False,
):
    """
    Process a single *_os.npy file and generate:
      *_mock_psf_mask.npy
      *_mock_psf_gt.npy
      *_os_mock.npy
    """
    name = os_path.name
    if not name.endswith("_os.npy"):
        return False, f"skip (not *_os.npy): {os_path}"

    # Base path without the "_os.npy" suffix
    base = os_path.with_name(name[:-len("_os.npy")])
    mask_path = Path(str(base) + "_mask.npy")
    calib_path = Path(str(base) + "_calib_ours.npy")

    out_mask_path = Path(str(base) + "_mock_psf_mask.npy")
    out_gt_path = Path(str(base) + "_mock_psf_gt.npy")
    out_osmock_path = Path(str(base) + "_os_mock.npy")

    if (not overwrite) and out_mask_path.exists() and out_gt_path.exists() and out_osmock_path.exists():
        return True, f"exists, skip: {os_path}"

    if not mask_path.exists():
        return False, f"missing mask: {mask_path}"
    if not calib_path.exists():
        return False, f"missing calib_ours: {calib_path}"

    # Load arrays
    os_img = np.load(os_path)
    bg_mask = np.load(mask_path)
    calib_img = np.load(calib_path)

    # Only 2D images are supported
    if os_img.ndim != 2:
        return False, f"os image is not 2D: {os_path}, shape={os_img.shape}"
    if bg_mask.shape != os_img.shape:
        return False, f"mask shape mismatch: {mask_path}, {bg_mask.shape} vs {os_img.shape}"
    if calib_img.shape != os_img.shape:
        return False, f"calib shape mismatch: {calib_path}, {calib_img.shape} vs {os_img.shape}"

    # Per-file reproducible random seed
    local_seed = (seed + stable_int_hash(str(os_path))) % (2**32 - 1)
    rng = np.random.default_rng(local_seed)

    # Place mock PSFs
    src_only, mock_gt, mock_mask_img, centers = place_mock_psfs(
        os_img=os_img.astype(np.float32),
        bg_mask=bg_mask,
        calib_img=calib_img.astype(np.float32),
        rng=rng,
        n_psf_min=n_psf_min,
        n_psf_max=n_psf_max,
        edge_margin=edge_margin,
        min_center_dist=min_center_dist,
        peak_min=peak_min,
        peak_max=peak_max,
    )

    # Create os_mock (original image + source-only)
    os_mock = os_img.astype(np.float32) + src_only

    # Save outputs (float32 / uint8)
    np.save(out_mask_path, mock_mask_img.astype(np.uint8))
    np.save(out_gt_path, mock_gt.astype(np.float32))
    np.save(out_osmock_path, os_mock.astype(np.float32))

    msg = (
        f"ok: {os_path.name} | placed={len(centers)} | "
        f"save -> {out_mask_path.name}, {out_gt_path.name}, {out_osmock_path.name}"
    )
    return True, msg


def main():
    parser = argparse.ArgumentParser(description="Generate mock PSFs for all *_os.npy recursively.")
    parser.add_argument("--data", type=str, required=True, help="Root directory, e.g. /path/to/muscat3_r")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--n-psf-min", type=int, default=32, help="Minimum number of mock PSFs per image")
    parser.add_argument("--n-psf-max", type=int, default=64, help="Maximum number of mock PSFs per image")
    parser.add_argument("--edge-margin", type=int, default=128, help="Minimum distance from image borders (pixels)")
    parser.add_argument("--min-center-dist", type=int, default=48, help="Minimum distance between source centers (pixels)")
    parser.add_argument("--peak-min", type=float, default=500.0, help="Minimum center peak value (after background addition)")
    parser.add_argument("--peak-max", type=float, default=5000.0, help="Maximum center peak value (after background addition)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    os_files = sorted(root.rglob("*_os.npy"))
    print(f"[INFO] Found {len(os_files)} files matching *_os.npy under: {root}")

    ok_cnt = 0
    fail_cnt = 0

    for p in os_files:
        ok, msg = process_one_os_file(
            os_path=p,
            seed=args.seed,
            n_psf_min=args.n_psf_min,
            n_psf_max=args.n_psf_max,
            edge_margin=args.edge_margin,
            min_center_dist=args.min_center_dist,
            peak_min=args.peak_min,
            peak_max=args.peak_max,
            overwrite=args.overwrite,
        )
        if ok:
            ok_cnt += 1
            print("[OK ]", msg)
        else:
            fail_cnt += 1
            print("[ERR]", msg)

    print(f"\n[DONE] success={ok_cnt}, failed={fail_cnt}")


if __name__ == "__main__":
    main()