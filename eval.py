import json
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits

# -----------------------------
# same constants as metric.py
# -----------------------------
LO = -200.0
HI = 2500.0
DR = HI - LO

# Optional SSIM dependency (same as metric.py)
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# -----------------------------
# same helpers as metric.py
# -----------------------------
def _np_clip(a: np.ndarray, lo: float = LO, hi: float = HI) -> np.ndarray:
    return np.clip(a, lo, hi).astype(np.float32, copy=False)

def _finite_fill(a: np.ndarray, fill: float | None = None) -> np.ndarray:
    out = a.astype(np.float32, copy=True)
    if fill is None:
        fill = float(np.nanmean(out)) if np.isfinite(out).any() else 0.0
    m = ~np.isfinite(out)
    if m.any():
        out[m] = fill
    return out

def _ssim2d(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a2 = _finite_fill(a)
    b2 = _finite_fill(b)
    if HAVE_SKIMAGE:
        return float(ssim_skimage(b2, a2, data_range=data_range))
    # coarse fallback (same as your metric.py)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    xmu = float(np.mean(a2)); ymu = float(np.mean(b2))
    xsig2 = float(np.var(a2, ddof=1)); ysig2 = float(np.var(b2, ddof=1))
    cov = float(np.mean((a2 - xmu) * (b2 - ymu)))
    num = (2 * xmu * ymu + C1) * (2 * cov + C2)
    den = (xmu**2 + ymu**2 + C1) * (xsig2 + ysig2 + C2)
    return float(num / den) if den != 0.0 else float("nan")

def compute_psnr_linear_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pr = _finite_fill(_np_clip(pred))
    g  = _finite_fill(_np_clip(gt))
    diff = (pr - g).astype(np.float64, copy=False)
    mse = float(np.mean(diff * diff, dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(DR) - 10.0 * np.log10(mse)

def compute_ssim_linear_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pr = _finite_fill(_np_clip(pred))
    g  = _finite_fill(_np_clip(gt))
    if g.ndim == 2:
        return _ssim2d(pr, g, DR)
    elif g.ndim == 3:
        C = g.shape[0]
        vals = []
        for c in range(C):
            vals.append(_ssim2d(pr[c], g[c], DR))
        return float(np.nanmean(np.array(vals, dtype=np.float64)))
    else:
        raise ValueError(f"Unsupported ndim for SSIM: {gt.ndim}")

def robust_nmad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))

def _masked_values_np(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if a.shape != m.shape:
        raise ValueError(f"Mask shape {m.shape} != image shape {a.shape}")
    m = m & np.isfinite(a)
    return a[m]

def read_fits_any(path: Path) -> np.ndarray:
    """Read fits / fits.fz -> float32 array (first HDU with data)."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float32)
    raise RuntimeError(f"No image data found in: {path}")

def read_array(path: Path) -> np.ndarray:
    """Read .npy or .fits/.fits.fz -> float32 array."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return read_fits_any(path)

def _find_file(directory: Path, stem_glob: str) -> list:
    """Find files matching stem_glob with .npy or .fits.fz extension (npy preferred)."""
    for ext in (".npy", ".fits.fz"):
        hits = list(directory.glob(f"{stem_glob}{ext}"))
        if hits:
            return hits
    return []

# -----------------------------
# eval one sample
# -----------------------------
def process_one(subdir: Path, pred_path: Path, label_suffix: str = "calib"):
    gtf   = _find_file(subdir, f"*_{label_suffix}")
    maskf = _find_file(subdir, "*_mask")
    if len(gtf) != 1 or len(maskf) != 1:
        return (subdir.name, None, f"[skip] {subdir.name}: gt/mask not unique")

    gt   = read_array(gtf[0])
    mask = np.load(maskf[0]) if maskf[0].suffix == ".npy" else read_fits_any(maskf[0])

    pred_hits = _find_file(pred_path, f"{subdir.name}_pred")
    if not pred_hits:
        return (subdir.name, None, f"[skip] {subdir.name}: pred missing in {pred_path}")
    pred = read_array(pred_hits[0])

    # PSNR/SSIM (same as metric.py: clipped + global DR)
    P = compute_psnr_linear_np(pred, gt)
    S = compute_ssim_linear_np(pred, gt)

    # NMAD/SIGMA on pred background (mask==1), pred itself (not residual)
    bg_mask = (mask == 1)
    vals = _masked_values_np(_finite_fill(_np_clip(pred)), bg_mask)
    N = robust_nmad(vals)
    G = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

    msg = f"{subdir.name}: PSNR={P:.3f}, SSIM={S:.6f}, NMAD_bg(pred)={N:.6f}, STD_bg(pred)={G:.6f}"
    return (subdir.name, (P, S, N, G), msg)

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--band_name", type=str, required=True)
    ap.add_argument("-y", "--label-suffix", default="calib", help="GT npy suffix, e.g. calib, mean")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    pred_path = Path(args.pred_path)
    band_name = args.band_name

    with open(data_path / "train_test_split.json", "r", encoding="utf-8") as f:
        test_names = set(json.load(f)["test"])

    subdirs = [p for p in data_path.iterdir() if p.is_dir() and p.name in test_names]

    all_psnr, all_ssim, all_nmad, all_sigma = [], [], [], []

    for sd in subdirs:
        name, metrics, msg = process_one(sd, pred_path, label_suffix=args.label_suffix)
        print(msg)
        if metrics is not None:
            P, S, N, G = metrics
            all_psnr.append(P)
            all_ssim.append(S)
            all_nmad.append(N)
            all_sigma.append(G)

    if all_psnr:
        mean_psnr  = float(np.nanmean(all_psnr))
        mean_ssim  = float(np.nanmean(all_ssim))
        mean_nmad  = float(np.nanmean(all_nmad))
        mean_sigma = float(np.nanmean(all_sigma))

        print("\n[MEAN over processed test images]")
        print(f"PSNR={mean_psnr:.3f}, SSIM={mean_ssim:.6f}, "
              f"NMAD_bg(pred)={mean_nmad:.6f}, STD_bg(pred)={mean_sigma:.6f}")

        out_json = pred_path / f"{band_name}.json"
        result = {
            "band_name": band_name,
            "num_images": int(len(all_psnr)),
            "LO": float(LO),
            "HI": float(HI),
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "nmad_bg_pred": mean_nmad,
            "sigma_bg_pred": mean_sigma,
            "std_bg_pred": mean_sigma,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVE] {out_json}")
    else:
        print("\n[MEAN] No images processed.")
