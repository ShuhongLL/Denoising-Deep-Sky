#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npy_2d(fp: Path) -> np.ndarray:
    """
    Load a .npy file as a 2D float32 array. Accepts (H,W) or (H,W,1); returns (H,W) float32.
    """
    arr = np.load(fp, allow_pickle=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D array (H,W) or (H,W,1); got {arr.shape} from {fp}")
    return arr.astype(np.float32, copy=False)


def _to_chw1(arr2d: np.ndarray) -> np.ndarray:
    """(H,W) -> (1,H,W)"""
    if arr2d.ndim != 2:
        raise ValueError(f"_to_chw1 expects (H,W); got {arr2d.shape}")
    return arr2d[None, ...]


class FitsPairDataset(Dataset):
    """
      <root>/<subdir>/<subdir>_os.npy
      <root>/<subdir>/<subdir>_gt.npy
      <root>/<subdir>/<subdir>_mask.npy   (1=background)

    The root directory must contain train_test_split.json with "train"/"test" subdir lists.

    Returns:
      x:       torch.FloatTensor (1,H,W)
      y:       torch.FloatTensor (1,H,W)
      bg_mask: torch.BoolTensor   (1,H,W)  # True = background
      os_path: str
      gt_path: str
    """

    def __init__(
        self,
        root: str,
        split_json: str = "train_test_split.json",
        split: str = "train",
        input_suffix: str = "os",
        label_suffix: str = "calib_ours",
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        jpath = self.root / split_json
        if not jpath.exists():
            raise FileNotFoundError(f"Split file not found: {jpath}")
        with jpath.open("r", encoding="utf-8") as f:
            j = json.load(f)
        if split not in j:
            raise KeyError(f"Split '{split}' missing in JSON. Keys: {list(j.keys())}")

        self.subdirs: List[str] = sorted(list(j[split]))
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.pairs: List[Tuple[Path, Path, Path]] = []
        for name in self.subdirs:
            d = self.root / name
            if not d.is_dir():
                raise FileNotFoundError(f"Missing data dir: {d}")

            os_fp   = d / f"{name}_{input_suffix}.npy"
            gt_fp   = d / f"{name}_{label_suffix}.npy"
            mask_fp = d / f"{name}_mask.npy"

            if not os_fp.exists():
                raise FileNotFoundError(f"Missing file: {os_fp}")
            if not gt_fp.exists():
                raise FileNotFoundError(f"Missing file: {gt_fp}")
            if not mask_fp.exists():
                raise FileNotFoundError(f"Missing file: {mask_fp}")

            self.pairs.append((os_fp, gt_fp, mask_fp))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No samples found under {self.root} for split='{split}'")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        os_fp, gt_fp, mask_fp = self.pairs[idx]

        x2d = _load_npy_2d(os_fp)
        y2d = _load_npy_2d(gt_fp)

        if self.clip_min is not None or self.clip_max is not None:
            cmin = self.clip_min if self.clip_min is not None else float(np.nanmin([x2d, y2d]))
            cmax = self.clip_max if self.clip_max is not None else float(np.nanmax([x2d, y2d]))
            x2d = np.clip(x2d, cmin, cmax)
            y2d = np.clip(y2d, cmin, cmax)

        m = np.load(mask_fp, allow_pickle=False)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]
        if m.ndim != 2:
            raise ValueError(f"bg_mask must be 2D (H,W); got {m.shape} from {mask_fp}")

        m_u8 = np.asarray(m, dtype=np.uint8)
        if m_u8.size > 0 and m_u8.max() == 255 and m_u8.min() in (0, 255):
            m_u8 = (m_u8 // 255).astype(np.uint8)
        bg_bool = (m_u8 == 1)  # True=background
        bg = bg_bool[None, ...]  # (1,H,W) np.bool_

        # (H,W) -> (1,H,W)
        x = _to_chw1(x2d)
        y = _to_chw1(y2d)

        # numpy -> torch
        x_t = torch.from_numpy(x)                  # float32
        y_t = torch.from_numpy(y)                  # float32
        bg_t = torch.from_numpy(bg).to(torch.bool) # bool

        return x_t, y_t, bg_t, str(os_fp), str(gt_fp)