#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute PSNR & SSIM for a folder of dehazed images against clear (GT) images.

Usage:
  python dehaze_eval_psnr_ssim.py --gt-dir /path/to/val/clear --pred-dir /path/to/val_dcp --save-csv dcp_metrics.csv
  python dehaze_eval_psnr_ssim.py --gt-dir /path/to/val/clear --pred-dir /path/to/val_aod --strip-suffix _AOD-Net --save-csv aod_metrics.csv
  python dehaze_eval_psnr_ssim.py --gt-dir /path/to/val/clear --pred-dir /path/to/val_ffa --save-csv ffa_metrics.csv --recursive

Notes:
- This script matches files by *stem* (filename without extension). If your predicted
  filenames include suffixes (e.g., 'IMG_0001_AOD-Net.jpg'), pass --strip-suffix _AOD-Net.
- Different image sizes are auto-resized (pred -> GT size) before metric computation.
- Outputs a CSV with per-image PSNR/SSIM and prints the averages.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def imread_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def strip_suffixes(stem: str, extra_suffixes):
    COMMON = ["_AOD-Net", "_DCP", "_FFA", "_ffa", "_dehaze", "_dehazed", "_out"]
    # extra_suffixes may contain None; filter falsy
    extras = [s for s in (extra_suffixes or []) if s]
    for suf in extras + COMMON:
        if suf and stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def find_gt(gt_dir: Path, stem: str):
    for ext in EXTS:
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # also try case-insensitive match
    low = stem.lower()
    for ext in EXTS:
        for p in gt_dir.glob(f"*{ext}"):
            if p.stem.lower() == low:
                return p
    return None


def compute_metrics(gt_bgr, pr_bgr):
    if gt_bgr.shape != pr_bgr.shape:
        pr_bgr = cv2.resize(pr_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pr = cv2.cvtColor(pr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    P = float(psnr(gt, pr, data_range=1.0))
    S = float(ssim(gt, pr, channel_axis=2, data_range=1.0))
    return P, S


def list_images(root: Path, recursive: bool = False):
    if recursive:
        for p in root.rglob("*"):
            if p.suffix.lower() in EXTS and p.is_file():
                yield p
    else:
        for p in root.iterdir():
            if p.suffix.lower() in EXTS and p.is_file():
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=str, required=True, help="Folder of clear (ground-truth) images")
    ap.add_argument("--pred-dir", type=str, required=True, help="Folder of dehazed images to evaluate")
    ap.add_argument(
        "--strip-suffix", type=str, default=None, help="Optional suffix to drop from predicted stems, e.g., _AOD-Net"
    )
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--save-csv", type=str, default=None, help="Optional path to write per-image metrics CSV")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pr_dir = Path(args.pred_dir)
    rows = []
    miss = 0
    psnrs, ssims = [], []

    files = list(list_images(pr_dir, recursive=args.recursive))
    if not files:
        print(f"[WARN] No images under {pr_dir}")
        return

    for pr_path in files:
        stem = strip_suffixes(pr_path.stem, [args.strip_suffix])
        gt_path = find_gt(gt_dir, stem)
        if gt_path is None:
            miss += 1
            print(f"[MISS] GT not found for {pr_path.name} (stem='{stem}')")
            continue

        try:
            gt = imread_any(gt_path)
            pr = imread_any(pr_path)
            P, S = compute_metrics(gt, pr)
            psnrs.append(P)
            ssims.append(S)
            rows.append([pr_path.name, gt_path.name, f"{P:.4f}", f"{S:.6f}"])
            print(f"[OK] {pr_path.name} -> PSNR={P:.2f}dB, SSIM={S:.4f}")
        except Exception as e:
            print(f"[FAIL] {pr_path.name} -> {e}")

    if rows:
        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))
        print("\n================ SUMMARY ================")
        print(f"Count: {len(rows)} | Missed GT: {miss}")
        print(f"Average PSNR: {mean_psnr:.3f} dB")
        print(f"Average SSIM: {mean_ssim:.5f}")

        if args.save_csv:
            outp = Path(args.save_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pred_file", "gt_file", "psnr", "ssim"])
                w.writerows(rows)
            print(f"[CSV] Wrote {outp}")
    else:
        print("[INFO] No pairs matched; nothing to summarize.")


if __name__ == "__main__":
    main()
