# -*- coding: utf-8 -*-
"""
FFA-Net batch dehazing for a folder of images.

- Loads pretrained FFA (e.g., ots_train_ffa_3_19.pk)
- Uses the paper's normalization (mean=[0.64,0.60,0.58], std=[0.14,0.15,0.152])
- Preserves original file names + extensions (important for YOLO label matching)
- No matplotlib; pure batch write to out folder
- Works on CPU or CUDA (auto)
- PyTorch 2.6+ safe-load compatible (falls back to weights_only=False if trusted)
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# --- import FFA from your repo (models.py or FFA_Net.py) ---
FFA = None
try:
    from FFA_Net import FFA as _FFA
    FFA = _FFA
except Exception:
    try:
        from FFA_Net import FFA as _FFA
        FFA = _FFA
    except Exception:
        try:
            import FFA_Net
            FFA = getattr(FFA_Net, "FFA", None)
        except Exception:
            FFA = None
if FFA is None:
    raise RuntimeError(
        "Could not import class FFA. Ensure your repo provides FFA(gps, blocks) "
        "in either models.py or FFA_Net.py and run this script from that repo."
    )

def list_images(root: Path, recursive: bool = False):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if recursive:
        for p in root.rglob("*"):
            if p.suffix.lower() in exts and p.is_file():
                yield p
    else:
        for p in root.iterdir():
            if p.suffix.lower() in exts and p.is_file():
                yield p

def _clean_state_dict_keys(state: dict) -> dict:
    # strip common prefixes like 'module.' or 'model.'
    def strip_prefix(d, prefix):
        if all(k.startswith(prefix) for k in d.keys()):
            return {k[len(prefix):]: v for k, v in d.items()}
        return d
    state = strip_prefix(state, "module.")
    state = strip_prefix(state, "model.")
    state = strip_prefix(state, "net.")
    return state

def _extract_state_dict(ckpt):
    # Accept plain state_dict, or dicts with typical keys
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net", "ema"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt  # assume it's already a state_dict

def build_net(weights: Path, gps: int, blocks: int, device: torch.device):
    net = FFA(gps=gps, blocks=blocks)

    ckpt = None
    # 1) Try safe load (weights_only=True default on PyTorch 2.6+)
    try:
        ckpt = torch.load(str(weights), map_location=device)
        print("[LOAD] success with default safe loader (weights_only=True).")
    except Exception as e1:
        print(f"[WARN] Safe load failed: {e1}\n[INFO] Trying safe loader with NumPy allowlist...")
        # 2) Add NumPy classes to the allowlist for safe loader
        try:
            import torch.serialization as ts
            from numpy.core.multiarray import scalar as np_scalar
            import numpy as np
            ts.add_safe_globals([np_scalar, np.dtype])
            ckpt = torch.load(str(weights), map_location=device)  # still weights_only=True
            print("[LOAD] success after adding NumPy to safe allowlist (weights_only=True).")
        except Exception as e2:
            print(f"[WARN] Allowlisted safe load still failed: {e2}\n"
                  f"[INFO] Retrying with weights_only=False (ONLY if you trust this file).")
            # 3) Final fallback: unsafe (legacy) loader
            ckpt = torch.load(str(weights), map_location=device, weights_only=False)
            print("[LOAD] success with weights_only=False (unsafe legacy loader).")

    # ----- extract a state_dict no matter how the checkpoint is structured -----
    def _extract_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for key in ("model", "state_dict", "net", "ema"):
                v = ckpt_obj.get(key, None)
                if isinstance(v, dict):
                    return v
        return ckpt_obj

    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Loaded checkpoint is not a state_dict and has no supported keys.")

    # strip common prefixes
    def _strip_prefix(d, prefix):
        return {k[len(prefix):]: v for k, v in d.items()} if all(k.startswith(prefix) for k in d) else d
    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")
    state = _strip_prefix(state, "net.")

    info = net.load_state_dict(state, strict=False)
    if getattr(info, "missing_keys", None):
        print("[LOAD] missing_keys:", len(info.missing_keys))
        for k in info.missing_keys[:10]: print("   -", k)
    if getattr(info, "unexpected_keys", None):
        print("[LOAD] unexpected_keys:", len(info.unexpected_keys))
        for k in info.unexpected_keys[:10]: print("   -", k)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        net = nn.DataParallel(net)
    net.to(device).eval()
    return net

@torch.no_grad()
def dehaze_folder(img_dir: Path, out_dir: Path, weights: Path,
                  gps: int = 3, blocks: int = 19, device_str: str = "auto",
                  recursive: bool = False, half: bool = False,
                  no_norm: bool = False, probe: bool = False):
    # device
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str.startswith("cuda") or (device_str == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # model
    net = build_net(weights, gps, blocks, device)

    # transforms
    if no_norm:
        preprocess = T.ToTensor()
        print("[INFO] Using NO normalization (debug).")
    else:
        preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.64, 0.60, 0.58], std=[0.14, 0.15, 0.152]),
        ])

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(list_images(img_dir, recursive=recursive))
    if not files:
        print(f"[WARN] no images under {img_dir}")
        return

    use_amp = (device.type == "cuda" and half)
    n_ok = 0
    for i, src in enumerate(files, 1):
        # read
        img = Image.open(src).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)  # 1x3xHxW

        # infer
        if use_amp:
            with torch.cuda.amp.autocast():
                pred = net(x)
        else:
            pred = net(x)

        # clamp to [0,1] and save with the SAME name & extension
        y = pred.clamp(0, 1).cpu().squeeze(0)  # 3xHxW
        dst = out_dir / src.name  # preserve filename -> keeps YOLO label alignment
        vutils.save_image(y, str(dst))

        if probe and i <= 10:
            a = np.array(img, dtype=np.int16)
            b = (y.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()).astype(np.int16)
            mad = float(np.mean(np.abs(a - b)))
            print(f"[{i}/{len(files)}] OK: {src.name} | mean |Δpix| = {mad:.2f}")
        else:
            print(f"[{i}/{len(files)}] OK: {src.name}")
        n_ok += 1

    print(f"[DONE] {n_ok}/{len(files)} images -> {out_dir}")

def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--img-dir",  type=str, required=True, help="input images folder")
    ps.add_argument("--out-dir",  type=str, required=True, help="output folder")
    ps.add_argument("--weights",  type=str, required=True, help="ots_train_ffa_3_19.pk (or path)")
    ps.add_argument("--gps",      type=int, default=3)
    ps.add_argument("--blocks",   type=int, default=19)
    ps.add_argument("--device",   type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ps.add_argument("--recursive", action="store_true", help="recurse into subfolders")
    ps.add_argument("--half",     action="store_true", help="use AMP on CUDA for speed")
    ps.add_argument("--no-norm",  action="store_true", help="feed raw ToTensor() without mean/std (debug)")
    ps.add_argument("--probe",    action="store_true", help="print mean |Δpix| for first 10 images (debug)")
    args = ps.parse_args()

    dehaze_folder(
        img_dir=Path(args.img_dir),
        out_dir=Path(args.out_dir),
        weights=Path(args.weights),
        gps=args.gps,
        blocks=args.blocks,
        device_str=args.device,
        recursive=args.recursive,
        half=args.half,
        no_norm=args.no_norm,
        probe=args.probe,
    )

if __name__ == "__main__":
    main()
