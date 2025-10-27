# save_clean_ffa.py
import argparse
import torch
import torch.serialization as ts
from numpy.core.multiarray import scalar as np_scalar
import numpy as np

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "net", "ema"):
            v = ckpt.get(k)
            if isinstance(v, dict):
                return v
    return ckpt

def strip_prefixes(state):
    def strip(d, p):
        return {k[len(p):]: v for k, v in d.items()} if all(k.startswith(p) for k in d) else d
    state = strip(state, "module.")
    state = strip(state, "model.")
    state = strip(state, "net.")
    return state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True, help="path to original .pk")
    ap.add_argument("--out", dest="out", required=True, help="path to write plain state_dict .pth")
    args = ap.parse_args()

    # 1) Safe loader with NumPy allow-list
    ts.add_safe_globals([np_scalar, np.dtype])
    try:
        ckpt = torch.load(args.inp, map_location="cpu")  # weights_only=True (default on PyTorch 2.6+)
        print("[LOAD] success with safe loader (weights_only=True + NumPy allowlist).")
    except Exception as e1:
        print(f"[WARN] Safe load failed: {e1}\n[INFO] Retrying with weights_only=False (trusted file).")
        # 2) Final fallback: legacy/unsafe loader (ONLY if you trust the file)
        ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
        print("[LOAD] success with weights_only=False.")

    state = extract_state_dict(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint did not contain a state_dict.")
    state = strip_prefixes(state)

    torch.save(state, args.out)
    print(f"[SAVE] wrote clean state_dict -> {args.out}")

if __name__ == "__main__":
    main()
