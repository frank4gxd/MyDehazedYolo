"""
AOD-Net batch dehazing via OpenCV DNN (no caffe).

What this does:
- Sanitizes training-only layers (EuclideanLoss/Accuracy/etc.)
- Removes any `input: "label"` and its dims
- Normalizes the header to a single Input layer (1,3,H,W) by default,
  or legacy `input + 4x input_dim` if you set --header legacy
- Final cleanup to ensure no stray top-level input_dim/input_shape remain

Usage (PowerShell):
  python aod_batch.py `
    --img-dir "E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-YOLO/images/test" `
    --out-dir "E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-AOD/images/test" `
    --model   "./AOD_Net.caffemodel" `
    --template "./test_template.prototxt" `
    --deploy  "./deployT_clean.prototxt" `
    --gpu 0 --out-blob sum

Tip: If your final blob isn't named 'sum', set --out-blob to the actual top name
(or omit --out-blob to take the last layer).
"""

import argparse
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

UNSUPPORTED = {"EuclideanLoss", "SoftmaxWithLoss", "SigmoidCrossEntropyLoss", "Accuracy", "HingeLoss"}


def strip_loss_and_label_blocks(ptxt: str) -> str:
    # Remove input: "label" (and optional dims) at top-level (CRLF/indent safe)
    ptxt = re.sub(r'(?ms)^\s*input\s*:\s*"label"\s*(?:\r?\n\s*input_dim\s*:\s*\d+\s*){0,4}', "", ptxt)

    # Remove TRAIN-only includes
    ptxt = re.sub(r"(?ms)include\s*\{\s*phase\s*:\s*TRAIN\s*\}", "", ptxt)

    # Remove loss/accuracy layers entirely; also strip label tops/bottoms in other blocks
    out, i, n = [], 0, len(ptxt)
    while i < n:
        j = ptxt.find("layer {", i)
        if j == -1:
            out.append(ptxt[i:])
            break
        out.append(ptxt[i:j])
        # find matching brace for this layer
        k, depth = j, 0
        while k < n:
            if ptxt[k] == "{":
                depth += 1
            elif ptxt[k] == "}":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        block = ptxt[j:k]
        m = re.search(r'type\s*:\s*"(.*?)"', block)
        typ = m.group(1).strip() if m else None
        if typ in UNSUPPORTED:
            # drop the whole loss/accuracy layer
            pass
        else:
            # remove any label tops/bottoms inside non-loss blocks
            block = re.sub(r'(?m)^\s*(top|bottom)\s*:\s*"label"\s*$', "", block)
            out.append(block)
        i = k
    return "".join(out)


def normalize_input_header(ptxt: str, H: int, W: int, mode: str = "input") -> str:
    """Mode = "input"  -> inject clean Input layer; zero top-level input/input_dim/input_shape mode = "legacy" -> inject
    legacy top-level input + 4x input_dim; zero Input layers.
    """
    # Remove any top-level input headers (CRLF-safe)
    ptxt = re.sub(r'(?m)^\s*input\s*:\s*".*?"\s*$', "", ptxt)
    ptxt = re.sub(r"(?m)^\s*input_dim\s*:\s*\d+\s*$", "", ptxt)
    ptxt = re.sub(r"(?ms)^\s*input_shape\s*\{.*?\}\s*$", "", ptxt)

    # Remove any existing Input layers entirely
    out, i, n = [], 0, len(ptxt)
    while i < n:
        j = ptxt.find("layer {", i)
        if j == -1:
            out.append(ptxt[i:])
            break
        out.append(ptxt[i:j])
        k, depth = j, 0
        while k < n:
            if ptxt[k] == "{":
                depth += 1
            elif ptxt[k] == "}":
                depth -= 1
                if depth == 0:
                    k += 1
                    break
            k += 1
        block = ptxt[j:k]
        m = re.search(r'type\s*:\s*"(.*?)"', block)
        is_input = bool(m and m.group(1).strip().lower() == "input")
        if not is_input:
            out.append(block)
        i = k
    body = "".join(out)

    if mode == "legacy":
        # Legacy header at the top, NO Input layer
        header = f'input: "data"\ninput_dim: 1\ninput_dim: 3\ninput_dim: {H}\ninput_dim: {W}\n'
        final = header + body
        return final

    # Default: clean Input layer inserted after network name if present
    mname = re.search(r'(^|\n)\s*name\s*:\s*".*?"\s*\n', body)
    insert_at = mname.end() if mname else 0
    input_layer = (
        "layer {\n"
        '  name: "data"\n'
        '  type: "Input"\n'
        '  top: "data"\n'
        "  input_param {\n"
        f"    shape {{ dim: 1 dim: 3 dim: {H} dim: {W} }}\n"
        "  }\n"
        "}\n"
    )
    final = body[:insert_at] + input_layer + body[insert_at:]

    # Safety pass: when using Input-mode, make sure no stray top-level input_dim/input_shape remain
    final = re.sub(r"(?m)^\s*input_dim\s*:\s*\d+\s*$", "", final)
    final = re.sub(r"(?ms)^\s*input_shape\s*\{.*?\}\s*$", "", final)
    return final


def render_sanitized_deploy(template_file: Path, deploy_file: Path, H: int, W: int, header_mode: str):
    tpl = template_file.read_text(encoding="utf-8", errors="ignore")
    rendered = tpl.format(height=H, width=W)  # assume {height}/{width} in template
    cleaned = strip_loss_and_label_blocks(rendered)
    cleaned = normalize_input_header(cleaned, H, W, mode=header_mode)
    deploy_file.write_text(cleaned, encoding="utf-8")
    # quick debug counts
    num_input = len(re.findall(r'type\s*:\s*"Input"', cleaned))
    num_input_dim = len(re.findall(r"(?m)^\s*input_dim\s*:", cleaned))
    num_input_shape = len(re.findall(r"(?m)^\s*input_shape\s*{", cleaned))
    print(
        f"[PROTO] Wrote {deploy_file} (H={H}, W={W}, header={header_mode}) | Input layers={num_input}, input_dim lines={num_input_dim}, input_shape blocks={num_input_shape}"
    )


def list_images(img_dir: Path):
    for name in os.listdir(img_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            yield img_dir / name


def make_net(deploy: Path, model: Path, gpu_id: int):
    net = cv2.dnn.readNetFromCaffe(str(deploy), str(model))
    if gpu_id is not None and gpu_id >= 0:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print(f"[DNN] CUDA backend (device={gpu_id})")
        except cv2.error as e:
            print(f"[DNN] CUDA unavailable ({e}); falling back to CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[DNN] CPU backend")
    return net


def forward_one(net, bgr, H, W, out_blob: str):
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1 / 255.0, size=(W, H), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        out = net.forward(out_blob)  # NCHW
    except cv2.error:
        out = net.forward()  # fallback to last layer
    out = out[0]  # CHW
    out = np.transpose(out, (1, 2, 0))  # HWC, RGB in [0,1]
    out_bgr = np.clip(out[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return out_bgr


def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--img-dir", type=str, default="../data/img")
    ps.add_argument("--out-dir", type=str, default="../data/result")
    ps.add_argument("--model", type=str, default="../AOD_Net.caffemodel")
    ps.add_argument("--template", type=str, default="test_template.prototxt")
    ps.add_argument("--deploy", type=str, default="deployT_clean.prototxt")
    ps.add_argument("--gpu", type=int, default=0, help="GPU id; -1 for CPU")
    ps.add_argument("--out-blob", type=str, default="sum")
    ps.add_argument(
        "--header",
        type=str,
        default="input",
        choices=["input", "legacy"],
        help='Header style: "input" (Input layer) or "legacy" (input + input_dim)',
    )
    args = ps.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model)
    template = Path(args.template)
    deploy = Path(args.deploy)
    gpu_id = None if args.gpu is None or int(args.gpu) < 0 else int(args.gpu)

    files = [p for p in list_images(img_dir) if p.is_file()]
    if not files:
        print(f"[WARN] no images under {img_dir}")
        return

    sample = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise SystemExit(f"read fail: {files[0]}")
    H, W = sample.shape[:2]

    # Render → Sanitize → Normalize
    render_sanitized_deploy(template, deploy, H, W, header_mode=args.header)

    # Load net
    net = make_net(deploy, model_path, gpu_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, t0 = 0, time.time()
    for p in files:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[SKIP] read fail: {p}")
            continue
        try:
            out_bgr = forward_one(net, bgr, H, W, args.out_blob)
            save_path = out_dir / f"{p.stem}_AOD-Net.jpg"
            cv2.imwrite(str(save_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"[OK] {p.stem}")
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {p}: {e}")

    dt = time.time() - t0
    print(f"[DONE] {n_ok}/{len(files)} images -> {out_dir} in {dt:.2f}s")


if __name__ == "__main__":
    main()
