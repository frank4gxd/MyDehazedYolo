# -*- coding: utf-8 -*-
"""
DCP (Dark Channel Prior) dehazing with guided filter refinement.
Usage:
  # 单图演示
  python dcp_dehaze.py --input demo.jpg --output out.jpg
  # 批量：把某个文件夹内的图像去雾到输出目录（保留文件名与子目录层级）
  python dcp_dehaze.py --indir "E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-YOLO/images/test" \
                       --outdir "E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-DCP/images/test"

依赖：opencv-python, numpy
可选：安装 opencv-contrib-python 后可用 ximgproc.guidedFilter（本实现已内置 numpy 版 guided filter）
"""
import argparse, os
from pathlib import Path
import numpy as np
import cv2

# -------------------------
# 基础工具
# -------------------------
def to_float(img_bgr):
    return img_bgr.astype(np.float32) / 255.0

def to_uint8(img_float):
    img = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    return img

def dark_channel(im, r=7):
    """
    im: float32, range[0,1], shape HxWx3
    r: window radius (patch size = 2r+1)
    """
    # 像素级最小通道
    min_per_pixel = np.min(im, axis=2)
    # 最小值滤波：用腐蚀实现
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    dark = cv2.erode(min_per_pixel, kernel)
    return dark

def estimate_atmospheric_light(im, dark, top_percent=0.001):
    """
    从暗通道中取 top 百分比(值大的位置)，在原图上选强度最大的像素作为 A（逐通道取RGB）
    im: float32 [0,1], HxWx3
    dark: float32 [0,1], HxW
    """
    H, W = dark.shape
    num = max(1, int(H * W * top_percent))
    flat_idx = np.argsort(dark.reshape(-1))[-num:]  # 暗通道中数值最高的若干像素索引
    # 在原图上找强度最大的那个像素
    im_reshaped = im.reshape(-1, 3)
    candidates = im_reshaped[flat_idx]
    # 按亮度排序（可用RGB和、或最大通道值）
    brightness = candidates.sum(axis=1)
    idx = flat_idx[np.argmax(brightness)]
    A = im_reshaped[idx]
    return A  # shape (3,)

def estimate_transmission(im, A, omega=0.95, r=7):
    """
    t = 1 - omega * dark_channel(I/A)
    """
    # 归一化到 A
    norm = im / (A.reshape(1, 1, 3) + 1e-8)
    t = 1.0 - omega * dark_channel(norm, r=r)
    return np.clip(t, 0.0, 1.0)

def box_filter(img, r):
    """使用 boxFilter 做均值滤波"""
    k = 2 * r + 1
    return cv2.boxFilter(img, -1, (k, k), borderType=cv2.BORDER_REFLECT)

def guided_filter(I, p, r=40, eps=1e-3):
    """
    引导滤波（灰度版）：I 为导向图(灰度, [0,1]), p 为待滤波图(单通道, [0,1])
    返回 q
    参考: K. He, J. Sun, X. Tang, Guided Image Filtering
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    q = mean_a * I + mean_b
    return q

def recover_radiance(im, t, A, t0=0.1):
    """
    J = (I - A) / max(t, t0) + A
    """
    t_expanded = np.maximum(t, t0)[:, :, None]
    J = (im - A.reshape(1, 1, 3)) / t_expanded + A.reshape(1, 1, 3)
    return np.clip(J, 0.0, 1.0)

def dcp_dehaze(img_bgr,
               dark_radius=7,        # 暗通道窗口半径（论文常用 patch size 15 -> r=7）
               omega=0.95,           # 去雾强度系数
               t0=0.1,               # 最小传输
               gf_radius=40,         # 引导滤波半径（像素）
               gf_eps=1e-3,          # 引导滤波正则
               A_top_percent=0.001   # 选 A 的候选比例
               ):
    # 1) 归一化
    I = to_float(img_bgr)[:, :, ::-1]  # BGR->RGB 做内部计算更直观
    I = I[:, :, ::-1]  # 也可直接使用 BGR，不影响数学；这里保持 BGR，注意一致性

    # 如果你想改为 RGB，去掉上两行，注意后续一致。这份实现用 BGR 也没问题。
    I = to_float(img_bgr)  # BGR in [0,1]

    # 2) 暗通道
    dark = dark_channel(I, r=dark_radius)

    # 3) 大气光
    A = estimate_atmospheric_light(I, dark, top_percent=A_top_percent)  # (3,)

    # 4) 粗 t
    t = estimate_transmission(I, A, omega=omega, r=dark_radius)

    # 5) 引导滤波精炼 t（用原图灰度作为导向）
    gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_refined = guided_filter(gray, t, r=gf_radius, eps=gf_eps)
    t_refined = np.clip(t_refined, 0.0, 1.0)

    # 6) 复原
    J = recover_radiance(I, t_refined, A, t0=t0)

    return to_uint8(J), (A, t, t_refined, dark)

# -------------------------
# CLI / 批处理
# -------------------------
def process_one(input_path, output_path, **kwargs):
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, f"read fail: {input_path}"
    out, _ = dcp_dehaze(img, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return True, str(output_path)

def process_dir(indir, outdir, exts=(".jpg", ".jpeg", ".png"), keep_rel=True, **kwargs):
    indir, outdir = Path(indir), Path(outdir)
    n_ok, n_all = 0, 0
    for p in indir.rglob("*"):
        if p.suffix.lower() in exts:
            n_all += 1
            rel = p.relative_to(indir) if keep_rel else p.name
            dst = outdir / rel
            ok, msg = process_one(p, dst, **kwargs)
            n_ok += int(ok)
    return n_ok, n_all

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="single image path")
    ap.add_argument("--output", type=str, help="output image path")
    ap.add_argument("--indir", type=str, help="input folder (batch)")
    ap.add_argument("--outdir", type=str, help="output folder (batch)")
    # 超参
    ap.add_argument("--patch", type=int, default=15, help="patch size for dark channel (odd); radius=r=(patch-1)//2")
    ap.add_argument("--omega", type=float, default=0.95, help="haze removal strength [0,1]")
    ap.add_argument("--t0", type=float, default=0.1, help="minimum transmission")
    ap.add_argument("--gf_r", type=int, default=40, help="guided filter radius")
    ap.add_argument("--gf_eps", type=float, default=1e-3, help="guided filter eps")
    ap.add_argument("--top_percent", type=float, default=0.001, help="top percentage for A estimation (e.g., 0.001 = 0.1%)")
    args = ap.parse_args()

    r = (args.patch - 1) // 2
    params = dict(
        dark_radius=r,
        omega=args.omega,
        t0=args.t0,
        gf_radius=args.gf_r,
        gf_eps=args.gf_eps,
        A_top_percent=args.top_percent
    )

    if args.input and args.output:
        ok, msg = process_one(Path(args.input), Path(args.output), **params)
        print("[OK]" if ok else "[FAIL]", msg)
    elif args.indir and args.outdir:
        n_ok, n_all = process_dir(args.indir, args.outdir, **params)
        print(f"[DONE] {n_ok}/{n_all} images written to {args.outdir}")
    else:
        print("Please specify either --input & --output, or --indir & --outdir.")

if __name__ == "__main__":
    main()
