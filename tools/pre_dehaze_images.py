"""
批量离线去雾（EnhancedDehazeNet 预计算，硬编码为你的训练配置）
- 读取单张图片
- RGB[0,1] -> EnhancedDehazeNet 前向 -> 取 fused/physics/direct
- 写回 BGR uint8 到指定目录（保留子目录结构）.

特性：
- 自动 pad 到 32 倍数，前向后再裁回原图尺寸
- 处理多种 checkpoint 格式（state_dict/model/net）
- 支持“雾量门控”（暗通道阈值）与跳过已存在文件
- 进度打印与异常回退（写回原图避免中断）
- 速度/稳定增强：CUDA TF32、cudnn.benchmark、inference_mode
"""

import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

# --- 导入你的增强版去雾模型 ---
# 确保 enhanced_dehaze_model.py 在 ultralytics/Dehaze/ 下，
# 并可被 `from ultralytics.Dehaze.enhanced_dehaze_model import EnhancedDehazeNet` 找到
try:
    from ultralytics.Dehaze.enhanced_dehaze_model import EnhancedDehazeNet
except Exception as e:
    print("[error] cannot import EnhancedDehazeNet:", e)
    print("        确认文件路径与包结构无误，例如 ultralytics/Dehaze/enhanced_dehaze_model.py")
    sys.exit(1)


# -------------------------
# 工具函数
# -------------------------
def is_image_file(p: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")) -> bool:
    return p.lower().endswith(exts)


def haze_score_dark_channel(img_bgr: np.ndarray) -> float:
    """暗通道均值：值越大→越有雾（0~1）."""
    rgb = img_bgr.astype(np.float32) / 255.0
    dc = rgb.min(axis=2)
    return float(dc.mean())


def pad_to_multiple(x: torch.Tensor, multiple: int = 32, value: float = 0.5):
    """把 NCHW pad 到 multiple 的倍数；返回 (padded, (orig_h, orig_w)) 注意：即使不 pad，也返回真实 (h,w)，避免 (0,0) 导致裁剪空张量。.
    """
    n, c, h, w = x.shape
    nh = (h + multiple - 1) // multiple * multiple
    nw = (w + multiple - 1) // multiple * multiple
    if nh == h and nw == w:
        return x, (h, w)
    y = torch.full((n, c, nh, nw), value, dtype=x.dtype, device=x.device)
    y[..., :h, :w] = x
    return y, (h, w)


def crop_back(y: torch.Tensor, orig_hw):
    """把 pad 后的 NCHW 裁回原始 HxW."""
    if isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 2:
        h, w = orig_hw
    else:
        return y  # 兜底：不裁剪
    return y[..., :h, :w]


def _build_hardcoded_model(device: str):
    """用与你的 YAML 完全一致的配置构建模型： - use_dino_backbone=True - dino_name='convnext_small.dinov3_lvd1689m' - base_ch=64,
    heads=4, norm_type='pono' - AOD/physics/edge/channel attention 全开.
    """
    try:
        net = (
            EnhancedDehazeNet(
                use_dino_backbone=True,
                dino_name="convnext_small.dinov3_lvd1689m",
                dino_freeze=True,
                base_ch=64,
                heads=4,
                norm_type="pono",
                use_edge_enhancement=True,
                use_channel_attention=True,
                use_aod_head=True,
                use_physics_guidance=True,
            )
            .to(device)
            .eval()
        )
        return net
    except ImportError as e:
        # 典型：timm 未安装
        print("[error] 构建 DINO 主干失败：", e)
        print("        需要安装 timm： pip install timm")
        sys.exit(1)


def load_dehaze_model(ckpt_path: str, device: str = "cuda", eval_mode: str = "fused"):
    """返回 (net, run_fn) - net: EnhancedDehazeNet (eval 模式) - run_fn: 接受 NCHW RGB[0,1] 的 Tensor，返回同尺寸 dehazed RGB[0,1].
    """
    net = _build_hardcoded_model(device)

    # ---- 先 strict 加载，若失败再回退到 partial ----
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("state_dict", "model", "net"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {type(sd)}")

        try:
            net.load_state_dict(sd, strict=True)
            print("[info] checkpoint loaded with strict=True")
        except Exception as e:
            print("[error] strict load failed (显示不匹配键列表)：")
            print(e)
            own = net.state_dict()
            compat = {k: v for k, v in sd.items() if (k in own and v.shape == own[k].shape)}
            miss = set(own.keys()) - set(compat.keys())
            unexp = set(sd.keys()) - set(compat.keys())
            print(f"[warn] partial load -> matched={len(compat)} missing={len(miss)} unexpected={len(unexp)}")
            net.load_state_dict({**own, **compat}, strict=False)
    else:
        print(f"[warn] checkpoint not found: {ckpt_path} -> 使用随机初始化（不推荐）")

    # 选择输出张量
    def select_output(out_dict: dict) -> torch.Tensor:
        m = eval_mode.lower().strip()
        if m == "fused" and "fused_dehazed" in out_dict:
            return out_dict["fused_dehazed"]
        if m == "physics" and "physics_dehazed" in out_dict:
            return out_dict["physics_dehazed"]
        # 回退 direct
        return out_dict.get("dehazed", next(iter(out_dict.values())))

    @torch.no_grad()
    def run_rgb01(x_rgb01: torch.Tensor) -> torch.Tensor:
        out = net(x_rgb01)
        y = select_output(out).clamp_(0.0, 1.0)
        return y

    return net, run_rgb01


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def bgr_to_tensor01(im_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
    return t


def tensor01_to_bgr(y: torch.Tensor) -> np.ndarray:
    y = y.squeeze(0).permute(1, 2, 0).clamp_(0, 1).detach().cpu().numpy()
    out = (y * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return bgr


# -------------------------
# 主流程
# -------------------------
def main():
    ap = argparse.ArgumentParser("Precompute Dehazed Images (EnhancedDehazeNet, hard-coded arch)")
    ap.add_argument("--src", required=True, help="源图片根目录或通配符 (例如 E:/.../images/train)")
    ap.add_argument("--dst", required=True, help="输出目录 (例如 E:/.../images/train_dehazed)")
    ap.add_argument("--ckpt", default=os.getenv("DEHAZE_CKPT", ""), help="EnhancedDehazeNet 预训练权重路径")
    ap.add_argument(
        "--device",
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
        choices=["cuda", "cpu", "mps"],
    )
    ap.add_argument(
        "--eval-mode", default=os.getenv("DEHAZE_EVAL_MODE", "fused"), choices=["fused", "physics", "direct"]
    )
    ap.add_argument("--exts", default="jpg,jpeg,png,bmp,webp,tif,tiff", help="扫描的图片后缀，逗号分隔")
    ap.add_argument("--gate-thresh", type=float, default=0.0, help="雾量门控阈值（0=不启用；建议 0.15~0.20）")
    ap.add_argument("--skip-existing", action="store_true", help="若目标文件已存在则跳过")
    ap.add_argument("--print-every", type=int, default=50, help="多少张打印一次进度")
    args = ap.parse_args()

    # 规避路径里混入全角引号
    args.src = str(args.src).replace("“", '"').replace("”", '"')
    args.dst = str(args.dst).replace("“", '"').replace("”", '"')

    # 速度/稳定小优化（CUDA）
    if args.device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # 'high'：允许 TF32，通常比 'highest' 更快，精度对本任务足够
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # 构建文件列表
    if os.path.isdir(args.src):
        root = Path(args.src)
        exts = tuple("." + e.lower() for e in args.exts.split(","))
        files = [str(p) for p in root.rglob("*") if (p.is_file() and is_image_file(str(p), exts))]
    else:
        files = [f for f in glob.glob(args.src, recursive=True) if is_image_file(f)]

    if not files:
        print(f"[error] no images found under: {args.src}")
        sys.exit(1)

    # 加载模型
    print(f"[info] device={args.device}  eval_mode={args.eval_mode}")
    print(f"[info] ckpt={args.ckpt}")
    _, run_model = load_dehaze_model(ckpt_path=args.ckpt, device=args.device, eval_mode=args.eval_mode)

    dst_root = Path(args.dst)
    src_root = Path(args.src) if os.path.isdir(args.src) else None

    n_ok, n_skip, n_err = 0, 0, 0

    # 用 inference_mode() 覆盖整段推理（比逐张 no_grad 稍快且省显存）
    with torch.inference_mode():
        for i, f in enumerate(files, 1):
            try:
                im_bgr = cv2.imread(f, cv2.IMREAD_COLOR)
                if im_bgr is None or im_bgr.size == 0:
                    print(f"[warn] unreadable: {f}")
                    n_err += 1
                    continue

                # 计算输出路径（保持相对目录结构）
                if src_root is not None and os.path.isdir(args.src):
                    rel = os.path.relpath(f, args.src)
                    out_path = dst_root / rel
                else:
                    out_path = dst_root / Path(f).name

                ensure_dir(out_path)

                if args.skip_existing and out_path.exists():
                    n_skip += 1
                    if i % args.print_every == 0:
                        print(f"[{i}/{len(files)}] skip exists -> {out_path}")
                    continue

                # 门控：雾量不足则直接拷贝原图（可提升整体鲁棒性）
                if args.gate_thresh > 0.0:
                    hs = haze_score_dark_channel(im_bgr)
                    if hs < args.gate_thresh:
                        cv2.imwrite(str(out_path), im_bgr)
                        n_ok += 1
                        if i % args.print_every == 0:
                            print(f"[{i}/{len(files)}] gate skip (hs={hs:.3f}) -> {out_path}")
                        continue

                # 准备张量（保持 [0,1]）
                x = bgr_to_tensor01(im_bgr, device=args.device)
                x, orig_hw = pad_to_multiple(x, multiple=32, value=0.5)

                # 前向
                y = run_model(x)
                y = crop_back(y, orig_hw)

                # 输出
                if y.numel() == 0:
                    out_bgr = im_bgr  # 兜底
                    print(f"[warn] empty tensor after crop, fallback to original -> {out_path}")
                else:
                    out_bgr = tensor01_to_bgr(y)

                cv2.imwrite(str(out_path), out_bgr)
                n_ok += 1

                if i % args.print_every == 0:
                    print(f"[{i}/{len(files)}] -> {out_path}")

            except KeyboardInterrupt:
                print("\n[info] interrupted by user")
                break
            except Exception as e:
                n_err += 1
                print(f"[error] file={f}\n{e!r}")
                traceback.print_exc()
                # 兜底写回原图，避免训练集缺图
                try:
                    if im_bgr is not None and im_bgr.size != 0:
                        if src_root is not None and os.path.isdir(args.src):
                            rel = os.path.relpath(f, args.src)
                            out_path = dst_root / rel
                        else:
                            out_path = dst_root / Path(f).name
                        ensure_dir(out_path)
                        cv2.imwrite(str(out_path), im_bgr)
                except Exception:
                    pass

    print(f"\n[done] total={len(files)}  ok={n_ok}  skipped={n_skip}  error={n_err}")
    print(f"[hint] 请在数据集 YAML 中把 train: 改到 {args.dst} 以使用去雾后的训练集。")


if __name__ == "__main__":
    main()
