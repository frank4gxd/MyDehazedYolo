# ultralytics/ultralytics/Dehaze/train_dino_yolo_concat.py
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torchvision.utils as vutils
from dino_yolo_concat import YOLO12WithDINO_Concat
from PIL import Image

from ultralytics import YOLO

_DUMPED = set()

import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 部分三方库在 Windows + 多进程下会卡死或崩 worker，统一限制 CPU 线程
try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# OpenCV 多线程在 DataLoader 多进程里常见争用
try:
    import cv2

    cv2.setNumThreads(0)
except Exception:
    pass

# 噪声告警（无害），可静音
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

# Windows 多进程关键：spawn + freeze_support
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _norm01(x):
    x = x - x.min()
    d = x.max().clamp_min(1e-6)
    return x / d


def _save_heatmap(chw, path_png):
    m = _norm01(chw.mean(0)).cpu()
    Image.fromarray((m * 255).byte().numpy()).save(path_png)


def _save_grid(chw, path_png, k=16, nrow=8):
    k = min(k, chw.shape[0])
    grid = vutils.make_grid(chw[:k].unsqueeze(1), nrow=nrow, normalize=True, scale_each=True).squeeze(0).cpu()
    Image.fromarray((grid * 255).byte().numpy()).save(path_png)


def make_dump_callback():
    # Ultralytics 调用签名：on_train_batch_start(trainer)
    def on_train_batch_start(trainer):
        try:
            epoch = int(getattr(trainer, "epoch", 0))
            total_epochs = int(getattr(trainer, "epochs", 0))
            batch_i = int(getattr(trainer, "batch_i", 0))

            is_checkpoint_epoch = (epoch % 50 == 0) or (epoch == total_epochs - 1)
            if not is_checkpoint_epoch or batch_i != 0 or epoch in _DUMPED:
                return

            batch = getattr(trainer, "batch", None)
            if batch is None or len(batch) == 0:
                return
            imgs = batch[0].to(trainer.device, non_blocking=True).float() / 255.0

            run_dir = getattr(trainer, "save_dir", "runs/debug")
            os.makedirs(run_dir, exist_ok=True)

            # 触发一次前向，抓取 DINO/Fused 特征
            was_training = trainer.model.training
            trainer.model.debug_dump = True
            trainer.model.eval()
            with torch.no_grad():
                _ = trainer.model(imgs)
            trainer.model.debug_dump = False
            if was_training:
                trainer.model.train()

            taps = getattr(trainer.model, "_last_debug_taps", None)
            if not taps:
                print(f"[warn] epoch {epoch}: no debug taps captured")
                return

            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dump_pt = os.path.join(run_dir, f"feats_e{epoch:03d}_b0_{stamp}.pt")
            torch.save(taps, dump_pt)
            print(f"[ok] saved tensor dump: {dump_pt}")

            # 保存 PNG 预览
            for name in ["dino", "fused"]:
                for lvl, feat in enumerate(taps[name]):  # list of maps
                    fm = feat[0].cpu()  # 第1张图 [C,H,W]
                    base = f"{name}_P{3 + lvl}_e{epoch:03d}"
                    _save_heatmap(fm, os.path.join(run_dir, f"{base}_mean.png"))
                    _save_grid(fm, os.path.join(run_dir, f"{base}_grid.png"))

            _DUMPED.add(epoch)

        except Exception as e:
            print(f"[error] dump failed at epoch {getattr(trainer, 'epoch', '?')}: {e}")

    return on_train_batch_start


def main():
    # 1) baseline YOLO12（与原始训练保持一致）
    y = YOLO(r"E:/Ivs_FrankGuo/Yolo12_Dehazed/ultralytics/ultralytics/cfg/models/12/yolo12.yaml")

    # 2) 注入 DINO concat 融合（默认只融合 P3/P4；DINO 冻结）
    core = y.model
    y.model = YOLO12WithDINO_Concat(
        core_yolo=core,
        dino_name="convnext_small.dinov3_lvd1689m",
        dino_pretrained=True,
        dino_out_indices=(1, 2, 3),
        use_levels=(True, True, False),  # 只融合 P3/P4 控制延迟
        reduce_ratio=8,
        dino_tune="frozen",
        imgsz_probe=512,
    ).to("cuda")

    # （可选）shape sanity check
    y.model.debug_dump = True
    _ = y.model(torch.zeros(1, 3, 512, 512, device="cuda"))
    y.model.debug_dump = False

    # 3) 注册每 50 epoch dump 的回调
    y.add_callback("on_train_batch_start", make_dump_callback())

    # 4) 训练（参数与 baseline 对齐）
    y.train(
        data=r"E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-YOLO/RTTS.yaml",
        imgsz=512,
        epochs=200,
        batch=8,
        device=0,
        workers=4,
        seed=405,
        deterministic=True,
        project="runs/rtts_fusion",
        name="y12n_dino_concat_frozen",
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
