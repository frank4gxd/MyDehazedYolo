# train_dino_yolo_fusion.py
import os
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torchvision.utils as vutils
from PIL import Image

from ultralytics import YOLO
from ultralytics.Dehaze.dino_yolo_fusion import YOLO12WithDINO

_dumped_epochs = set()


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
    # ✅ 注意：Ultralytics 只会调用 on_train_batch_start(trainer)
    def on_train_batch_start(trainer):
        try:
            epoch = int(getattr(trainer, "epoch", 0))
            total_epochs = int(getattr(trainer, "epochs", 0))
            batch_i = int(getattr(trainer, "batch_i", 0))
            # 仅在 第0个batch 且 每50个epoch一次，以及最后一个epoch
            is_checkpoint_epoch = (epoch % 50 == 0) or (epoch == total_epochs - 1)
            if not is_checkpoint_epoch or batch_i != 0 or epoch in _dumped_epochs:
                return

            # 取当前 batch（Ultralytics 会把 batch 放在 trainer.batch 里）
            batch = getattr(trainer, "batch", None)
            if batch is None or len(batch) == 0:
                return
            imgs = batch[0].to(trainer.device, non_blocking=True).float() / 255.0

            run_dir = getattr(trainer, "save_dir", "runs/debug")
            os.makedirs(run_dir, exist_ok=True)

            # 触发一次前向，抓取 dino/fused 特征
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
                for lvl, feat in enumerate(taps[name]):  # list of 3 maps
                    fm = feat[0].cpu()  # 取第1张图的 [C,H,W]
                    _save_heatmap(fm, os.path.join(run_dir, f"{name}_P{3 + lvl}_e{epoch:03d}_mean.png"))
                    _save_grid(fm, os.path.join(run_dir, f"{name}_P{3 + lvl}_e{epoch:03d}_grid.png"))

            _dumped_epochs.add(epoch)
        except Exception as e:
            print(f"[error] dump failed at epoch {getattr(trainer, 'epoch', '?')}: {e}")

    return on_train_batch_start


def main():
    # 1) 构建与 baseline 对齐的 YOLO12（用 yaml）
    y = YOLO(r"E:/Ivs_FrankGuo/Yolo12_Dehazed/ultralytics/ultralytics/cfg/models/12/yolo12.yaml")

    # 2) 注入 DINO 融合（默认 P3/P4；DINO 冻结）
    core = y.model
    y.model = YOLO12WithDINO(
        core_yolo=core,
        dino_name="convnext_small.dinov3_lvd1689m",
        dino_pretrained=True,  # 用 DINO 公开预训练
        dino_freeze=True,  # 不解冻 DINO
        dino_out_indices=(1, 2, 3),
        imgsz_probe=512,
        reduce_ratio=4,
        use_levels=(True, True, False),  # 只融合 P3/P4 控延迟
    ).to("cuda")

    # （可选）做一次 debug 前向，确认尺寸/通道
    y.model.debug_dump = True
    _ = y.model(torch.zeros(1, 3, 512, 512, device="cuda"))
    y.model.debug_dump = False

    # 3) 注册“每 50 epoch + 最后一轮导出特征”的回调
    y.add_callback("on_train_batch_start", make_dump_callback())

    # 4) 训练（与 baseline 对齐；注意不要再传 model=...）
    y.train(
        data=r"E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-YOLO/RTTS.yaml",
        imgsz=512,
        epochs=200,
        batch=8,
        device=0,
        workers=4,  # Windows 下 OK（已做 main-guard）
        seed=405,
        deterministic=True,
        project="runs/rtts_fusion",
        name="y12n_dino_hook_p3p4_frz3_2",
        # pretrained=True 是 Ultralytics 的默认行为，保持和 baseline 一致即可
    )


if __name__ == "__main__":
    mp.freeze_support()  # Windows 必备
    # mp.set_start_method("spawn", force=True)  # 可选：显式指定 spawn
    main()
