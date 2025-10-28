# -*- coding: utf-8 -*-
import torch.multiprocessing as mp
from ultralytics import YOLO
from dino_yolo_late_concat import attach_dino_late_concat  # 就导这个函数

def main():
    y = YOLO(r"E:/Ivs_FrankGuo/Yolo12_Dehazed/ultralytics/ultralytics/cfg/models/12/yolo12.yaml")

    # 只做这一步：挂接 Detect 前的最简融合（不改 loss、不改输出）
    attach_dino_late_concat(
        y,
        dino_name="convnext_small.dinov3_lvd1689m",
        use_levels=(True, True, True),
        reduce_ratio=4,
        alpha_init=0.6,
        diag="swap_p3",  # ← 先开：前5个batch把P3替换成DINO派生特征
        diag_batches=5
    )

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
        name="y12n_dino_late_concat_simple",
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
