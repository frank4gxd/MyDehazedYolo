# ultralytics/ultralytics/nn/modules/dino_fuse.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoCache(nn.Module):
    """在图最前面放一个 DinoCache： - 前向：规范化 -> DINO 提取多尺度特征 -> 缓存到类变量 -> 返回原 x（不改动数据流）.
    """

    LAST = None  # class-level 缓存 (P3,P4,P5)

    def __init__(self, name="convnext_small.dinov3_lvd1689m", out_indices=(1, 2, 3), freeze=True, pretrained=True):
        super().__init__()
        import timm
        from timm.data import resolve_model_data_config

        self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        cfg = resolve_model_data_config(self.backbone)
        mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x_norm = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        if self.freeze:
            with torch.no_grad():
                feats = self.backbone(x_norm)
        else:
            feats = self.backbone(x_norm)
        DinoCache.LAST = tuple(feats)  # (stride/8, /16, /32)
        return x  # 不改变主干输入


class DinoFuse(nn.Module):
    """单输入融合层（from: -1）： out = y + sigmoid(alpha) * Mix([y, Adapt(D)]) - level: 0/1/2 -> 对应 DINO 的 (P3/P4/P5) -
    shrink_ratio: 通道压缩比例，越大越省算 - init_p: 门控初值（概率视角），默认 0.1 保守可学 - bn_init: 最后一层 BN 的缩放初值，默认 0.1，避免死分支.
    """

    def __init__(self, level=0, shrink_ratio=4, init_p=0.1, bn_init=0.1):
        super().__init__()
        self.level = int(level)
        self.shrink_ratio = int(shrink_ratio)
        self.init_p = float(init_p)
        self.bn_init = float(bn_init)

        # 延迟构建
        self.adapt = None
        self.mix = None
        self.alpha_raw = None
        self._built = False

    def _build(self, c_yolo: int, c_dino: int, device):
        c_adapt = max(1, c_yolo // self.shrink_ratio)
        self.adapt = nn.Conv2d(c_dino, c_adapt, kernel_size=1, bias=False).to(device)

        self.mix = nn.Sequential(
            nn.Conv2d(c_yolo + c_adapt, c_yolo, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_yolo),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_yolo, c_yolo, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_yolo),
        ).to(device)

        with torch.no_grad():
            # 不是 0，给梯度留路；太大会抖，0.1 较稳
            self.mix[-1].weight.fill_(self.bn_init)
            self.mix[-1].bias.zero_()

        # 门控按“概率初值”初始化（sigmoid(alpha_raw) ≈ init_p）
        alpha_init = math.log(self.init_p / (1.0 - self.init_p))
        self.alpha_raw = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32, device=device))

        self._built = True

    def forward(self, y: torch.Tensor):
        assert DinoCache.LAST is not None, (
            "DinoFuse: DinoCache has not run. Make sure DinoCache is placed at the top of the model."
        )

        d = DinoCache.LAST[self.level]
        if not self._built:
            self._build(c_yolo=y.shape[1], c_dino=d.shape[1], device=y.device)

        d = self.adapt(d)
        if d.shape[2:] != y.shape[2:]:
            d = F.interpolate(d, size=y.shape[2:], mode="bilinear", align_corners=False)

        mixed = self.mix(torch.cat([y, d], dim=1))
        gate = torch.sigmoid(self.alpha_raw)  # ∈ (0,1)
        return y + gate * mixed  # 残差，稳定
