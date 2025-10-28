# dino_yolo_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# DINOv3 ConvNeXt backbone via timm（冻结，前向无梯度）
# -----------------------------
class DinoConvNeXt(nn.Module):
    def __init__(self, name='convnext_small.dinov3_lvd1689m',
                 pretrained=True, freeze=True, out_indices=(1, 2, 3)):
        super().__init__()
        import timm
        from timm.data import resolve_model_data_config

        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=out_indices
        )
        cfg = resolve_model_data_config(self.backbone)
        mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std  = torch.tensor(cfg.get("std",  (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._frozen = freeze

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        # x: [B,3,H,W]，期望 0~1 浮点
        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        feats = self.backbone(x)  # list of 3 maps, strides ~ [8,16,32]
        return feats


# -----------------------------
# 安全融合：残差 + 零初始化门控 + 层级开关
# -----------------------------
class ConcatReduce(nn.Module):
    """
    out = yolo + sigmoid(alpha) * mixed
    - adapt: 1x1 把 DINO 通道压到更小
    - mix:   (yolo ⊕ adapt(dino)) → 3x3 + BN + SiLU + 3x3 + BN
    - 初始 BN(weight)=0, bias=0，alpha=0  => 初值完全等价纯 YOLO
    """
    def __init__(self, c_yolo: int, c_dino: int, shrink_ratio: int = 4):
        super().__init__()
        c_dino_adapt = max(1, c_yolo // shrink_ratio)
        self.adapt = nn.Conv2d(c_dino, c_dino_adapt, kernel_size=1, bias=False)

        self.mix = nn.Sequential(
            nn.Conv2d(c_yolo + c_dino_adapt, c_yolo, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_yolo),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_yolo, c_yolo, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_yolo),
        )

        self.alpha = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0 → 初值无改动
        self.enable = True

        with torch.no_grad():
            if isinstance(self.mix[-1], nn.BatchNorm2d):
                self.mix[-1].weight.fill_(0.0)
                self.mix[-1].bias.zero_()

    def forward(self, f_yolo: torch.Tensor, f_dino: torch.Tensor):
        if not self.enable:
            return f_yolo
        d = self.adapt(f_dino)
        if d.shape[2:] != f_yolo.shape[2:]:
            d = F.interpolate(d, size=f_yolo.shape[2:], mode='bilinear', align_corners=False)
        mixed = self.mix(torch.cat([f_yolo, d], dim=1))
        gate = torch.sigmoid(self.alpha)
        # 关键：残差形式，不再 (mixed - yolo)
        return f_yolo + gate * mixed


# -----------------------------
# YOLO12 外壳：在 Detect 前用 pre-hook 融合（不改 loss、不改 Detect 输出）
# -----------------------------
class YOLO12WithDINO(nn.Module):
    def __init__(self, core_yolo: nn.Module,
                 dino_name='convnext_small.dinov3_lvd1689m',
                 dino_pretrained=True, dino_freeze=True,
                 dino_out_indices=(1, 2, 3),
                 imgsz_probe=512,
                 reduce_ratio=4,
                 use_levels=(True, True, True)):
        super().__init__()
        self.core = core_yolo
        self.dino = DinoConvNeXt(dino_name, dino_pretrained, dino_freeze, dino_out_indices)

        # 透传元属性（Trainer 会用到）
        for k in ("yaml", "names", "nc", "stride", "args"):
            if hasattr(core_yolo, k):
                setattr(self, k, getattr(core_yolo, k))

        self.detect = self._find_detect(self.core)
        if self.detect is None:
            raise RuntimeError("Detect module not found in YOLO12 model.")

        # 探测 YOLO 的 P3/P4/P5 通道数（用一次 dummy 前向 + pre-hook）
        yolo_ch = self._probe_yolo_channels(imgsz_probe)

        # 探测 DINO 通道
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz_probe, imgsz_probe, device=self._device())
            d_feats = self.dino(dummy)
            dino_ch = [f.shape[1] for f in d_feats]

        # 三层融合器
        self.use_levels = list(use_levels)
        self.fuse = nn.ModuleList([
            ConcatReduce(yc, dc, shrink_ratio=reduce_ratio) for yc, dc in zip(yolo_ch, dino_ch)
        ])

        self.enable_fusion = True
        self.debug_dump = False
        self._last_debug_taps = None

    # 先走父类，再回退 core
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self.__dict__.get("core", None)
            if core is not None and hasattr(core, name):
                return getattr(core, name)
            raise

    # ---------- helpers ----------
    def _device(self):
        for p in self.core.parameters(recurse=True):
            return p.device
        return torch.device('cpu')

    def _find_detect(self, model: nn.Module):
        # 寻找 Detect（适配不同封装）
        mods = []
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            mods = list(model.model.model)
        elif hasattr(model, 'model'):
            mods = list(model.model.children())
        else:
            mods = list(model.children())
        for m in mods[::-1]:
            if m.__class__.__name__.lower() == 'detect':
                return m
        last = mods[-1] if len(mods) else None
        if last is not None and last.__class__.__name__.lower() == 'detect':
            return last
        return None

    def _probe_yolo_channels(self, imgsz: int):
        holder = {}

        def _probe_hook(mod, inputs):
            feats = inputs[0]                  # Detect 输入是 [P3,P4,P5]
            holder['yolo_ch'] = [f.shape[1] for f in feats]
            return None

        h = self.detect.register_forward_pre_hook(_probe_hook)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=self._device())
            _ = self.core(dummy)               # 触发一次 pre-hook
        h.remove()

        if 'yolo_ch' not in holder or len(holder['yolo_ch']) != 3:
            raise RuntimeError("Failed to probe YOLO12 P3/P4/P5 channels.")
        return holder['yolo_ch']

    # ---------- forward ----------
    def forward(self, x):
        # 1) 取出 imgs 并归一化到 0~1（兼容 batch=dict）
        if isinstance(x, dict):
            imgs = x.get('img', None)
            if imgs is None:
                raise ValueError("Batch dict missing key 'img'")
        else:
            imgs = x
        if imgs.dtype != torch.float32:
            imgs = imgs.float()
        if imgs.max() > 1.5:
            imgs = imgs / 255.0

        # 2) 先跑 DINO（冻结、无梯度）
        dino_feats = self.dino(imgs)  # list of 3

        # 3) Detect 前注册一次性 pre-hook 做融合
        def _fusion_hook(mod, inputs):
            feats = list(inputs[0])  # [P3,P4,P5]
            fused = []
            for i in range(3):
                if self.enable_fusion and self.use_levels[i]:
                    fused.append(self.fuse[i](feats[i], dino_feats[i]))
                else:
                    fused.append(feats[i])

            if self.debug_dump:
                self._last_debug_taps = {
                    "yolo":  [t.detach().float().cpu() for t in feats],
                    "dino":  [t.detach().float().cpu() for t in dino_feats],
                    "fused": [t.detach().float().cpu() for t in fused],
                }
            return (fused,)   # 替换 Detect 的输入

        h = self.detect.register_forward_pre_hook(_fusion_hook)
        try:
            out = self.core(x)  # 仍把原始 x（dict/Tensor）交给核心 YOLO
        finally:
            h.remove()
        return out
