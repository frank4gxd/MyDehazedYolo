# dino_yolo_late_concat.py
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Tuple
try:
    import timm
except Exception:
    timm = None

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

class _DinoLateConcatHook(nn.Module):
    def __init__(self, dino_name: str, reduce_ratio: int, alpha_init: float,
                 use_levels: Tuple[bool,bool,bool], diag: str, diag_batches: int):
        super().__init__()
        assert timm is not None, "pip install timm"
        self.dino = timm.create_model(dino_name, pretrained=True, features_only=True, out_indices=(1,2,3))
        for p in self.dino.parameters():  # 默认冻结 DINO，稳定
            p.requires_grad = False

        self.use_levels = use_levels
        self.reduce_ratio = max(2, reduce_ratio)
        self.alpha_init = float(alpha_init)
        self.diag = diag  # 'none' | 'zero_p3' | 'swap_p3' | 'noise_p3'
        self.diag_batches = int(diag_batches)
        self.batches_seen = 0

        # 运行时构建（等拿到 yolo feats 和 dino feats 的通道数再建头）
        self.proj = None
        self.head = None
        self.alpha = None

        self.latest_imgs = None  # 由“首层 pre-hook”填入

    # ---- 首层：抓取原始图像 ----
    @torch.no_grad()
    def capture_input_hook(self, module, inputs):
        x = inputs[0]
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        # 保存到本模块，Detect 前再用
        self.latest_imgs = x.detach()

    # ---- Detect 前：做融合并替换输入 ----
    def detect_pre_hook(self, module, inputs):
        feats = list(inputs[0])  # [P3,P4,P5]
        if self.latest_imgs is None:
            # 没拿到图像，就不改动（不应发生）
            return inputs

        imgs = self.latest_imgs.to(feats[0].device, non_blocking=True)
        mean = _IMAGENET_MEAN.to(imgs.device); std = _IMAGENET_STD.to(imgs.device)
        imgs_norm = (imgs - mean) / std

        with torch.no_grad():
            dino_feats = self.dino(imgs_norm)  # list of 3

        # 首次调用时构建轻量头
        if self.proj is None:
            self.proj = nn.ModuleList()
            self.head = nn.ModuleList()
            self.alpha = nn.ParameterList()
            for li in range(3):
                cin_y = feats[li].shape[1]
                cin_d = dino_feats[li].shape[1]
                red = min(cin_y, max(16, cin_d // self.reduce_ratio))
                self.proj.append(nn.Conv2d(cin_d, red, kernel_size=1, bias=False))
                # Depthwise 3x3 + Pointwise 1x1（很轻但比纯1x1有感）
                self.head.append(nn.Sequential(
                    nn.Conv2d(cin_y + red, cin_y + red, kernel_size=3, padding=1,
                              groups=cin_y + red, bias=False),
                    nn.Conv2d(cin_y + red, cin_y, kernel_size=1, bias=True),
                ))
                a = torch.full((1,), self.alpha_init)
                self.alpha.append(nn.Parameter(a))
            # 把子模块挂到 Detect 上，确保进优化器
            module.add_module("_dino_proj", self.proj)
            module.add_module("_dino_head", self.head)
            module.add_module("_dino_alpha", self.alpha)

        fused, deltas = [], []
        for li in range(3):
            y = feats[li]
            if not self.use_levels[li]:
                fused.append(y); deltas.append(torch.tensor(0., device=y.device))
                continue
            d = self.proj[li](dino_feats[li])
            d = F.interpolate(d, size=y.shape[-2:], mode="bilinear", align_corners=False)
            # 轻量标准化，避免尺度不匹配
            d = d / (d.std(dim=(2,3), keepdim=True) + 1e-5)

            z = torch.cat([y, d], dim=1)
            delta = self.head[li](z)
            a = torch.sigmoid(self.alpha[li])
            y_hat = y + a * delta
            fused.append(y_hat)
            deltas.append(delta.abs().mean())

        # --- 诊断模式：强制改 P3，验证 hook 确实生效 ---
        if self.diag != "none" and self.batches_seen < self.diag_batches:
            if self.diag == "zero_p3":
                fused[0] = torch.zeros_like(fused[0])
            elif self.diag == "swap_p3":
                fused[0] = F.interpolate(self.proj[0](dino_feats[0]),
                                         size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
            elif self.diag == "noise_p3":
                fused[0] = fused[0] + torch.randn_like(fused[0]) * 0.1
            print(f"[DINO-HOOK] diag {self.diag} active (batch {self.batches_seen+1}/{self.diag_batches})")
        # 打点：前几批打印 α 和 Δ
        if self.batches_seen < 3:
            a_vals = [round(float(torch.sigmoid(a).item()),3) for a in self.alpha]
            d_vals = [round(float(x.item()),4) for x in deltas]
            print(f"[DINO-HOOK] alpha={a_vals}  Δ|P3,P4,P5|={d_vals}")
        self.batches_seen += 1

        # 替换 Detect 的输入：必须返回新的输入tuple
        return (fused,)

def attach_dino_late_concat(yolo_obj,
                            dino_name: str = "convnext_small.dinov3_lvd1689m",
                            use_levels: Tuple[bool,bool,bool]=(True,True,True),
                            reduce_ratio: int = 4,
                            alpha_init: float = 0.6,
                            diag: str = "none",     # 'none' | 'zero_p3' | 'swap_p3' | 'noise_p3'
                            diag_batches: int = 5):
    """在 Detect 前插入 DINO→YOLO 融合；optimizer 会包含新增参数。"""
    mlist: nn.ModuleList = yolo_obj.model.model
    # 找第一层（抓 imgs）与 Detect（做融合）
    first = mlist[0]
    detect = None
    for m in mlist:
        if "detect" in m.__class__.__name__.lower():
            detect = m; break
    assert detect is not None, "Detect module not found."

    hook = _DinoLateConcatHook(dino_name, reduce_ratio, alpha_init, use_levels, diag, diag_batches)
    # 注册两个 pre-hook（顺序无所谓）
    first.register_forward_pre_hook(hook.capture_input_hook, with_kwargs=False)
    detect.register_forward_pre_hook(hook.detect_pre_hook, with_kwargs=False)
    print("[hook] DINO late-concat attached (two-stage pre-hooks).")
