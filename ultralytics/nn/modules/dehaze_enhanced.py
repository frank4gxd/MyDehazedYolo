# ultralytics/ultralytics/nn/modules/dehaze_enhanced.py

from __future__ import annotations

import inspect
import os
import warnings
from typing import Any

import torch
import torch.nn as nn

# --- 安全导入 EnhancedDehazeNet：优先 ultralytics.Dehaze，回退顶层 Dehaze ---
_ENHANCED_CLS = None
_ERRS = []
for _imp in ("ultralytics.Dehaze.enhanced_dehaze_model", "Dehaze.enhanced_dehaze_model"):
    try:
        _mod = __import__(_imp, fromlist=["EnhancedDehazeNet"])
        _ENHANCED_CLS = getattr(_mod, "EnhancedDehazeNet")
        break
    except Exception as e:
        _ERRS.append((_imp, repr(e)))

if _ENHANCED_CLS is None:
    lines = ["Cannot import EnhancedDehazeNet from either path:"]
    for path, err in _ERRS:
        lines.append(f"  - {path}: {err}")
    raise ImportError("\n".join(lines))


def _coerce_bool_env(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


class DehazeEnhanced(nn.Module):
    """Drop-in 去雾层 for YOLO*. 期望输入/输出: NCHW, RGB, float in [0, 1]，通道恒等 3→3。.

    YAML 示例（推荐放在 backbone 第一层）:
    - [-1, 1, DehazeEnhanced, {
            "ckpt": "E:/Ivs_FrankGuo/Yolo12_Dehazed/ultralytics/checkpoints_ft_ohaze_enhanced/best_ft.pth",
            "eval_mode": "fused",                # 'direct' | 'physics' | 'fused'
            "freeze_all": true,                  # 去雾分支只做推理（默认 True）
            "strict": true,                      # 严格加载
            "use_dino_backbone": true,
            "dino_name": "convnext_small.dinov3_lvd1689m",
            "dino_freeze": true,
            "base_ch": 64,
            "heads": 4,
            "use_aod_head": true,
            "use_physics_guidance": true,
            "use_edge_enhancement": true,
            "use_channel_attention": true,
            "norm_type": "pono"
    }]

    也可用环境变量覆盖（优先级：YAML > 环境变量 > 默认）: DEHAZE_CKPT, DEHAZE_EVAL_MODE, DEHAZE_BASE_CH, DEHAZE_DINO, DEHAZE_DINO_FREEZE,
    DEHAZE_FREEZE_ALL, DEHAZE_STRICT, DEHAZE_ALLOW_PARTIAL
    """

    def __init__(self, c1: int = 3, c2: int | None = None, *args: Any, **kwargs: Any):
        super().__init__()
        if c2 is None:
            c2 = c1
        assert c1 == 3 and c2 == 3, "DehazeEnhanced expects 3->3 (RGB in/out)."

        # ---------- 读取与剥离本层控制参数 ----------
        self.eval_mode: str = str(kwargs.pop("eval_mode", os.getenv("DEHAZE_EVAL_MODE", "fused"))).lower()
        if self.eval_mode not in {"fused", "direct", "physics"}:
            warnings.warn(f"[DehazeEnhanced] Unknown eval_mode='{self.eval_mode}', fallback to 'fused'.")
            self.eval_mode = "fused"

        freeze_all = bool(kwargs.pop("freeze_all", _coerce_bool_env(os.getenv("DEHAZE_FREEZE_ALL"), True)))
        strict = bool(kwargs.pop("strict", _coerce_bool_env(os.getenv("DEHAZE_STRICT"), True)))
        allow_partial = _coerce_bool_env(os.getenv("DEHAZE_ALLOW_PARTIAL"), False)

        ckpt = kwargs.pop(
            "ckpt",
            os.getenv(
                "DEHAZE_CKPT", "E:/Ivs_FrankGuo/Yolo12_Dehazed/ultralytics/checkpoints_ft_ohaze_enhanced/best_ft.pth"
            ),
        )

        # 一些常用超参可由环境变量覆盖（若 YAML 未显式给）
        kwargs.setdefault("base_ch", int(os.getenv("DEHAZE_BASE_CH", "64")))
        kwargs.setdefault("use_dino_backbone", _coerce_bool_env(os.getenv("DEHAZE_USE_DINO", "1"), True))
        kwargs.setdefault("dino_name", os.getenv("DEHAZE_DINO", "convnext_small.dinov3_lvd1689m"))
        kwargs.setdefault("dino_freeze", _coerce_bool_env(os.getenv("DEHAZE_DINO_FREEZE", "1"), True))

        # ---------- 只把 EnhancedDehazeNet 支持的参数传进去 ----------
        sig = inspect.signature(_ENHANCED_CLS.__init__)
        valid_keys = set(k for k in sig.parameters.keys() if k != "self")
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        # 构建网络
        try:
            self.net = _ENHANCED_CLS(**filtered_kwargs)
        except TypeError as e:
            raise TypeError(
                f"EnhancedDehazeNet(**kwargs) failed.\n"
                f"Allowed keys: {sorted(valid_keys)}\n"
                f"Given keys:   {sorted(list(kwargs.keys()))}"
            ) from e

        # ---------- 加载 ckpt ----------
        if ckpt and os.path.isfile(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            if isinstance(sd, dict):
                for k in ("state_dict", "model", "net"):
                    if k in sd and isinstance(sd[k], dict):
                        sd = sd[k]
                        break
            if not isinstance(sd, dict):
                raise RuntimeError("Unsupported checkpoint format: expecting a (state)dict.")

            if strict:
                self.net.load_state_dict(sd, strict=True)
            else:
                own = self.net.state_dict()
                compat = {k: v for k, v in sd.items() if (k in own and v.shape == own[k].shape)}
                miss = set(own.keys()) - set(compat.keys())
                unexp = set(sd.keys()) - set(compat.keys())
                self.net.load_state_dict({**own, **compat}, strict=False)
                if allow_partial:
                    warnings.warn(
                        f"[DehazeEnhanced] partial load: matched={len(compat)}, "
                        f"missing={len(miss)}, unexpected={len(unexp)}"
                    )
                else:
                    raise RuntimeError(
                        "strict=False but allow_partial=0 and shapes mismatch. "
                        "Either set DEHAZE_ALLOW_PARTIAL=1 or supply a matching checkpoint."
                    )
        else:
            warnings.warn(f"[DehazeEnhanced] ckpt not found or empty: {ckpt}. Start from random init.")

        # ---------- 关键修复：确保整支网络权重为 FP32 ----------
        self._ensure_fp32()

        # ---------- 冻结/解冻 ----------
        if freeze_all:
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()

        self._frozen = freeze_all

    # ---- 保证 net(含 DINO) 为 float32 权重/缓冲 ----
    def _ensure_fp32(self):
        # 把参数与缓冲区统统转成 float32
        self.net.float()
        # 对 DINO Provider 的内部 backbone 再保险
        if hasattr(self.net, "dino"):
            try:
                self.net.dino.float()
                if hasattr(self.net.dino, "backbone"):
                    self.net.dino.backbone.float()
            except Exception:
                pass

    # ---- 若外部 YOLO 在某些阶段把 requires_grad 改了，这里重申冻结状态，并保持 eval ----
    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad = False
            self._ensure_fp32()
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen or not any(p.requires_grad for p in self.net.parameters())

    def _select_output(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        # EnhancedDehazeNet.forward 返回 dict
        if self.eval_mode == "fused" and "fused_dehazed" in out:
            return out["fused_dehazed"]
        if self.eval_mode == "physics" and "physics_dehazed" in out:
            return out["physics_dehazed"]
        # 回退到 direct
        return out.get("dehazed", next(iter(out.values())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在半精度/AMP 下强制用 fp32 前向，保证稳定 + 统一 dtype
        orig_dtype = x.dtype
        need_cast = orig_dtype in (torch.float16, torch.bfloat16)
        xin = x.float() if need_cast else x

        # 如果外部有人把我们权重转成 half/bf16，这里即时拉回 FP32（轻量检测）
        try:
            p0 = next(self.net.parameters())
            if p0.dtype is not torch.float32:
                self._ensure_fp32()
        except StopIteration:
            pass

        # 关闭 autocast，整段 Dehaze 分支在 FP32 下执行
        device_type = "cuda" if xin.is_cuda else ("mps" if xin.device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=device_type, enabled=False):
            if self.frozen:
                with torch.no_grad():
                    out = self.net(xin)
            else:
                out = self.net(xin)

        y = self._select_output(out) if isinstance(out, dict) else out

        if need_cast:
            y = y.to(orig_dtype, non_blocking=True)

        return y.clamp_(0.0, 1.0)

    def extra_repr(self) -> str:
        trainable = any(p.requires_grad for p in self.net.parameters())
        return f"freeze_all={not trainable}, eval_mode='{self.eval_mode}'"
