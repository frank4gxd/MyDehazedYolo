# ultralytics/ultralytics/Dehaze/dino_yolo_concat.py
import torch
import torch.nn.functional as F
from torch import nn

try:
    import timm
except ImportError as e:
    raise ImportError("This module requires 'timm'. Please: pip install timm") from e


class YOLO12WithDINO_Concat(nn.Module):
    """
    将 DINO 特征与 YOLO P3/P4/P5 逐层 concat 融合：
      d_red = Conv1x1(dino)  (降维到 C_yolo/reduce_ratio)
      fused = Conv1x1( concat(yolo, d_red) ) -> C_yolo
    参数：
      - use_levels: (P3,P4,P5) 各层是否融合
      - reduce_ratio: DINO 通道先降到 C_yolo/reduce_ratio
      - dino_tune: 'frozen' | 'trainable' | 'partial'
    """

    def __init__(
        self,
        core_yolo: nn.Module,
        dino_name: str = "convnext_small.dinov3_lvd1689m",
        dino_pretrained: bool = True,
        dino_out_indices=(1, 2, 3),
        use_levels=(True, True, True),
        reduce_ratio: int = 8,
        dino_tune: str = "frozen",
        imgsz_probe: int = 512,
    ):
        super().__init__()
        # 先注册子模块（很重要，保证 nn.Module 能在 _modules 里找到它）
        self.core_yolo = core_yolo

        self.use_levels = list(use_levels)
        self.reduce_ratio = int(reduce_ratio)
        self.debug_dump = False
        self._last_debug_taps = None

        # 透传 Trainer 可能访问的关键属性
        for attr in ("nc", "names", "yaml", "inplace", "stride", "args", "cfg"):
            if hasattr(core_yolo, attr):
                setattr(self, attr, getattr(core_yolo, attr))

        # 1) 构建 DINO 并设置训练/冻结策略
        self.dino = timm.create_model(
            dino_name, pretrained=dino_pretrained, features_only=True, out_indices=dino_out_indices
        )
        self._apply_dino_tune(dino_tune)

        # 2) 探测 YOLO 的 P3/P4/P5 通道数（按 use_levels 对齐）
        with torch.no_grad():
            dev = next(core_yolo.parameters()).device
            dummy = torch.zeros(1, 3, imgsz_probe, imgsz_probe, device=dev)
            yolo_feats = self._forward_yolo_feats(dummy)
            self.yolo_chs = [t.shape[1] for t in yolo_feats]

        # 3) 为每个融合层构建 reducer 与 fuser
        dino_chs_all = list(self.dino.feature_info.channels())  # timm 提供
        self.level_ids = [i for i, flag in enumerate(self.use_levels) if flag]

        reducers, fusers = [], []
        for li in self.level_ids:
            c_y = self.yolo_chs[li]
            c_d = dino_chs_all[li if li < len(dino_chs_all) else -1]
            c_red = max(8, c_y // self.reduce_ratio)
            reducers.append(nn.Conv2d(c_d, c_red, kernel_size=1, bias=False))
            fusers.append(nn.Conv2d(c_y + c_red, c_y, kernel_size=1, bias=True))
        self.reducers = nn.ModuleList(reducers)
        self.fusers = nn.ModuleList(fusers)

    # ---------- helpers ----------
    def _apply_dino_tune(self, mode: str):
        d = getattr(self, "dino", None)
        if d is None:
            return
        mode = (mode or "frozen").lower()
        if mode == "frozen":
            for p in d.parameters():
                p.requires_grad = False
            d.eval()
        elif mode in ("trainable", "train", "open", "finetune"):
            for p in d.parameters():
                p.requires_grad = True
            d.train()
        elif mode in ("partial", "last"):
            for p in d.parameters():
                p.requires_grad = False
            # 按模型结构开放后两段（若存在）
            if hasattr(d, "stages") and isinstance(d.stages, (list, tuple)):
                for m in d.stages[-2:]:
                    for p in m.parameters():
                        p.requires_grad = True
                d.train()
            else:
                d.eval()
        else:
            for p in d.parameters():
                p.requires_grad = False
            d.eval()

    def _forward_core_graph(self, x):
        """
        严格沿 YOLO 计算图前向，收集各层输出，并按 Detect.f 取回 P3/P4/P5。
        """
        mdl = self.core_yolo.model  # nn.ModuleList
        y = []
        cur = x
        for m in mdl:
            f = getattr(m, "f", -1)
            if f != -1:
                idxs = f if isinstance(f, (list, tuple)) else [f]
                src = [(cur if j == -1 else y[j]) for j in idxs]
                cur = m(src if len(src) > 1 else src[0])
            else:
                cur = m(cur)
            y.append(cur)

        det = None
        for m in mdl:
            if m.__class__.__name__.lower().endswith("detect"):
                det = m
                break
        if det is None:
            raise RuntimeError("Detect head not found in core YOLO model")

        fidx = det.f if isinstance(det.f, (list, tuple)) else [det.f]
        feats_all = [(cur if j == -1 else y[j]) for j in fidx]  # [P3,P4,P5]
        return y, det, feats_all

    def _forward_yolo_feats(self, x):
        _, _, feats_all = self._forward_core_graph(x)
        out = []
        for i, use in enumerate(self.use_levels):
            if use and i < len(feats_all):
                out.append(feats_all[i])
        return out

    def _dino_feats_resized(self, x, ref_feats):
        d_feats_all = self.dino(x)  # len == len(out_indices)
        outs = []
        for k, li in enumerate(self.level_ids):
            fd = d_feats_all[li if li < len(d_feats_all) else -1]
            fr = ref_feats[k]
            if fd.shape[-2:] != fr.shape[-2:]:
                fd = F.interpolate(fd, size=fr.shape[-2:], mode="bilinear", align_corners=False)
            outs.append(fd)
        return outs

    def _fuse(self, yolo_feats_used, dino_feats_used):
        d_reduced, fused = [], []
        for k, _ in enumerate(self.level_ids):
            yk = yolo_feats_used[k]
            dk = self.reducers[k](dino_feats_used[k])
            fk = self.fusers[k](torch.cat([yk, dk], dim=1))
            d_reduced.append(dk)
            fused.append(fk)
        return d_reduced, fused

    # ---------- forward ----------
    def forward(self, x):
        y_layers, det, feats_all = self._forward_core_graph(x)
        yolo_used = [feats_all[i] for i, flag in enumerate(self.use_levels) if flag]

        with torch.set_grad_enabled(any(p.requires_grad for p in self.dino.parameters())):
            dino_resized = self._dino_feats_resized(x, yolo_used)

        d_red, fused_used = self._fuse(yolo_used, dino_resized)

        fused_all = []
        it = iter(fused_used)
        for flag, fa in zip(self.use_levels, feats_all):
            fused_all.append(next(it) if flag else fa)

        if getattr(self, "debug_dump", False):
            self._last_debug_taps = {
                "dino": [t.detach() for t in d_red],
                "fused": [t.detach() for t in fused_used],
            }

        out = det(fused_all)
        return out

    # 关键修复：先走 nn.Module 的 __getattr__（保证 _modules/_parameters 正常工作）
    # 只有当超类取不到属性时，才尝试转发给 core_yolo
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self.__dict__.get("core_yolo", None)
            if core is not None and hasattr(core, name):
                return getattr(core, name)
            raise
