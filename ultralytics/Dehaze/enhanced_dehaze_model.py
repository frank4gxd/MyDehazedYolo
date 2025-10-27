# enhanced_dehaze_model.py
# 增强版DehazeNet - 解决边界模糊和细节不清晰问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


# ==========================
# 基础组件一：LayerNorm2d
# ==========================
class LayerNorm2d(nn.Module):
    """对 4D 特征图 (N, C, H, W) 的通道维 C 做层归一化。"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        # 在 FP32 做 LN，避免 AMP 下 Half/Float 冲突
        x = self.ln(x.float()).to(orig_dtype)
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        return x


# ==========================
# 基础组件二：PONO (Positional Normalization)
# ==========================
class PONO(nn.Module):
    """位置归一化 (Positional Normalization)"""

    def __init__(self, input_size: Optional[tuple] = None, affine: bool = False, eps: float = 1e-5):
        super(PONO, self).__init__()
        self.eps = eps
        self.affine = affine
        if affine and input_size is not None:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x: torch.Tensor) -> tuple:
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, correction=0) + self.eps).sqrt()
        x_norm = (x - mean) / std
        if self.affine and self.gamma is not None:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mean, std


# ==========================
# 基础组件三：MS (Modulation and Scaling)
# ==========================
class MS(nn.Module):
    """调制与缩放 (Modulation and Scaling)"""

    def __init__(self):
        super(MS, self).__init__()

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


# ==========================
# 基础组件四：MHSA2d（2D特征上的多头自注意力）
# ==========================
class MHSA2d(nn.Module):
    """在 2D 特征图上做多头自注意力"""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)  # (N, S, C)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)  # token 维度 (C) 上 LN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (N, HW, C)
        out_dtype = tokens.dtype
        # 在 FP32 做 LN/Attn/Linear，避免 AMP 下 Half/Float 冲突
        device_type = 'cuda' if tokens.is_cuda else ('mps' if tokens.device.type == 'mps' else 'cpu')
        with torch.amp.autocast(device_type=device_type, enabled=False):
            t32 = tokens.float()
            t32 = self.norm(t32)
            out32, _ = self.attn(t32, t32, t32, need_weights=False)
            out32 = self.proj(out32) + t32  # 残差
        out = out32.to(out_dtype).transpose(1, 2).reshape(n, c, h, w)
        return out


# ==========================
# 基础组件五：ConvBlock
# ==========================
class ConvBlock(nn.Module):
    """3×3 卷积 (+ 可选 LayerNorm2d 或 PONO) + ReLU"""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = 'none', stride: int = 1):
        super().__init__()
        self.norm_type = norm_type
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

        if norm_type == 'ln':
            self.norm = LayerNorm2d(out_ch)
        elif norm_type == 'pono':
            self.pono = PONO(affine=False)
            self.ms = MS()
        elif norm_type == 'none':
            self.norm = None
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.norm_type == 'pono':
            x_norm, mean, std = self.pono(x)
            x = self.relu(x_norm)
            x = self.ms(x, mean, std)
        elif self.norm_type == 'ln':
            x = self.norm(x)
            x = self.relu(x)
        else:  # 'none'
            x = self.relu(x)

        return x


# ==========================
# 改进组件一：通道注意力
# ==========================
class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        out_dtype = y.dtype
        device_type = 'cuda' if y.is_cuda else ('mps' if y.device.type == 'mps' else 'cpu')
        # FC 在 FP32，再 cast 回来
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y32 = self.fc(y.float())
        y = y32.to(out_dtype).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==========================
# 改进组件二：边缘增强模块
# ==========================
class EdgeEnhancement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # depthwise Laplacian：每个通道各自卷积
        self.lap = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        with torch.no_grad():
            k = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            k = k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.lap.weight[:] = k

        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        edge = self.lap(x)               # 直接得到 [N,C,H,W] 的边缘
        edge = self.edge_conv(edge)
        return x + self.alpha * edge


# ==========================
# 改进组件三：增强解码器块
# ==========================
class EnhancedDecoderBlock(nn.Module):
    """增强的解码器块 - 更深的卷积和注意力机制"""

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = 'pono', use_attention: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 更深的卷积块增强细节恢复
        self.conv1 = ConvBlock(in_ch, out_ch, norm_type=norm_type)
        self.conv2 = ConvBlock(out_ch, out_ch, norm_type=norm_type)
        self.conv3 = ConvBlock(out_ch, out_ch, norm_type=norm_type)

        # 添加通道注意力
        if use_attention:
            self.ca = ChannelAttention(out_ch)
        else:
            self.ca = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.ca is not None:
            x = self.ca(x)

        return x


# ==========================
# 基础组件六：空间注意力门 (Spatial Attention Gate)
# ==========================
class SpatialAttentionGate(nn.Module):
    """空间注意力门 (AG)"""

    def __init__(self, F_g: int, F_l: int, F_int: int, norm_type: str = 'ln'):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # 根据norm_type选择归一化方式
        if norm_type == 'ln':
            self.W_g.add_module('norm', LayerNorm2d(F_int))
            self.W_x.add_module('norm', LayerNorm2d(F_int))
            self.psi_norm = LayerNorm2d(1)
        elif norm_type == 'pono':
            self.W_g_pono = PONO(affine=False)
            self.W_g_ms = MS()
            self.W_x_pono = PONO(affine=False)
            self.W_x_ms = MS()
            self.psi_pono = PONO(affine=False)
            self.psi_ms = MS()
        elif norm_type != 'none':
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.norm_type = norm_type
        self.psi_conv = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if self.norm_type == 'pono':
            g1_norm, g_mean, g_std = self.W_g_pono(g1)
            g1 = self.W_g_ms(g1_norm, g_mean, g_std)
            x1_norm, x_mean, x_std = self.W_x_pono(x1)
            x1 = self.W_x_ms(x1_norm, x_mean, x_std)

        psi = self.relu(g1 + x1)
        psi = self.psi_conv(psi)

        if self.norm_type == 'ln':
            psi = self.psi_norm(psi)
        elif self.norm_type == 'pono':
            psi_norm, psi_mean, psi_std = self.psi_pono(psi)
            psi = self.psi_ms(psi_norm, psi_mean, psi_std)

        psi = self.psi_sigmoid(psi)
        return x * psi


# ==========================
# AOD头：用于估计大气光学深度（雾浓度）
# ==========================
class AODHead(nn.Module):
    """AOD头：估计雾浓度图"""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.aod_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]，表示雾浓度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aod_net(x)


# ==========================
# 物理引导的去雾模块
# ==========================
class PhysicsGuidedDehazing(nn.Module):
    """使用物理模型引导的去雾：J = (I - A) / t + A

    该实现会自动将低分辨率的 transmission_map 或非 1x1 的 atmospheric_light
    上采样到 hazy_img 的空间分辨率以匹配输入，避免尺寸不一致的广播错误。
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                hazy_img: torch.Tensor,
                transmission_map: torch.Tensor,
                atmospheric_light: torch.Tensor,
                epsilon: float = 1e-8) -> torch.Tensor:
        """
        Args:
            hazy_img: 有雾图像 [N, 3, H, W]
            transmission_map: 透射率图 [N, 1, h_t, w_t]（可能是低分辨率）
            atmospheric_light: 大气光 [N, 3, 1, 1] 或 [N, 3, h_a, w_a]
            epsilon: 防止除零的小值
        Returns:
            dehazed_img: 去雾后的图像 [N, 3, H, W]
        """
        # 基本维度检查（更友好地报错）
        assert hazy_img.dim() == 4 and hazy_img.size(1) == 3, \
            f"hazy_img must be [N,3,H,W], got {tuple(hazy_img.shape)}"
        assert transmission_map.dim() == 4 and transmission_map.size(1) == 1, \
            f"transmission_map must be [N,1,h,w], got {tuple(transmission_map.shape)}"
        assert atmospheric_light.dim() == 4 and atmospheric_light.size(1) == 3, \
            f"atmospheric_light must be [N,3,1,1] or [N,3,h,w], got {tuple(atmospheric_light.shape)}"

        # 1) clamp transmission 到合理区间，避免过小导致数值爆炸
        t = torch.clamp(transmission_map, 0.1, 1.0)

        # 2) 若 t 与输入分辨率不同，上采样到输入分辨率（双线性）
        if t.shape[2:] != hazy_img.shape[2:]:
            t = F.interpolate(t, size=hazy_img.shape[2:], mode='bilinear', align_corners=False)

        # 3) 处理 atmospheric_light：如果不是全局 1x1，则上采样到输入分辨率
        if atmospheric_light.shape[2:] != hazy_img.shape[2:] and atmospheric_light.shape[2:] != (1, 1):
            atmospheric_light = F.interpolate(atmospheric_light, size=hazy_img.shape[2:],
                                              mode='bilinear', align_corners=False)

        # 4) 如果 atmospheric_light 是 [N,3,1,1]，广播会自动发生
        dehazed = (hazy_img - atmospheric_light) / (t + epsilon) + atmospheric_light

        # 5) 限制到 [0,1]
        dehazed = torch.clamp(dehazed, 0.0, 1.0)
        return dehazed


# ==========================
# 大气光估计模块
# ==========================
class AtmosphericLightEstimator(nn.Module):
    """估计全局大气光"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 3),
            nn.Sigmoid()  # 大气光通常在 [0, 1] 范围内
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局平均池化
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)  # [N, C]
        out_dtype = pooled.dtype
        device_type = 'cuda' if pooled.is_cuda else ('mps' if pooled.device.type == 'mps' else 'cpu')
        # 全连接层在 FP32，结果 cast 回来
        with torch.amp.autocast(device_type=device_type, enabled=False):
            A32 = self.fc(pooled.float())  # [N, 3]
        return A32.to(out_dtype).unsqueeze(-1).unsqueeze(-1)


# ==========================
# DINOv3 ConvNeXt backbone provider (timm)
# ==========================
class DinoConvNeXtProvider(nn.Module):
    def __init__(self, model_name='convnext_small.dinov3_lvd1689m',
                 pretrained=True, freeze=True, out_indices=(0, 1, 2, 3)):
        super().__init__()
        try:
            import timm
            from timm.data import resolve_model_data_config
        except ImportError as e:
            raise ImportError("Please install timm: pip install timm") from e

        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          features_only=True, out_indices=out_indices)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # register normalization buffers so we can normalize x on-the-fly
        cfg = resolve_model_data_config(self.backbone)
        mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
        self.register_buffer("dinomean", mean, persistent=False)
        self.register_buffer("dinostd", std, persistent=False)

    def forward(self, x: torch.Tensor):
        mean = self.dinomean.to(device=x.device, dtype=x.dtype)
        std = self.dinostd.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return self.backbone(x)


# ==========================

# 改进的主干：EnhancedDehazeNet
# ==========================
class EnhancedDehazeNet(nn.Module):
    """增强版DehazeNet - 解决边界模糊和细节不清晰问题"""

    def __init__(self, in_ch: int = 3, base_ch: int = 48, heads: int = 4,
                 bottleneck_type: str = 'attention', use_aod_head: bool = True,
                 use_physics_guidance: bool = True, norm_type: str = 'pono',
                 use_dino_backbone: bool = False,
                 dino_name: str = 'convnext_small.dinov3_lvd1689m',
                 dino_freeze: bool = True,
                 use_edge_enhancement: bool = True,
                 use_channel_attention: bool = True):
        super().__init__()
        B = base_ch
        self.out_channels = B
        self.bottleneck_type = bottleneck_type
        self.use_aod_head = use_aod_head
        self.use_physics_guidance = use_physics_guidance
        self.norm_type = norm_type
        self.use_dino_backbone = use_dino_backbone
        self.use_edge_enhancement = use_edge_enhancement
        self.use_channel_attention = use_channel_attention

        if use_dino_backbone:
            # 1) DINO backbone
            self.dino = DinoConvNeXtProvider(dino_name, pretrained=True, freeze=dino_freeze)
            # 2) Channel adapters: [96,192,384,768] -> [B,2B,4B,8B]
            self.adapt1 = nn.Conv2d(96, 1 * B, 1)
            self.adapt2 = nn.Conv2d(192, 2 * B, 1)
            self.adapt3 = nn.Conv2d(384, 4 * B, 1)
            self.adapt4 = nn.Conv2d(768, 8 * B, 1)
        else:
            # original encoders
            self.enc1 = ConvBlock(in_ch, B, norm_type=norm_type, stride=2)
            self.enc2 = ConvBlock(B, 2 * B, norm_type=norm_type, stride=2)
            self.enc3 = ConvBlock(2 * B, 4 * B, norm_type=norm_type, stride=2)
            self.enc4 = ConvBlock(4 * B, 8 * B, norm_type=norm_type, stride=2)

        # ------- 瓶颈 -------
        if bottleneck_type == 'attention':
            # 自注意力瓶颈
            self.attn_in = nn.Conv2d(8 * B, 4 * B, kernel_size=1, stride=1, padding=0)

            # 根据norm_type选择瓶颈归一化方式
            if norm_type == 'pono':
                self.pono_bottleneck = PONO(affine=False)
                self.ms_bottleneck = MS()
            elif norm_type == 'ln':
                self.ln_bottleneck = LayerNorm2d(4 * B)
            elif norm_type != 'none':
                raise ValueError(f"Unsupported norm_type: {norm_type}")

            self.attn = MHSA2d(dim=4 * B, num_heads=heads, dropout=0.0)
            self.fuse_1x1 = nn.Conv2d(4 * B + 8 * B, 4 * B, kernel_size=1, stride=1, padding=0)
        else:
            # 卷积瓶颈
            self.bottleneck_conv = nn.Sequential(
                nn.Conv2d(8 * B, 8 * B, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8 * B, 8 * B, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.fuse_1x1 = nn.Conv2d(8 * B + 8 * B, 4 * B, kernel_size=1)

        # ------- AOD头 -------
        if use_aod_head:
            self.aod_head = AODHead(8 * B)  # 在编码器最深层特征上估计AOD
            self.atmospheric_light_estimator = AtmosphericLightEstimator(8 * B)
            self.physics_dehazing = PhysicsGuidedDehazing()

        # ------- 增强的解码器 -------
        self.enhanced_dec4 = EnhancedDecoderBlock(4 * B, 4 * B, norm_type, use_channel_attention)
        self.enhanced_dec3 = EnhancedDecoderBlock(4 * B, 2 * B, norm_type, use_channel_attention)
        self.enhanced_dec2 = EnhancedDecoderBlock(2 * B, B, norm_type, use_channel_attention)
        self.enhanced_dec1 = EnhancedDecoderBlock(B, B, norm_type, use_channel_attention)

        # NEW: extra upsample to reach full-res when using DINO (e4 is 1/32)
        self.enhanced_dec0 = EnhancedDecoderBlock(B, B, norm_type,
                                                  use_channel_attention) if self.use_dino_backbone else None

        # ------- 空间注意力门 -------
        self.ag_skip3 = SpatialAttentionGate(F_g=4 * B, F_l=4 * B, F_int=2 * B, norm_type=norm_type)
        self.ag_skip2 = SpatialAttentionGate(F_g=2 * B, F_l=2 * B, F_int=1 * B, norm_type=norm_type)
        self.ag_skip1 = SpatialAttentionGate(F_g=1 * B, F_l=1 * B, F_int=B // 2, norm_type=norm_type)

        # ------- 跳连融合 1x1 conv -------
        self.fuse_d4 = nn.Conv2d(8 * B, 4 * B, kernel_size=1)
        self.fuse_d3 = nn.Conv2d(4 * B, 2 * B, kernel_size=1)
        self.fuse_d2 = nn.Conv2d(2 * B, B, kernel_size=1)

        # ------- 边缘增强 -------
        if use_edge_enhancement:
            self.edge_enhance = EdgeEnhancement(B)

        # ------- 增强的输出精炼模块 -------
        self.enhanced_refine = nn.Sequential(
            nn.Conv2d(B, B, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(B, B, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(B, B, 3, 1, 1),
            nn.ReLU(inplace=True),
        ) if use_edge_enhancement else nn.Sequential(
            nn.Conv2d(B, B, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(B, B, 3, 1, 1),
        )

        self.out_conv = nn.Conv2d(B, 3, kernel_size=3, stride=1, padding=1)

    # ----- helper: choose encoder path -----
    def _encode_features(self, x):
        if self.use_dino_backbone:
            c1, c2, c3, c4 = self.dino(x)  # DINO features
            e1, e2, e3, e4 = self.adapt1(c1), self.adapt2(c2), self.adapt3(c3), self.adapt4(c4)
            return e1, e2, e3, e4
        else:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            return e1, e2, e3, e4

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4 = self._encode_features(x)

        # AOD估计
        aod_outputs = {}
        if self.use_aod_head:
            transmission_map = self.aod_head(e4)  # 估计透射率图
            atmospheric_light = self.atmospheric_light_estimator(e4)  # 估计大气光
            aod_outputs = {
                'transmission_map': transmission_map,
                'atmospheric_light': atmospheric_light
            }

        # 瓶颈处理
        if self.bottleneck_type == 'attention':
            a_in = self.attn_in(e4)

            # 根据norm_type应用不同的归一化
            if self.norm_type == 'pono':
                a_in_norm, mean, std = self.pono_bottleneck(a_in)
                a_out = self.attn(a_in_norm)
                a_out = self.ms_bottleneck(a_out, mean, std)
            elif self.norm_type == 'ln':
                a_in_norm = self.ln_bottleneck(a_in)
                a_out = self.attn(a_in_norm)
            else:  # 'none'
                a_out = self.attn(a_in)

            fcat = torch.cat([a_out, e4], dim=1)
        else:
            b = self.bottleneck_conv(e4)
            fcat = torch.cat([b, e4], dim=1)

        f = self.fuse_1x1(fcat)

        # 使用增强的解码器
        d4 = self.enhanced_dec4(f)
        d4_skip = self.ag_skip3(d4, e3)
        d4 = torch.cat([d4, d4_skip], dim=1)
        d4 = self.fuse_d4(d4)

        d3 = self.enhanced_dec3(d4)
        d3_skip = self.ag_skip2(d3, e2)
        d3 = torch.cat([d3, d3_skip], dim=1)
        d3 = self.fuse_d3(d3)

        d2 = self.enhanced_dec2(d3)
        d2_skip = self.ag_skip1(d2, e1)
        d2 = torch.cat([d2, d2_skip], dim=1)
        d2 = self.fuse_d2(d2)

        d1 = self.enhanced_dec1(d2)

        # 边缘增强
        if self.use_edge_enhancement:
            d1 = self.edge_enhance(d1)

        return d1, aod_outputs

    def forward(self, x: torch.Tensor) -> dict:
        d1, aod_outputs = self.forward_features(x)

        # NEW: if DINO backbone is used, upsample once more to H×W
        if self.enhanced_dec0 is not None:
            d1 = self.enhanced_dec0(d1)

        d1 = d1 + self.enhanced_refine(d1)
        direct_output = self.out_conv(d1)

        outputs = {'dehazed': direct_output}

        # 如果使用AOD头和物理引导
        if self.use_aod_head and self.use_physics_guidance:
            transmission_map = aod_outputs['transmission_map']
            atmospheric_light = aod_outputs['atmospheric_light']

            physics_output = self.physics_dehazing(x, transmission_map, atmospheric_light)

            assert direct_output.shape[2:] == physics_output.shape[2:], \
                f"Fuse size mismatch: direct={direct_output.shape}, physics={physics_output.shape}"

            alpha = 0.7
            fused_output = alpha * direct_output + (1 - alpha) * physics_output
            outputs.update({
                'physics_dehazed': physics_output,
                'transmission_map': transmission_map,
                'atmospheric_light': atmospheric_light,
                'fused_dehazed': fused_output
            })
        return outputs


# ==========================
# 改进的损失函数
# ==========================
class EnhancedDehazeLoss(nn.Module):
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        self.loss_weights = loss_weights or {
            'mse': 1.0,
            'ssim': 0.5,
            'edge': 0.8,
            'perceptual': 0.0
        }
        self.mse_loss = nn.MSELoss()

        lap = torch.tensor(
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", lap, persistent=False)

    def ssim_loss(self, x, y, window_size: Optional[int] = None, size_average: bool = True):
        try:
            from pytorch_msssim import ssim
        except ImportError:
            return self.mse_loss(x, y)

        x32, y32 = x.float(), y.float()
        h, w = x32.shape[-2], x32.shape[-1]
        if min(h, w) < 3:
            return self.mse_loss(x, y)

        ws = min(11 if window_size is None else window_size, min(h, w))
        if ws % 2 == 0: ws -= 1
        ws = max(3, ws)

        with torch.amp.autocast(device_type=('cuda' if x32.is_cuda else 'cpu'), enabled=False):
            try:
                return 1.0 - ssim(x32, y32, data_range=1.0, win_size=ws, size_average=size_average)
            except Exception:
                return self.mse_loss(x, y)

    def edge_loss(self, pred, target):
        # 关键：同时对齐 device 和 dtype（不要只对齐 dtype）
        k = self.laplacian_kernel.to(device=pred.device, dtype=pred.dtype)

        pred_edges = F.conv2d(pred.mean(dim=1, keepdim=True), k, padding=1)
        target_edges = F.conv2d(target.mean(dim=1, keepdim=True), k, padding=1)

        # 数值检查（可留可去，出问题时返回0保持训练继续）
        if torch.isnan(pred_edges).any() or torch.isinf(pred_edges).any():
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        if torch.isnan(target_edges).any() or torch.isinf(target_edges).any():
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        return F.l1_loss(pred_edges, target_edges)

    def forward(self, pred, target):
        pred   = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        losses: Dict[str, torch.Tensor] = {}

        # 输入数值检查
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            losses['mse'] = self.mse_loss(pred, target)
            losses['total'] = losses['mse']
            return losses

        if self.loss_weights.get('mse', 0.0) > 0.0:
            losses['mse'] = self.mse_loss(pred, target)

        if self.loss_weights.get('ssim', 0.0) > 0.0:
            ssim_loss_val = self.ssim_loss(pred, target)
            if torch.isfinite(ssim_loss_val):
                losses['ssim'] = ssim_loss_val

        if self.loss_weights.get('edge', 0.0) > 0.0:
            edge_loss_val = self.edge_loss(pred, target)
            if torch.isfinite(edge_loss_val):
                losses['edge'] = edge_loss_val

        # 用张量 0 初始化更稳
        total = pred.new_tensor(0.0)
        for name, w in self.loss_weights.items():
            if name in losses and torch.isfinite(losses[name]):
                total = total + w * losses[name]

        if not torch.isfinite(total):
            total = self.mse_loss(pred, target)

        losses['total'] = total
        return losses


if __name__ == "__main__":
    # 测试增强模型
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.randn(1, 3, 512, 512).to(device)

    # 测试增强模型
    net_enhanced = EnhancedDehazeNet(
        use_dino_backbone=False,
        use_edge_enhancement=True,
        use_channel_attention=True
    ).to(device)

    y_enhanced = net_enhanced(x)
    print("[Enhanced] keys:", y_enhanced.keys())
    print("[Enhanced] output shape:", y_enhanced['dehazed'].shape)

    # 测试损失函数
    target = torch.randn_like(y_enhanced['dehazed'])
    loss_fn = EnhancedDehazeLoss()
    losses = loss_fn(y_enhanced['dehazed'], target)
    print("Losses:", losses)
