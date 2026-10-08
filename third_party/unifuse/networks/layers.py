import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by activation"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


# Based on https://github.com/sunset1995/py360convert
class Cube2Equirec(nn.Module):
    def __init__(self, face_w, equ_h, equ_w):
        super().__init__()
        """
        face_w: int, the length of each face of the cubemap
        equ_h: int, height of the equirectangular image
        equ_w: int, width of the equirectangular image
        """
        self.face_w = face_w
        self.equ_h = equ_h
        self.equ_w = equ_w
        self._create_lookup()

    def _create_lookup(self):
        equ_h, equ_w = self.equ_h, self.equ_w

        jj, ii = np.meshgrid(np.arange(equ_w), np.arange(equ_h))
        u = (jj / equ_w) * 2 * np.pi - np.pi
        v = (ii / equ_h) * np.pi - np.pi / 2

        x = np.cos(v) * np.sin(u)
        y = np.sin(v)
        z = np.cos(v) * np.cos(u)

        absx, absy, absz = np.abs(x), np.abs(y), np.abs(z)
        is_x = (absx >= absy) & (absx >= absz)
        is_y = (absy > absx) & (absy >= absz)
        is_z = (absz > absx) & (absz > absy)

        face = np.zeros((equ_h, equ_w), dtype=np.int64)

        # faces order: [F, R, B, L, U, D]
        face[is_z & (z > 0)] = 0
        face[is_x & (x > 0)] = 1
        face[is_z & (z < 0)] = 2
        face[is_x & (x < 0)] = 3
        face[is_y & (y > 0)] = 4
        face[is_y & (y < 0)] = 5

        uc = np.zeros_like(x)
        vc = np.zeros_like(x)

        m = face == 0
        uc[m] = x[m] / absz[m]
        vc[m] = -y[m] / absz[m]

        m = face == 1
        uc[m] = -z[m] / absx[m]
        vc[m] = -y[m] / absx[m]

        m = face == 2
        uc[m] = -x[m] / absz[m]
        vc[m] = -y[m] / absz[m]

        m = face == 3
        uc[m] = z[m] / absx[m]
        vc[m] = -y[m] / absx[m]

        m = face == 4
        uc[m] = x[m] / absy[m]
        vc[m] = z[m] / absy[m]

        m = face == 5
        uc[m] = x[m] / absy[m]
        vc[m] = -z[m] / absy[m]

        grid = np.stack([uc, vc], axis=-1).astype(np.float32)
        self.register_buffer("grid", torch.from_numpy(grid))  # [H,W,2]
        self.register_buffer("face", torch.from_numpy(face))  # [H,W]

    def forward(self, cube_feat):
        """
        cube_feat: [B, C, face_w, face_w*6] with faces concatenated along width
        return:    [B, C, equ_h, equ_w]
        """
        b, c, fh, fw6 = cube_feat.shape
        assert fh == self.face_w, f"Expected face_h={self.face_w}, got {fh}"
        assert fw6 == self.face_w * 6, f"Expected face_w*6={self.face_w*6}, got {fw6}"

        faces = torch.split(cube_feat, self.face_w, dim=-1)  # 6 * [B,C,fh,fh]
        faces = torch.stack(faces, dim=1)                    # [B,6,C,fh,fh]
        faces = faces.view(b * 6, c, fh, fh)                 # [B*6,C,fh,fh]

        grid = self.grid.to(cube_feat.device)
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)          # [B,H,W,2]

        face = self.face.to(cube_feat.device)
        face = face.unsqueeze(0).repeat(b, 1, 1)             # [B,H,W]

        out = torch.zeros((b, c, self.equ_h, self.equ_w),
                          device=cube_feat.device, dtype=cube_feat.dtype)

        for f in range(6):
            mask = (face == f)
            if not mask.any():
                continue
            f_feat = faces[f::6]  # [B,C,fh,fh]
            sampled = F.grid_sample(
                f_feat, grid, mode="bilinear", padding_mode="border", align_corners=False
            )  # [B,C,H,W]
            out = torch.where(mask.unsqueeze(1), sampled, out)

        return out


class Concat(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, equi_feat, c2e_feat):
        x = torch.cat([equi_feat, c2e_feat], 1)
        x = self.relu(self.conv(x))
        return x


class BiProj(nn.Module):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.conv_c2e = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_e2c = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, equi_feat, c2e_feat):
        aaa = self.conv_e2c(equi_feat)
        tmp_equi = self.conv_c2e(c2e_feat)
        mask_equi = self.conv_mask(torch.cat([aaa, tmp_equi], dim=1))
        tmp_equi = tmp_equi.clone()
        tmp_equi *= mask_equi
        return equi_feat + tmp_equi


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CEELayer(nn.Module):
    def __init__(self, channels, SE=True):
        super().__init__()
        self.res_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)
        self.res_bn1 = nn.BatchNorm2d(channels)

        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.SE = SE
        if self.SE:
            self.selayer = SELayer(channels * 2)

        self.conv = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, equi_feat, c2e_feat):
        x = torch.cat([equi_feat, c2e_feat], 1)
        if self.SE:
            x = self.selayer(x)
        x = self.conv(x)

        residual = x
        out = self.res_conv1(torch.cat([equi_feat, c2e_feat], 1))
        out = self.res_bn1(out)
        out = self.relu(out)
        out = self.res_conv2(out)
        out = self.res_bn2(out)
        out += residual
        out = self.relu(out)
        return out


# === [MOD-1] New: Stable Degradation-aware Stem (方案B的一部分) =====================
# 作用：
# - 在不改变主干encoder第一层(仍吃3通道RGB)的前提下，对水下退化做轻量“输入稳定化”
# - 同时显式输出质量图Q（0~1），用于ERP侧门控融合（版本2核心）
# 设计要点：
# - out_channels 默认=3，确保可直接喂给ResNet conv1
# - 使用 GroupNorm 替代 BatchNorm（小batch更稳）
# - residual加回去用小系数corr_scale，避免输入分布漂移导致训练不稳定
class UnderwaterInputAdaptationStem(nn.Module):
    """Residual underwater input adapter with optional legacy Q output."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=16,
        corr_scale=0.05,
        predict_quality=False,
    ):
        super().__init__()
        self.corr_scale = float(corr_scale)

        self.corr = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, base_channels), num_channels=base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, base_channels), num_channels=base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1, bias=True),
        )

        self.quality = None
        if predict_quality:
            self.quality = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(4, base_channels), num_channels=base_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(base_channels, 1, 1, padding=0, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x):
        dx = self.corr(x)
        x_corr = x + self.corr_scale * dx
        q = None if self.quality is None else self.quality(x)
        return x_corr, q


class IdentityInputStem(nn.Module):
    """Identity replacement used by the input-adaptation ablation."""

    def forward(self, x):
        return x, None


# Backward-compatible import name for archived experiments.
DegradationAwareStem = UnderwaterInputAdaptationStem
# === [MOD-1] End =================================================================


# === [MOD-2] New: Degradation-aware Gated Fusion (Version-2核心) ===================
# 作用：
# - 融合发生在 ERP 分支（单向融合不变）：ERP特征 + (cube->ERP投影)特征
# - 用质量图Q引导门控权重g，让网络在退化严重区域更谨慎地使用不可靠特征
# - 公式语义：fused = g*ERP + (1-g)*Cube2ERP
# 设计要点：
# - g通过一个很小的gate_net学习得到（输出0~1）
# - refine为轻量细化卷积（可去掉，但保留能让融合后更平滑稳定）
class DegradationAwareGatedFusion(nn.Module):
    """
    Degradation-aware gated fusion:
      g = gate(equi_feat, c2e_feat, quality_map) in [0,1]
      fused = g*equi_feat + (1-g)*c2e_feat
    """
    def __init__(self, channels, **kwargs):
        super().__init__()
        hidden = max(16, channels // 4)

        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, 1, bias=False), # nn.Conv2d(输入通道数, 输出通道数, 卷积核大小, 偏置项设置)
            nn.GroupNorm(num_groups=min(4, hidden), num_channels=hidden), # 分组归一化；num_groups=min(4, hidden)：分组数：最多分4组，但如果hidden小于4，则按hidden分组；num_channels=hidden：输入通道数；作用：对特征进行归一化，加速训练收敛，提高模型稳定性
            nn.SiLU(inplace=True), # Sigmoid Linear Unit激活函数；公式：SiLU(x) = x * sigmoid(x)；inplace=True：原地操作，节省内存
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 精炼网络：作用：归一化：稳定训练，加速收敛
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, channels), num_channels=channels),
            nn.SiLU(inplace=True),
        )
    # 前向传播;作用：确保必须提供水下退化质量图
    def forward(self, equi_feat, c2e_feat, quality_map):
        if quality_map is None:
            raise ValueError("quality_map must be provided for DegradationAwareGatedFusion")

        if quality_map.shape[2:] != equi_feat.shape[2:]:
            quality_map = F.interpolate(
                quality_map, size=equi_feat.shape[2:], mode="bilinear", align_corners=False # 作用：将质量图缩放到与特征图相同尺寸；shape[2:]：获取特征图的高度和宽度；mode="bilinear"：双线性插值；align_corners=False：不考虑对角线像素的权重
            )
        # 特征拼接
        x = torch.cat([equi_feat, c2e_feat, quality_map], dim=1) # dim=1：在通道维度拼接；输入：equi_feat：等矩形投影特征；c2e_feat：从立方体到等矩形的投影特征；quality_map：水下退化质量图
        g = self.gate_net(x)  # [B,1,H,W] 作用：生成门控图g ∈ [0,1]^[B×1×H×W]
        fused = g * equi_feat + (1.0 - g) * c2e_feat
        return self.refine(fused) # 对融合特征进行进一步处理，增强特征质量
        # 设计原理：简单的加权融合可能丢失细节，3×3卷积可恢复空间上下文
# === [MOD-2] End =================================================================


class QualityFreeGatedFusion(nn.Module):
    """
    Q-free gated fusion ablation:
      g = gate(equi_feat, c2e_feat) in [0,1]
      fused = g*equi_feat + (1-g)*c2e_feat

    This keeps the gated ERP/Cube fusion mechanism but removes Q from the
    gate input, isolating the value of gated ERP/Cube fusion.
    """
    def __init__(self, channels, **kwargs):
        super().__init__()
        hidden = max(16, channels // 4)

        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(num_groups=min(4, hidden), num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, channels), num_channels=channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, equi_feat, c2e_feat):
        x = torch.cat([equi_feat, c2e_feat], dim=1)
        g = self.gate_net(x)
        fused = g * equi_feat + (1.0 - g) * c2e_feat
        return self.refine(fused)


class FixedAverageFusion(nn.Module):
    """Gate ablation with a fixed 0.5/0.5 ERP/Cube mixing ratio."""

    def __init__(self, channels, **kwargs):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(4, channels), num_channels=channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, equi_feat, c2e_feat):
        return self.refine(0.5 * equi_feat + 0.5 * c2e_feat)
