from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .resnet import *
from .mobilenet import *

# === [MOD-3] Import modified UniFuse layers ====================================
# The final gated model uses an input adapter without a Q prediction head.
from .layers import (
    Conv3x3,
    ConvBlock,
    upsample,
    Cube2Equirec,
    Concat,
    BiProj,
    CEELayer,
    UnderwaterInputAdaptationStem,
    IdentityInputStem,
    DegradationAwareGatedFusion,
    QualityFreeGatedFusion,
    FixedAverageFusion,
)
# === [MOD-3] End =================================================================


class UniFuse(nn.Module):
    """
    UniFuse with underwater input adaptation and gated ERP/Cube fusion.

    The final ``gated`` path does not instantiate or evaluate a Q prediction
    head. The old Q-conditioned ``dagf`` path remains available only so archived
    experiments can still be loaded.
    """

    def __init__(
        self,
        num_layers,
        equi_h,
        equi_w,
        pretrained=False,
        max_depth=10.0,
        fusion_type="cee",
        se_in_fusion=True,
        use_input_adaptation=True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h // 2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion
        self.use_input_adaptation = bool(use_input_adaptation)
        self.max_depth = max_depth

        # -------------------------
        # Encoders (ResNet)
        # -------------------------
        encoder = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152,
        }
        if num_layers not in encoder:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.equi_encoder = encoder[num_layers](pretrained)
        self.cube_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # Only the archived Q-conditioned DAGF path instantiates quality heads.
        predict_quality = fusion_type == "dagf"
        if not self.use_input_adaptation and predict_quality:
            raise ValueError("The archived Q-conditioned DAGF path requires input adaptation")
        StemLayer = UnderwaterInputAdaptationStem if self.use_input_adaptation else IdentityInputStem
        if self.use_input_adaptation:
            stem_kwargs = dict(
                in_channels=3,
                out_channels=3,
                base_channels=16,
                corr_scale=0.05,
                predict_quality=predict_quality,
            )
            self.stem_equi = StemLayer(**stem_kwargs)
            self.stem_cube = StemLayer(**stem_kwargs)
        else:
            self.stem_equi = StemLayer()
            self.stem_cube = StemLayer()
        # === [MOD-4] End =======================================================

        # -------------------------
        # Decoder + Fusion blocks
        # -------------------------
        self.equi_dec_convs = OrderedDict()
        self.c2e = {}

        # === [MOD-5] Add gated ERP/Cube fusion types ===============================
        # Gated ERP/Cube fusion keeps UniFuse unidirectional cube-to-ERP fusion.
        # The v3 default uses fusion_type="gated", where Q is not a gate input.
        Fusion_dict = {
            "cat": Concat,
            "biproj": BiProj,
            "cee": CEELayer,
            "dagf": DegradationAwareGatedFusion,
            "dagf_no_q": QualityFreeGatedFusion,
            "gated": QualityFreeGatedFusion,
            "fixed_avg": FixedAverageFusion,
        }
        if fusion_type not in Fusion_dict:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        FusionLayer = Fusion_dict[fusion_type]
        # === [MOD-5] End =======================================================

        # Fusion blocks
        if fusion_type == "cee":
            self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        else:
            # Concat / BiProj / gated variants accept **kwargs and ignore SE if unused
            self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
            self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)

        # cube -> equirect operators + decoder convs
        self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))

    def forward(self, input_equi_image, input_cube_image):
        """
        input_equi_image: [B,3,H,W]
        input_cube_image: [B,3,cube_h,cube_h*6]
        """
        b = input_equi_image.shape[0]

        # Apply the underwater input adapters before the encoders.
        equi_corr, q_equi = self.stem_equi(input_equi_image)
        cube_corr, _ = self.stem_cube(input_cube_image)

        # 2) cube faces -> batch
        cube_inputs = torch.cat(torch.split(cube_corr, self.cube_h, dim=-1), dim=0)  # [B*6,3,cube_h,cube_h]

        # 3) encode ERP (use equi_corr instead of raw input)
        x = self.equi_encoder.conv1(equi_corr)
        x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
        equi_enc_feat0 = x
        x = self.equi_encoder.maxpool(x)
        equi_enc_feat1 = self.equi_encoder.layer1(x)
        equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
        equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
        equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)

        # 4) encode Cube (faces)
        x = self.cube_encoder.conv1(cube_inputs)
        x = self.cube_encoder.relu(self.cube_encoder.bn1(x))
        cube_enc_feat0 = x
        x = self.cube_encoder.maxpool(x)
        cube_enc_feat1 = self.cube_encoder.layer1(x)
        cube_enc_feat2 = self.cube_encoder.layer2(cube_enc_feat1)
        cube_enc_feat3 = self.cube_encoder.layer3(cube_enc_feat2)
        cube_enc_feat4 = self.cube_encoder.layer4(cube_enc_feat3)

        # merge faces back to width-concat form
        cube_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, b, dim=0), dim=-1)
        cube_enc_feat3 = torch.cat(torch.split(cube_enc_feat3, b, dim=0), dim=-1)
        cube_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, b, dim=0), dim=-1)
        cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, b, dim=0), dim=-1)
        cube_enc_feat0 = torch.cat(torch.split(cube_enc_feat0, b, dim=0), dim=-1)

        # cube -> equirect
        c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
        c2e_enc_feat3 = self.c2e["4"](cube_enc_feat3)
        c2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
        c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
        c2e_enc_feat0 = self.c2e["1"](cube_enc_feat0)

        outputs = {}

        # Legacy "dagf" consumes Q; the default "gated" path is Q-free.
        if self.fusion_type == "dagf":
            q4 = F.interpolate(q_equi, size=equi_enc_feat4.shape[2:], mode="bilinear", align_corners=False)
            fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4, q4)
        else:
            fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, c2e_enc_feat4)
        # === [MOD-7] End ==========================================================

        x = self.equi_dec_convs["upconv_5"](fused_feat4)
        x = upsample(x)

        if self.fusion_type == "dagf":
            q3 = F.interpolate(q_equi, size=equi_enc_feat3.shape[2:], mode="bilinear", align_corners=False)
            fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, c2e_enc_feat3, q3)
        else:
            fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, c2e_enc_feat3)

        x = torch.cat([x, fused_feat3], 1)
        x = self.equi_dec_convs["deconv_4"](x)
        x = self.equi_dec_convs["upconv_4"](x)
        x = upsample(x)

        if self.fusion_type == "dagf":
            q2 = F.interpolate(q_equi, size=equi_enc_feat2.shape[2:], mode="bilinear", align_corners=False)
            fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, c2e_enc_feat2, q2)
        else:
            fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, c2e_enc_feat2)

        x = torch.cat([x, fused_feat2], 1)
        x = self.equi_dec_convs["deconv_3"](x)
        x = self.equi_dec_convs["upconv_3"](x)
        x = upsample(x)

        if self.fusion_type == "dagf":
            q1 = F.interpolate(q_equi, size=equi_enc_feat1.shape[2:], mode="bilinear", align_corners=False)
            fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, c2e_enc_feat1, q1)
        else:
            fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, c2e_enc_feat1)

        x = torch.cat([x, fused_feat1], 1)
        x = self.equi_dec_convs["deconv_2"](x)
        x = self.equi_dec_convs["upconv_2"](x)
        x = upsample(x)

        if self.fusion_type == "dagf":
            q0 = F.interpolate(q_equi, size=equi_enc_feat0.shape[2:], mode="bilinear", align_corners=False)
            fused_feat0 = self.equi_dec_convs["fusion_1"](equi_enc_feat0, c2e_enc_feat0, q0)
        else:
            fused_feat0 = self.equi_dec_convs["fusion_1"](equi_enc_feat0, c2e_enc_feat0)

        x = torch.cat([x, fused_feat0], 1)
        x = self.equi_dec_convs["deconv_1"](x)
        x = self.equi_dec_convs["upconv_1"](x)
        x = upsample(x)

        x = self.equi_dec_convs["deconv_0"](x)
        disp = self.equi_dec_convs["depthconv_0"](x)
        pred_depth = self.max_depth * torch.sigmoid(disp)

        outputs["pred_depth"] = pred_depth

        # Only archived Q-conditioned DAGF models expose a quality map.
        if q_equi is not None:
            outputs["quality_map"] = q_equi
        # === [MOD-8] End =========================================================

        return outputs


