import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] added for potential augmentation logic


# [REG] DropPath implementation
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


# --------------------------------------------------------------------------- #
# Helper Modules
# --------------------------------------------------------------------------- #
class SqueezeExcite3D(nn.Module):
    """Channel‑wise attention for 3‑D tensors."""
    def __init__(self, in_ch: int, se_ratio: float = 0.25, head_dropout: float = 0.0):  # [REG] added head_dropout param
        super().__init__()
        squeezed = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv3d(in_ch, squeezed, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()  # [REG]
        self.fc2 = nn.Conv3d(squeezed, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool3d(x, 1)
        s = F.silu(self.fc1(s))
        s = self.dropout(s)  # [REG]
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv3D(nn.Module):
    """MobileNet‑style inverted bottleneck block for 3‑D."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        drop_prob: float = 0.0,
        se_ratio: float = 0.25,
        head_dropout: float = 0.0,  # [REG]
        drop_path_prob: float = 0.0,  # [REG]
    ):
        super().__init__()
        self.use_resid = stride == 1 and in_ch == out_ch
        self.drop_path_prob = drop_path_prob  # [REG]
        mid = in_ch * expand_ratio
        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_ch, mid, 1, bias=False),
                nn.BatchNorm3d(mid),
                nn.SiLU(inplace=True),
            ])

        layers.extend([
            nn.Conv3d(
                mid, mid, kernel_size, stride, padding=kernel_size // 2,
                groups=mid, bias=False
            ),
            nn.BatchNorm3d(mid),
            nn.SiLU(inplace=True),
            SqueezeExcite3D(mid, se_ratio, head_dropout=head_dropout),  # [REG]
            nn.Conv3d(mid, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
        ])

        self.block = nn.Sequential(*layers)
        self.drop_prob = drop_prob

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_path_prob, self.training)  # [REG]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_resid:
            y = self._drop_path(y)  # [REG]
            y = x + y
        return y


class HybridBlock(MBConv3D):
    """MBConv3D block followed by a real‑valued FFT↔IFFT cycle and dropout."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        drop_prob: float = 0.0,
        se_ratio: float = 0.25,
        fft_drop_prob: float = 0.0,
        head_dropout: float = 0.0,  # [REG]
        drop_path_prob: float = 0.0,  # [REG]
    ):
        super().__init__(in_ch, out_ch, stride, expand_ratio, kernel_size,
                         drop_prob, se_ratio, head_dropout=head_dropout, drop_path_prob=drop_path_prob)  # [REG]
        self.fft_dropout = nn.Dropout3d(fft_drop_prob) if fft_drop_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        # Apply FFT along spatial dimensions, then inverse FFT.
        y_fft = torch.fft.fftn(y, dim=(2, 3, 4))
        y_ifft = torch.fft.ifftn(y_fft, dim=(2, 3, 4))
        y_real = y_ifft.real
        y_real = self.fft_dropout(y_real)
        if self.use_resid:
            y_real = self._drop_path(y_real)  # [REG]
            y_real = x + y_real
        return y_real


# --------------------------------------------------------------------------- #
# Main Model
# --------------------------------------------------------------------------- #
class Image3DClfModel(nn.Module):
    """3‑D hybrid convolution‑frequency EfficientNet‑Lite style classifier."""
    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 12,
        drop_path_prob: float = 0.2,  # [REG]
        head_dropout: float = 0.05,  # [REG]
        fft_drop_prob: float = 0.1,  # [REG]
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.fft_drop_prob = fft_drop_prob  # [REG]

        # ---- Helper functions ----
        def round_filters(filters: int) -> int:
            """Round filters to nearest multiple of 8."""
            return int(math.ceil(filters / 8.0) * 8)

        # ---- Stem ----
        in_ch = round_filters(base_dim)
        self.stem = nn.Sequential(
            nn.Conv3d(1, in_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(in_ch),
            nn.SiLU(inplace=True),
        )

        # ---- Stage configuration ----
        stages_cfg = [
            (2, 1),
            (2, 2),
            (1, 4),
        ]

        # ---- Build blocks ----
        blocks = []
        total_blocks = model_depth
        base_repeats = [2, 3, 3]
        extra = max(0, total_blocks - sum(base_repeats))
        repeats = base_repeats[:]
        i = 0
        while extra > 0:
            repeats[i % 3] += 1
            i += 1
            extra -= 1

        in_ch = round_filters(base_dim)
        block_idx = 0
        for stage_idx, (stride, mult) in enumerate(stages_cfg):
            out_ch = round_filters(base_dim * mult)
            stage_drop_path_prob = (stage_idx + 1) / len(stages_cfg) * (0.3 * self.drop_path_prob)  # [REG]
            for r in range(repeats[stage_idx]):
                blk_stride = stride if r == 0 else 1
                drop_rate = 0.2 * block_idx / total_blocks
                blocks.append(
                    HybridBlock(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        stride=blk_stride,
                        expand_ratio=4,
                        kernel_size=3,
                        drop_prob=drop_rate,
                        se_ratio=0.25,
                        fft_drop_prob=self.fft_drop_prob,
                        head_dropout=self.head_dropout,  # [REG]
                        drop_path_prob=stage_drop_path_prob,  # [REG]
                    )
                )
                in_ch = out_ch
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

        # ---- Head ----
        head_ch = round_filters(base_dim * 8)
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm3d(head_ch),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(self.head_dropout),  # [REG]
        )

        # ---- Classifier ----
        self.classifier = nn.Linear(head_ch, label_num)

        # ---- Weight init ----
        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    # --------------------------------------------------------------------- #
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        label: torch.LongTensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        x = pixel_values
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, label)
        return {"logits": logits, "loss": loss}

    # --------------------------------------------------------------------- #
    @torch.inference_mode()
    def predict(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        x = pixel_values
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return {"logits": logits}