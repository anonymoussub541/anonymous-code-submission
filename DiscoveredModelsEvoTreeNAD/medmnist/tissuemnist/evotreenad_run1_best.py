import math
from typing import Any, Dict, List
import numpy as np  # [REG] Added for MixUp

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: int, divisor: int = 8) -> int:
    """Round up `v` to the nearest multiple of `divisor`."""
    return int(math.ceil(v / divisor) * divisor)


def _compute_blocks_per_stage(model_depth: int, ratios: List[int]) -> List[int]:
    """Distribute total blocks across stages proportionally to `ratios`."""
    total_ratio = sum(ratios)
    blocks = [max(1, model_depth * r // total_ratio) for r in ratios]
    remainder = model_depth - sum(blocks)
    for i in range(remainder):
        blocks[i] += 1
    return blocks

# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class SEBlock(nn.Module):
    """Squeeze‑Excitation block for channel‑wise attention."""

    def __init__(self, channels: int, reduction: int = 8, head_dropout: float = 0.0):  # [REG] head_dropout arg
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=head_dropout)  # [REG] Dropout between fc1 and fc2
        self.fc2 = nn.Linear(reduced, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)  # [REG] Dropout applied
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualBlock(nn.Module):
    """
    MobileNet‑V2 style inverted residual block with optional SE.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int = 1,
        expansion: int = 4,
        use_se: bool = True,
        head_dropout: float = 0.0,     # [REG] head_dropout propagation
        drop_path_prob: float = 0.0    # [REG] drop_path_prob arg
    ):
        super().__init__()
        hidden = in_c * expansion
        self.use_residual = stride == 1 and in_c == out_c
        self.drop_path_prob = drop_path_prob  # [REG] store drop_path_prob

        # Pointwise expansion
        self.conv_pw1 = nn.Conv2d(in_c, hidden, 1, bias=False)
        self.bn_pw1 = nn.BatchNorm2d(hidden)
        self.silu_pw1 = nn.SiLU(inplace=True)

        # Depthwise convolution
        self.conv_dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(hidden)
        self.silu_dw = nn.SiLU(inplace=True)

        # Pointwise projection
        self.conv_pw2 = nn.Conv2d(hidden, out_c, 1, bias=False)
        self.bn_pw2 = nn.BatchNorm2d(out_c)
        self.silu_pw2 = nn.SiLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_c, head_dropout=head_dropout)  # [REG] pass head_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_pw1(x)
        out = self.bn_pw1(out)
        out = self.silu_pw1(out)

        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = self.silu_dw(out)

        out = self.conv_pw2(out)
        out = self.bn_pw2(out)
        out = self.silu_pw2(out)

        if self.use_se:
            out = self.se(out)

        if self.use_residual:
            out = drop_path(out, self.drop_path_prob, self.training) + residual  # [REG] DropPath before fusion
        else:
            return out
        return out


class ImageClfModel(nn.Module):
    """
    Image classification model complying with the specified interface and constraints.
    """

    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 12,
        *,
        drop_path_prob: float = 0.2,   # [REG] new arg
        head_dropout: float = 0.05,    # [REG] new arg
        mixup_alpha: float = 0.2        # [REG] new arg
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG] store
        self.head_dropout = head_dropout      # [REG] store
        self.mixup_alpha = mixup_alpha        # [REG] store

        # Stage depth allocation ratios (sum = 6)
        ratios = [1, 2, 2, 1]
        blocks_per_stage = _compute_blocks_per_stage(model_depth, ratios)

        # Stem block
        first_channels = _make_divisible(base_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                first_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(first_channels),
            nn.SiLU(inplace=True),
        )

        # Channel dimensions for each stage
        stage_channels = [
            _make_divisible(base_dim),
            _make_divisible(base_dim * 2),
            _make_divisible(base_dim * 4),
            _make_divisible(base_dim * 6),
        ]

        # Build hierarchical stages
        stages: List[nn.Module] = []
        in_c = stage_channels[0]
        for stage_idx, (num_blocks, out_c) in enumerate(
            zip(blocks_per_stage, stage_channels)
        ):
            stage_drop_path_prob = self.drop_path_prob * (stage_idx + 1) / len(stage_channels)  # [REG] progressive schedule
            for blk_idx in range(num_blocks):
                stride = 2 if stage_idx > 0 and blk_idx == 0 else 1
                stages.append(
                    InvertedResidualBlock(
                        in_c,
                        out_c,
                        stride=stride,
                        expansion=4,
                        use_se=True,
                        head_dropout=self.head_dropout,        # [REG] pass head_dropout
                        drop_path_prob=stage_drop_path_prob    # [REG] pass drop_path_prob
                    )
                )
                in_c = out_c  # update input channels for next block

        self.features = nn.Sequential(*stages)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # -> [B, C_last, 1, 1]
        self.head = nn.Linear(in_c, label_num)   # -> [B, label_num]

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def forward(
        self,
        pixel_values: torch.FloatTensor,  # [B, 1, 64, 64] – normalized grayscale image
        label: torch.LongTensor,          # [B] – ground‑truth class indices
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        """
        # [REG] Apply MixUp during training
        if self.training:
            pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)  # [REG] MixUp
        else:
            label_a, label_b, lam = label, label, 1.0

        x = self.stem(pixel_values)                 # -> [B, C1, 64, 64]
        x = self.features(x)                        # -> [B, C_last, 8, 8]
        x = self.avg_pool(x).flatten(1)             # -> [B, C_last]
        logits = self.head(x)                       # -> [B, label_num]
        loss_a = F.cross_entropy(logits, label_a)
        loss_b = F.cross_entropy(logits, label_b)
        loss = lam * loss_a + (1 - lam) * loss_b     # [REG] MixUp loss combination
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        pixel_values: torch.FloatTensor,  # [B, 1, 64, 64]
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for inference.
        """
        x = self.stem(pixel_values)                 # -> [B, C1, 64, 64]
        x = self.features(x)                        # -> [B, C_last, 8, 8]
        x = self.avg_pool(x).flatten(1)             # -> [B, C_last]
        logits = self.head(x)                       # -> [B, label_num]
        return {"logits": logits}