import math
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DepthwiseConv3D(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias, groups=in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SqueezeExcite3D(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25, head_dropout: float = 0.05):  # [REG] head_dropout arg added
        super().__init__()
        squeezed = max(1, int(channels * se_ratio))
        self.fc1 = nn.Conv3d(channels, squeezed, kernel_size=1)
        self.fc2 = nn.Conv3d(squeezed, channels, kernel_size=1)
        self.dropout = nn.Dropout(head_dropout)  # [REG] dropout between fc1 and fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool3d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = self.dropout(s)  # [REG] apply dropout after activation
        s = torch.sigmoid(self.fc2(s))
        return x * s


class EfficientResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_ratio: float = 0.25, expand_ratio: float = 1.5,
                 head_dropout: float = 0.05, drop_path_prob: float = 0.3):  # [REG] added args
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        mid_ch = int(in_ch * expand_ratio)

        self.pw_expand = PointwiseConv3D(in_ch, mid_ch, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=self._safe_group(mid_ch), num_channels=mid_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.dw = DepthwiseConv3D(mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=self._safe_group(mid_ch), num_channels=mid_ch)
        self.act2 = nn.ReLU(inplace=True)

        self.se = SqueezeExcite3D(mid_ch, se_ratio=se_ratio, head_dropout=head_dropout)  # [REG] pass head_dropout

        self.pw_reduce = PointwiseConv3D(mid_ch, out_ch, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=self._safe_group(out_ch), num_channels=out_ch)

        self.drop_path_prob = drop_path_prob  # [REG] store per-block drop_path prob

    @staticmethod
    def _safe_group(channels: int) -> int:
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_expand(x)
        out = self.gn1(out)
        out = self.act1(out)

        out = self.dw(out)
        out = self.gn2(out)
        out = self.act2(out)

        out = self.se(out)

        out = self.pw_reduce(out)
        out = self.gn3(out)
        if self.use_res_connect:
            out = drop_path(out, self.drop_path_prob, self.training) + x  # [REG] apply drop_path before residual addition
        return out


class Image3DClfModel(nn.Module):
    '''
    Summary & Reflections: Slim efficient 3D CNN using depthwise separable convolutions and SE; reduces computation, keeps salient 3D context learning.
    Input Utilisation: 3D volume → Conv stem → multi-stage depthwise-separable residual blocks → global pool → classifier.
    Unchanged: from scratch vs. prior.
    Upgrade vs. Previous: Replaced heavy/mixed convolutions with depthwise-separable blocks, trimmed channels/stages; expect faster training, lower params, robust accuracy.
    '''
    def __init__(self, base_dim: int = 32, model_depth: int = 12,
                 drop_path_prob: float = 0.2, head_dropout: float = 0.05):  # [REG] added args
        super().__init__()
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout      # [REG]

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=self._safe_group(base_dim), num_channels=base_dim),
            nn.ReLU(inplace=True),
        )

        stage_out_channels = [base_dim, base_dim * 2, base_dim * 3]
        base_repeats = [2, 3, 2]
        repeats = self._scale_repeats(base_repeats, model_depth)

        in_ch = base_dim
        stages = []
        for idx, (out_ch, n_repeat) in enumerate(zip(stage_out_channels, repeats)):
            stride = 1 if idx == 0 else 2
            stage_drop_prob = drop_path_prob * (0.3 + 0.7 * idx / (len(stage_out_channels)-1))  # [REG] progressive drop_path scheduling
            for i in range(n_repeat):
                s = stride if i == 0 else 1
                stages.append(EfficientResidualBlock3D(in_ch, out_ch, stride=s, se_ratio=0.25,
                                                       head_dropout=head_dropout,
                                                       drop_path_prob=stage_drop_prob))
                in_ch = out_ch
        self.features = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(in_ch, 1),
        )

        self._initialize_weights()

    @staticmethod
    def _safe_group(channels: int) -> int:
        for g in [8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    @staticmethod
    def _scale_repeats(base_repeats: List[int], target_depth: int) -> List[int]:
        base_total = sum(base_repeats)
        if target_depth == base_total:
            return base_repeats
        ratio = target_depth / base_total
        scaled = [max(1, round(r * ratio)) for r in base_repeats]
        diff = target_depth - sum(scaled)
        i = 0
        while diff != 0:
            scaled[i % len(scaled)] += 1 if diff > 0 else -1
            diff = target_depth - sum(scaled)
            i += 1
        return scaled

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        label: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        # [REG] Optional input augmentation (MixUp example)
        if self.training and torch.rand(1).item() < 0.1:  # [REG] 10% probability
            lam = torch.rand(1).item()
            batch_size = pixel_values.size(0)
            index = torch.randperm(batch_size)
            pixel_values = lam * pixel_values + (1 - lam) * pixel_values[index, :]
            label = lam * label + (1 - lam) * label[index]
        x = self.stem(pixel_values)
        x = self.features(x)
        logits = self.head(x).squeeze(1)
        loss = nn.BCEWithLogitsLoss()(logits, label.float())
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(pixel_values)
        x = self.features(x)
        logits = self.head(x).squeeze(1)
        return {"logits": logits}