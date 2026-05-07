import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# [REG] DropPath implementation
class DropPath(nn.Module):  # [REG] added DropPath
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class BlendNorm3d(nn.Module):
    """Blend of BatchNorm3d and InstanceNorm3d with a learnable gate per channel."""
    def __init__(self, num_channels: int, eps: float = 1e-5, momentum: float = 0.1,
                 blend_init: float = 0.5):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_channels, eps=eps, momentum=momentum, affine=True)
        self.inorm = nn.InstanceNorm3d(num_channels, eps=eps, momentum=momentum, affine=True)
        self.gate = nn.Parameter(torch.full((num_channels, 1, 1, 1), float(blend_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn_out = self.bn(x)
        in_out = self.inorm(x)
        g = torch.sigmoid(self.gate)          # [C,1,1,1]
        return g * bn_out + (1.0 - g) * in_out


class AntiAliasBlur3d(nn.Module):
    """Fixed 3×3×3 Gaussian blur applied per channel (depthwise convolution)."""
    def __init__(self, channels: int, kernel_size: int = 3, sigma: float = 1.0):
        super().__init__()
        k = kernel_size
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss[:, None, None] * gauss[None, :, None] * gauss[None, None, :]
        kernel = kernel.reshape(1, 1, k, k, k)
        self.register_buffer('kernel', kernel.repeat(channels, 1, 1, 1, 1))
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.kernel, padding=self.padding, groups=self.groups)


class DepthwiseConv3d(nn.Module):
    """3×3×3 depthwise convolution (groups == in_channels)."""
    def __init__(self, in_channels: int, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3,
                              stride=stride, padding=padding,
                              bias=False, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InvertedResidualBlock(nn.Module):
    """Depthwise → Pointwise residual block with BlendNorm3d."""
    def __init__(self, channels: int, drop_path_prob: float = 0.0, head_dropout: float = 0.0):  # [REG] added args
        super().__init__()
        self.dw = DepthwiseConv3d(channels)
        self.bn1 = BlendNorm3d(channels)
        self.prelu = nn.PReLU()
        # [REG] dropout after activation
        self.dropout = nn.Dropout3d(p=head_dropout)  # [REG] dropout3d
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = BlendNorm3d(channels)
        self.drop_path = DropPath(drop_path_prob)  # [REG] apply DropPath to branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)          # [B, C, D, H, W]
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.dropout(x)     # [REG] dropout
        x = self.pw(x)
        x = self.bn2(x)
        x = self.drop_path(x)   # [REG] DropPath on non-identity branch
        return x + residual  # residual connection


class DownsampleBlock(nn.Module):
    """Anti‑aliased down‑sampling: blur → stride‑2 conv."""
    def __init__(self, in_ch: int, out_ch: int,
                 blur_kernel_size: int = 3, blur_sigma: float = 1.0,
                 drop_path_prob: float = 0.0):  # [REG] added arg
        super().__init__()
        self.blur = AntiAliasBlur3d(in_ch, kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = BlendNorm3d(out_ch)
        self.drop_path = DropPath(drop_path_prob)  # [REG] DropPath

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blur(x)          # blur before down‑sampling
        x = self.conv(x)          # [B, out_ch, D/2, H/2, W/2]
        x = self.bn(x)
        x = self.drop_path(x)     # [REG] DropPath
        return x


class Image3DClfModel(nn.Module):
    """
    3‑D medical image binary classifier with efficient depthwise‑separable residual blocks
    and anti‑aliased down‑sampling.
    """
    def __init__(self,
                 base_dim: int = 32,
                 model_depth: int = 12,
                 blur_kernel_size: int = 3,
                 blur_sigma: float = 1.0,
                 drop_path_prob: float = 0.2,     # [REG] added arg
                 head_dropout: float = 0.05,      # [REG] added arg
                 augment_inputs: bool = False):   # [REG] added arg
        super().__init__()
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.drop_path_prob = drop_path_prob     # [REG]
        self.head_dropout = head_dropout         # [REG]
        self.augment_inputs = augment_inputs     # [REG]

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_dim, kernel_size=3, stride=1, padding=1, bias=False),
            BlendNorm3d(base_dim),
            nn.ReLU(inplace=True)
        )

        stage_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        stage_depths = self._split_depth(model_depth, num_stages=3)
        stages: List[nn.Module] = []
        in_ch = base_dim
        for stage_idx, (out_ch, blocks) in enumerate(zip(stage_channels[1:], stage_depths)):
            stage_drop_prob = drop_path_prob * 0.3 * (stage_idx + 1)  # [REG] progressive scheduling
            stages.append(DownsampleBlock(in_ch,
                                          out_ch,
                                          blur_kernel_size=self.blur_kernel_size,
                                          blur_sigma=self.blur_sigma,
                                          drop_path_prob=stage_drop_prob))  # [REG]
            for _ in range(blocks - 1):
                stages.append(InvertedResidualBlock(out_ch,
                                                    drop_path_prob=stage_drop_prob,
                                                    head_dropout=head_dropout))  # [REG]
            in_ch = out_ch
        self.features = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(p=head_dropout),  # [REG] dropout before linear
            nn.Linear(in_ch, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()
        self._init_weights()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        label: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        # [REG] optional input augmentation
        if self.augment_inputs and self.training:  # [REG]
            pixel_values = self._augment_inputs(pixel_values)  # [REG]
        x = self.stem(pixel_values)
        x = self.features(x)
        logits = self.head(x).squeeze(1)
        loss = self.loss_fn(logits, label)
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

    @staticmethod
    def _split_depth(total_depth: int, num_stages: int = 3) -> List[int]:
        base = total_depth // num_stages
        depths = [base] * num_stages
        for i in range(total_depth - base * num_stages):
            depths[i % num_stages] += 1
        return depths

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # [REG] simple augmentation method
    def _augment_inputs(self, x: torch.Tensor) -> torch.Tensor:  # [REG]
        if torch.rand(()) < 0.5:
            # random flip
            dims = [2, 3, 4]
            dim = dims[torch.randint(0, len(dims), (1,)).item()]
            x = torch.flip(x, dims=[dim])
        return x