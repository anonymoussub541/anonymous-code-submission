import math
from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] Added for potential augmentation utils


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


class ChannelGate3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, head_dropout: float = 0.05) -> None:  # [REG] head_dropout arg
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, max(1, channels // reduction), kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(head_dropout)  # [REG] dropout between fc1 and fc2
        self.fc2 = nn.Conv3d(max(1, channels // reduction), channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.dropout(scale)  # [REG]
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dropout: float = 0.0, head_dropout: float = 0.05, drop_path_prob: float = 0.0) -> None:  # [REG] added head_dropout & drop_path_prob
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout)
        self.gate = ChannelGate3D(growth_rate, head_dropout=head_dropout)
        self.drop_path_prob = drop_path_prob  # [REG]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.bn(x)))
        out = self.dropout(out)
        out = self.gate(out)
        out = drop_path(out, self.drop_path_prob, self.training)  # [REG]
        return out


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, dropout: float = 0.0, head_dropout: float = 0.05, drop_path_prob: float = 0.0) -> None:  # [REG]
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(num_layers):
            # [REG] Apply progressive drop_path schedule inside block layers
            layer_drop_prob = drop_path_prob * ((i + 1) / num_layers)
            self.layers.append(DenseLayer3D(channels, growth_rate, dropout, head_dropout=head_dropout, drop_path_prob=layer_drop_prob))
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, 1))
            features.append(new_feat)
        return torch.cat(features, 1)


class Transition3D(nn.Module):
    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        out_channels = int(in_channels * compression)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class Image3DClfModel(nn.Module):
    def __init__(
        self,
        label_num: int = 11,
        base_dim: int = 32,
        model_depth: int = 12,
        growth_rate: int = None,
        transition_compress: float = 0.5,
        *,
        drop_path_prob: float = 0.2,  # [REG]
        head_dropout: float = 0.05,  # [REG]
        input_aug_prob: float = 0.2  # [REG]
    ) -> None:
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        if growth_rate is None:
            growth_rate = base_dim // 2
        self.growth_rate = growth_rate
        self.transition_compress = transition_compress
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.input_aug_prob = input_aug_prob  # [REG]

        blocks_per_stage = model_depth // 4

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(base_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        channels = base_dim
        self.stages = nn.ModuleList()
        for idx in range(4):
            block_drop_path_prob = drop_path_prob * ((idx + 1) / 4)  # [REG] progressive scheduling
            block = DenseBlock3D(blocks_per_stage, channels, growth_rate, dropout=0.1 + 0.05 * idx, head_dropout=head_dropout, drop_path_prob=block_drop_path_prob)  # [REG]
            channels = block.out_channels
            self.stages.append(block)
            if idx != 3:
                trans = Transition3D(channels, compression=transition_compress)
                channels = trans.out_channels
                self.stages.append(trans)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels, label_num)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    # [REG] Simple MixUp augmentation
    def _apply_input_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        if not self.training or self.input_aug_prob <= 0.0:
            return x, y
        if torch.rand(1).item() < self.input_aug_prob:
            lam = np.random.beta(1.0, 1.0)
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, (y_a, y_b, lam)
        return x, y

    def forward(
        self,
        *,
        pixel_values: torch.FloatTensor,
        label: torch.LongTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x, lbl = self._apply_input_augmentation(pixel_values, label)  # [REG]
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        pooled = self.global_pool(x)
        logits = self.fc(torch.flatten(pooled, 1))
        if isinstance(lbl, tuple):  # [REG] loss for MixUp
            y_a, y_b, lam = lbl
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
        else:
            loss = F.cross_entropy(logits, lbl)
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        *,
        pixel_values: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        x = pixel_values
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        pooled = self.global_pool(x)
        logits = self.fc(torch.flatten(pooled, 1))
        return {"logits": logits}