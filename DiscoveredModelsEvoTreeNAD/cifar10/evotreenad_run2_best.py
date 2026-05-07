import math
import random  # [REG] for augmentation probability
import numpy as np  # [REG] for beta distribution and bbox calculations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_depth(total_depth: int, ratios):
    """Distribute total_depth among stages according to ratios, ensuring integer counts."""
    total = sum(ratios)
    parts = [max(1, int(math.floor(total_depth * r / total))) for r in ratios]
    diff = total_depth - sum(parts)
    idx = 0
    while diff > 0:
        parts[idx] += 1
        diff -= 1
        idx = (idx + 1) % len(parts)
    return parts

# [REG] Stochastic depth (DropPath) implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class ECABlock(nn.Module):
    """Efficient Channel Attention – 1‑D convolution on pooled features."""
    def __init__(self, channels: int, ksize: int = 3, head_dropout: float = 0.05):  # [REG] added head_dropout argument
        super().__init__()
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.ksize, padding=self.ksize // 2, bias=False)
        self.dropout = nn.Dropout(head_dropout)  # [REG] dropout in attention
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.dropout(y)  # [REG] dropout before sigmoid
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y


class MBConvBlock(nn.Module):
    """
    MBConv with ECA attention.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expansion: int = 4,
        se_ratio: float = 0.25,  # kept for API compatibility but not used
        head_dropout: float = 0.05,  # [REG] propagate head_dropout
    ):
        super().__init__()
        hidden = in_ch * expansion
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act1 = nn.SiLU()

        self.dw_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.SiLU()

        self.eca = ECABlock(hidden, ksize=3, head_dropout=head_dropout)  # [REG] pass head_dropout

        self.conv3 = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dw_conv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.eca(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x + residual


class FractalMBConvBlock(nn.Module):
    """
    Multi‑path block that stacks 1‑path and 2‑path MBConv sequences.
    The outputs are averaged to produce the final representation.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expansion: int = 4,
        se_ratio: float = 0.25,
        num_paths: int = 2,
        drop_path_prob: float = 0.2,  # [REG] added drop_path_prob
        head_dropout: float = 0.05,  # [REG] propagate head_dropout
    ):
        super().__init__()
        self.num_paths = num_paths
        self.paths = nn.ModuleList()
        self.drop_path_prob = drop_path_prob  # [REG] store for use
        for p in range(num_paths):
            if p == 0:
                block = MBConvBlock(
                    in_ch, out_ch, stride=stride, expansion=expansion, se_ratio=se_ratio,
                    head_dropout=head_dropout  # [REG]
                )
            else:
                block = nn.Sequential(
                    MBConvBlock(
                        in_ch, out_ch, stride=stride, expansion=expansion, se_ratio=se_ratio,
                        head_dropout=head_dropout  # [REG]
                    ),
                    MBConvBlock(
                        out_ch, out_ch, stride=1, expansion=expansion, se_ratio=se_ratio,
                        head_dropout=head_dropout  # [REG]
                    ),
                )
            self.paths.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for block in self.paths:
            y = block(x)
            y = drop_path(y, self.drop_path_prob, self.training)  # [REG] apply drop_path per branch
            out += y
        return out / self.num_paths


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Spatial Pooling (PSP) module that aggregates multi‑scale context.
    """
    def __init__(self, in_channels: int, pool_sizes=[1, 2, 4, 8]):
        super().__init__()
        inter_channels = max(1, in_channels // len(pool_sizes))
        self.stages = nn.ModuleList()
        for size in pool_sizes:
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            ))
        # Bottleneck to fuse concatenated features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels * len(pool_sizes), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        out = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            out.append(y)
        out = torch.cat(out, dim=1)
        out = self.bottleneck(out)
        return out


class ImageClfModel(nn.Module):
    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 15,
        drop_path_prob: float = 0.2,  # [REG] added arg
        head_dropout: float = 0.05,  # [REG] added arg
        aug_prob: float = 0.9,  # [REG] augmentation probability
        mixup_alpha: float = 1.0,  # [REG]
        cutmix_alpha: float = 1.0,  # [REG]
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.aug_prob = aug_prob  # [REG]
        self.mixup_alpha = mixup_alpha  # [REG]
        self.cutmix_alpha = cutmix_alpha  # [REG]

        # Stem: 3×3 conv → BN → SiLU
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(),
        )

        # Stage configuration: 3 stages with 3‑5‑7 blocks (depth distribution)
        stage_ratios = [3, 5, 7]
        stage_depths = split_depth(model_depth, stage_ratios)
        hidden_dims = [base_dim, base_dim * 2, base_dim * 4]  # 32, 64, 128

        self.stages = nn.ModuleList()
        in_ch = base_dim
        for i, (out_ch, depth) in enumerate(zip(hidden_dims[1:], stage_depths)):
            blocks = []
            stage_drop_prob = self.drop_path_prob * (i + 1) / len(stage_depths)  # [REG] progressive scheduling
            for j in range(depth):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(
                    FractalMBConvBlock(
                        in_ch,
                        out_ch,
                        stride=stride,
                        expansion=4,
                        se_ratio=0.25,
                        num_paths=2,
                        drop_path_prob=stage_drop_prob,  # [REG]
                        head_dropout=head_dropout,  # [REG]
                    )
                )
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        # Pyramid Pooling Bridge before global average pooling
        self.ppm = PyramidPoolingModule(in_channels=in_ch, pool_sizes=[1, 2, 4, 8])
        self.dropout = nn.Dropout(0.5)

        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, label_num)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    # [REG] CutMix augmentation
    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(images.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

    # [REG]
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

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
        pixel_values: torch.FloatTensor,  # [B, 3, 32, 32] – input images
        label: Optional[torch.LongTensor] = None,  # [B]
        **kwargs,
    ) -> dict:
        if label is not None and self.training and random.random() < self.aug_prob:  # [REG] apply augmentation
            if random.random() < 0.5:
                pixel_values, label_a, label_b, lam = self._cutmix(pixel_values, label, alpha=self.cutmix_alpha)
            else:
                pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)
        else:
            label_a, label_b, lam = label, label, 1.0

        x = self.stem(pixel_values)  # [B, base_dim, 32, 32]
        for stage in self.stages:
            x = stage(x)
        x = self.ppm(x)  # [B, in_ch, 32, 32]
        x = self.dropout(x)
        x = self.avg_pool(x)  # [B, C, 1, 1]
        x = torch.flatten(x, 1)  # [B, C]
        logits = self.fc(x)  # [B, label_num]
        out = {"logits": logits}
        if label is not None:
            loss_a = F.cross_entropy(logits, label_a)
            loss_b = F.cross_entropy(logits, label_b)
            loss = lam * loss_a + (1 - lam) * loss_b  # [REG] mix loss
            out["loss"] = loss
        return out

    def predict(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> dict:
        return {"logits": self.forward(pixel_values, **kwargs)["logits"]}