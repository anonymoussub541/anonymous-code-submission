import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] required for MixUp beta sampling
from typing import Any, Dict, List


def _split_depth(depth: int, parts: int) -> List[int]:
    base = depth // parts
    rem = depth - base * parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class BlurPool(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size == 3:
            filt = torch.tensor([1., 2., 1.], dtype=torch.float32)
        elif kernel_size == 5:
            filt = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        else:
            raise ValueError("Unsupported kernel")
        filt2d = filt[:, None] * filt[None, :]
        filt2d /= filt2d.sum()
        self.register_buffer("filt", filt2d[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self.pad(x), self.filt, stride=2, groups=x.shape[1])


class GhostModule(nn.Module):
    def __init__(self, in_c: int, out_c: int, ratio: int = 2, kernel_size: int = 1, dw_kernel_size: int = 3, act: bool = True, head_dropout: float = 0.0):  # [REG] add head_dropout
        super().__init__()
        init_c = int(out_c / ratio)
        new_c = out_c - init_c
        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_c, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_c),
            nn.SiLU() if act else nn.Identity()
        )
        self.cheap_conv = nn.Conv2d(init_c, new_c, dw_kernel_size, padding=dw_kernel_size // 2, groups=init_c, bias=False)
        self.cheap_bn = nn.BatchNorm2d(new_c)
        self.cheap_act = nn.SiLU() if act else nn.Identity()
        self.dropout = nn.Dropout2d(p=head_dropout)  # [REG] Dropout2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x1 = self.dropout(x1)  # [REG] apply dropout between fc-like layers
        x2 = self.cheap_conv(x1)
        x2 = self.cheap_bn(x2)
        x2 = self.cheap_act(x2)
        return torch.cat([x1, x2], dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, head_dropout: float = 0.0):  # [REG] head_dropout
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.dropout = nn.Dropout(p=head_dropout)  # [REG] dropout between fc1 and fc2
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = self.dropout(w)  # [REG]
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 4, dim, 1)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return shortcut + self.gamma[None, :, None, None] * x


class GhostNeXtBottleneck(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, se_reduction: int = 4, head_dropout: float = 0.0, drop_path_prob: float = 0.0):  # [REG]
        super().__init__()
        self.drop_path_prob = drop_path_prob  # [REG]
        self.ghost = GhostModule(in_c, out_c, act=True, head_dropout=head_dropout)  # [REG]
        self.down = BlurPool(out_c) if stride != 1 else nn.Identity()
        self.cnblock = ConvNeXtBlock(out_c)
        self.se = SEBlock(out_c, se_reduction, head_dropout=head_dropout)  # [REG]
        self.proj = nn.Conv2d(out_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            sc = []
            if stride != 1:
                sc.append(BlurPool(in_c))
            sc.extend([nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)])
            self.shortcut = nn.Sequential(*sc)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.ghost(x)
        x = self.down(x)
        # [REG] apply drop_path before fusion in multi-branch residual
        x = drop_path(self.cnblock(x), self.drop_path_prob, self.training)
        x = self.se(x)
        x = self.bn(self.proj(x))
        return x + self.shortcut(identity)


class MacroPyramidPooling(nn.Module):
    def __init__(self, in_c: int, pool_sizes: List[int], drop_path_prob: float = 0.0):  # [REG]
        super().__init__()
        self.drop_path_prob = drop_path_prob  # [REG]
        self.branches = nn.ModuleList()
        for size in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(in_c, in_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_c),
                    nn.SiLU()
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[2:]
        pooled = []
        for branch in self.branches:
            p = branch(x)
            if p.shape[2:] != out_size:
                p = F.interpolate(p, size=out_size, mode='bilinear', align_corners=False)
            p = drop_path(p, self.drop_path_prob, self.training)  # [REG]
            pooled.append(p)
        return torch.stack(pooled, dim=0).sum(dim=0)


class ImageClfModel(nn.Module):
    def __init__(self, label_num: int, base_dim: int = 32, model_depth: int = 12, *, drop_path_prob: float = 0.2, head_dropout: float = 0.05, mixup_alpha: float = 0.2, use_mixup: bool = True):  # [REG]
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.mixup_alpha = mixup_alpha  # [REG]
        self.use_mixup = use_mixup  # [REG]

        stage_depths = _split_depth(model_depth, 3)
        channels = [base_dim, base_dim * 2, base_dim * 4]

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU()
        )

        layers = []
        in_c = base_dim
        for stage_idx in range(3):
            depth = stage_depths[stage_idx]
            out_c = channels[stage_idx]
            stride = 2 if stage_idx > 0 else 1
            stage_drop_path_prob = self.drop_path_prob * (stage_idx + 1) / 3.0  # [REG] progressive
            for b_idx in range(depth):
                layers.append(GhostNeXtBottleneck(in_c, out_c, stride=stride if b_idx == 0 else 1, se_reduction=4, head_dropout=head_dropout, drop_path_prob=stage_drop_path_prob))
                in_c = out_c
        self.features = nn.Sequential(*layers)

        self.mpp = MacroPyramidPooling(channels[-1], pool_sizes=[1, 2, 4], drop_path_prob=self.drop_path_prob * 0.3)  # [REG]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], label_num)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij"
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        return torch.cat([x, coords], dim=1)

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
        *,
        pixel_values: torch.FloatTensor,
        label: torch.LongTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x = self._add_coord_channels(pixel_values)
        if self.training and self.use_mixup:  # [REG]
            x, y_a, y_b, lam = self._mixup(x, label)
        x = self.stem(x)
        x = self.features(x)
        x = self.mpp(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if self.training and self.use_mixup:  # [REG]
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
        else:
            loss = F.cross_entropy(logits, label)
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        *,
        pixel_values: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x = self._add_coord_channels(pixel_values)
        x = self.stem(x)
        x = self.features(x)
        x = self.mpp(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return {"logits": logits}