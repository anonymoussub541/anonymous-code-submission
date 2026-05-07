import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] needed for MixUp beta distribution


# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int,
                 drop_path_prob: float = 0.0):  # [REG] add drop_path_prob
        super().__init__()
        hidden_dim = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch
        self.drop_path_prob = drop_path_prob  # [REG] store

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                       nn.BatchNorm2d(hidden_dim),
                       nn.SiLU(inplace=True)]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim),
                   nn.SiLU(inplace=True),
                   nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + drop_path(self.block(x), self.drop_path_prob, self.training)  # [REG] apply drop_path here
        return self.block(x)


def _split_depth(total: int, parts: int) -> List[int]:
    base = total // parts
    remainder = total % parts
    depths = [base] * parts
    for i in range(remainder):
        depths[i] += 1
    return depths


class ImageClfModel(nn.Module):
    def __init__(self,
                 label_num: int,
                 base_dim: int = 32,
                 model_depth: int = 12,
                 *,
                 drop_path_prob: float = 0.2,  # [REG] new arg
                 head_dropout: float = 0.05,   # [REG] new arg
                 mixup_alpha: float = 0.2,     # [REG] new arg
                 use_mixup: bool = True        # [REG] new arg
                 ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth

        # [REG] store regularization params
        self.drop_path_prob = drop_path_prob
        self.head_dropout = head_dropout
        self.mixup_alpha = mixup_alpha
        self.use_mixup = use_mixup

        dims = [base_dim,
                base_dim * 2,
                base_dim * 4,
                base_dim * 4,
                base_dim * 6]

        stage_depths = _split_depth(model_depth, 5)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(inplace=True)
        )

        stages = []
        in_channels = base_dim
        strides = [1, 2, 2, 1, 2]

        for idx, (out_ch, depth, stride) in enumerate(zip(dims, stage_depths, strides)):
            stage_blocks = []
            # [REG] progressive drop_path per stage
            stage_drop_prob = self.drop_path_prob * (0.3 + 0.7 * (idx / (len(dims)-1)))
            for d in range(depth):
                s = stride if d == 0 else 1
                stage_blocks.append(InvertedResidual(in_channels, out_ch, s, expand_ratio=6,
                                                     drop_path_prob=stage_drop_prob))
                in_channels = out_ch
            stages.append(nn.Sequential(*stage_blocks))
        self.stages = nn.ModuleList(stages)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, max(128, in_channels), 1, bias=False),
            nn.BatchNorm2d(max(128, in_channels)),
            nn.SiLU(inplace=True),
            nn.Dropout(self.head_dropout)  # [REG] dropout in head
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(128, in_channels), label_num)

        self._init_weights()

        self.num_layers = sum(stage_depths) + 1

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

    # [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if (not self.use_mixup) or self.mixup_alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def forward(self,
                pixel_values: torch.FloatTensor,
                label: torch.LongTensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        if self.training and self.use_mixup:
            pixel_values, y_a, y_b, lam = self._mixup(pixel_values, label)
        else:
            y_a, y_b, lam = label, label, 1.0

        x = self.extract_features(pixel_values)
        x = self.pool(x).flatten(1)
        logits = self.fc(x)

        loss_a = F.cross_entropy(logits, y_a)
        loss_b = F.cross_entropy(logits, y_b)
        loss = lam * loss_a + (1 - lam) * loss_b

        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(self,
                pixel_values: torch.FloatTensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        x = self.extract_features(pixel_values)
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        return {"logits": logits}