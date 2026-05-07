import math
from typing import Any, Dict, List, Tuple
import numpy as np  # [REG] added for MixUp random beta distribution

import torch
import torch.nn as nn
import torch.nn.functional as F


# [REG] DropPath implementation
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def _split_depth(total: int, parts: int) -> List[int]:
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.size()
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class SelectiveKernelConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: List[int] = [3, 5, 7],
        stride: int = 1,
        groups: int = 2,
        reduce_factor: int = 4,
        shuffle_groups: int = 4,
        head_dropout: float = 0.05,  # [REG] added dropout rate
    ):
        super().__init__()
        self.shuffle_groups = shuffle_groups
        self.branches = nn.ModuleList()
        padding_list = [(k // 2) for k in kernel_sizes]
        for k, p in zip(kernel_sizes, padding_list):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=p, groups=groups, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )
        mid_ch = max(out_ch // reduce_factor, 4)
        self.fc_reduce = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),  # [REG] dropout after fc1 activation
        )
        self.fc_attn = nn.Conv2d(mid_ch, out_ch * len(kernel_sizes), 1, bias=True)
        self.kernel_sizes = len(kernel_sizes)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_feats = [branch(x) for branch in self.branches]
        branch_feats = [channel_shuffle(bf, self.shuffle_groups) for bf in branch_feats]
        U = sum(branch_feats)
        s = self.fc_reduce(U)
        attn = self.fc_attn(s)
        attn = attn.view(x.size(0), self.kernel_sizes, self.out_ch, 1, 1)
        attn = attn.softmax(dim=1)
        out = sum(attn[:, i] * branch_feats[i] for i in range(self.kernel_sizes))
        return out


class SKBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2, shuffle_groups: int = 4,
                 drop_path_prob: float = 0.0, block_id: int = 0, total_blocks: int = 1):  # [REG] added params
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )
        self.skconv = SelectiveKernelConv(mid_ch, mid_ch, stride=stride, groups=mid_ch, shuffle_groups=shuffle_groups)
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_res_connect = stride == 1 and in_ch == out_ch
        # [REG] progressive drop_path
        self.drop_path_prob = drop_path_prob * (block_id / max(1, total_blocks - 1)) if total_blocks > 1 else drop_path_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.skconv(out)
        out = self.project(out)
        if self.use_res_connect:
            out = drop_path(out, self.drop_path_prob, self.training)  # [REG] drop_path before residual add
            return x + out
        else:
            return out


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)


class ImageClfModel(nn.Module):
    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 15,
        shuffle_groups: int = 4,
        gem_p_init: float = 3.0,
        drop_path_prob: float = 0.2,  # [REG] added
        head_dropout: float = 0.05,   # [REG] added
        use_mixup: bool = True,       # [REG] added
        mixup_alpha: float = 0.2      # [REG] added
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.shuffle_groups = shuffle_groups
        self.drop_path_prob = drop_path_prob
        self.head_dropout = head_dropout
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(inplace=True),
        )

        stage_depths = _split_depth(model_depth, 3)
        dims = [base_dim, base_dim * 2, base_dim * 4]
        strides = [1, 2, 2]

        stages: List[nn.Module] = []
        in_ch = base_dim
        block_id_global = 0
        total_blocks_global = sum(stage_depths)
        for sd, out_ch, st in zip(stage_depths, dims, strides):
            blocks: List[nn.Module] = []
            blocks.append(SKBlock(in_ch, out_ch, stride=st, shuffle_groups=shuffle_groups,
                                   drop_path_prob=drop_path_prob, block_id=block_id_global,
                                   total_blocks=total_blocks_global))
            block_id_global += 1
            in_ch = out_ch
            for _ in range(sd - 1):
                blocks.append(SKBlock(in_ch, out_ch, stride=1, shuffle_groups=shuffle_groups,
                                       drop_path_prob=drop_path_prob, block_id=block_id_global,
                                       total_blocks=total_blocks_global))
                block_id_global += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.gem_pool = GeMPool2d(p=gem_p_init, eps=1e-6, learn_p=True)
        self.head_fc = nn.Sequential(  # [REG] added dropout before final fc
            nn.Dropout(self.head_dropout),
            nn.Linear(dims[-1], label_num)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def forward(
        self,
        *,
        pixel_values: torch.FloatTensor,
        label: torch.LongTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        if self.training and self.use_mixup:
            pixel_values, labels_a, labels_b, lam = self._mixup(pixel_values, label)
        else:
            labels_a, labels_b, lam = label, label, 1.0

        x = self.stem(pixel_values)
        for stage in self.stages:
            x = stage(x)
        x = self.gem_pool(x)
        x = torch.flatten(x, 1)
        logits = self.head_fc(x)
        if self.training and self.use_mixup:
            loss = lam * F.cross_entropy(logits, labels_a) + (1 - lam) * F.cross_entropy(logits, labels_b)
        else:
            loss = F.cross_entropy(logits, label)
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict(
        self,
        *,
        pixel_values: torch.FloatTensor,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(pixel_values)
        for stage in self.stages:
            x = stage(x)
        x = self.gem_pool(x)
        x = torch.flatten(x, 1)
        logits = self.head_fc(x)
        return {"logits": logits}