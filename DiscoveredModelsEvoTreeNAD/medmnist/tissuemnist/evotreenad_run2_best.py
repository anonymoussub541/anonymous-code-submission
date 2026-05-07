import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] for MixUp beta distribution
from typing import Any, Dict, List


# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def split_depth(total: int, n_stages: int = 4) -> List[int]:
    base = total // n_stages
    depths = [base] * n_stages
    for i in range(total - base * n_stages):
        depths[i % n_stages] += 1
    return depths


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, se_r: int = 8, head_dropout: float = 0.05):  # [REG] head_dropout arg
        super().__init__()
        squeezed = max(4, in_ch // se_r)
        self.fc1 = nn.Conv2d(in_ch, squeezed, 1, bias=False)
        self.fc2 = nn.Conv2d(squeezed, in_ch, 1, bias=False)
        self.mish = nn.Mish()
        self.dropout = nn.Dropout(head_dropout)  # [REG] dropout between fc1 and fc2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.mish(self.fc1(s))
        s = self.dropout(s)  # [REG] dropout applied
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expansion: int = 4, se_r: int = 8, head_dropout: float = 0.05):
        super().__init__()
        hidden_dim = in_ch * expansion
        self.use_residual = (stride == 1) and (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(),
            SqueezeExcite(hidden_dim, se_r=se_r, head_dropout=head_dropout),  # [REG]
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class InceptionMBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expansion: int = 4, se_r: int = 8,
                 drop_path_prob: float = 0.3, head_dropout: float = 0.05, stage_idx: int = 0):  # [REG]
        super().__init__()
        self.use_residual = (stride == 1) and (in_ch == out_ch)
        self.dp_prob = drop_path_prob * (0.3 + 0.2 * stage_idx)  # [REG] progressive scheduling
        # Path A: Standard MBConv
        self.path_a = MBConvBlock(in_ch, out_ch, stride, expansion, se_r, head_dropout=head_dropout)
        # Path B: 5x5 depthwise-sep + SE
        mid_ch_b = in_ch * expansion
        self.path_b = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch_b, 1, bias=False),
            nn.BatchNorm2d(mid_ch_b),
            nn.Mish(),
            nn.Conv2d(mid_ch_b, mid_ch_b, 5, stride, 2, groups=mid_ch_b, bias=False),
            nn.BatchNorm2d(mid_ch_b),
            nn.Mish(),
            SqueezeExcite(mid_ch_b, se_r=se_r, head_dropout=head_dropout),  # [REG]
            nn.Conv2d(mid_ch_b, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
        )
        # Path C: 3x3 avgpool + 1x1 conv
        self.path_c_pool = nn.AvgPool2d(3, stride=stride, padding=1)
        self.path_c_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
        )
        # Projection after concat
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = drop_path(self.path_a(x), self.dp_prob, self.training)  # [REG]
        out_b = drop_path(self.path_b(x), self.dp_prob, self.training)  # [REG]
        out_c = drop_path(self.path_c_proj(self.path_c_pool(x)), self.dp_prob, self.training)  # [REG]
        out = torch.cat([out_a, out_b, out_c], dim=1)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class ImageClfModel(nn.Module):
    def __init__(self, label_num: int, base_dim: int = 32, model_depth: int = 12, expansion: int = 4, se_r: int = 8,
                 drop_path_prob: float = 0.2, head_dropout: float = 0.05, mixup_alpha: float = 0.2, use_mixup: bool = True):  # [REG]
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.expansion = expansion
        self.se_r = se_r
        self.num_layers = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout      # [REG]
        self.mixup_alpha = mixup_alpha        # [REG]
        self.use_mixup = use_mixup            # [REG]
        assert base_dim % 2 == 0
        main_ch = base_dim // 2
        edge_ch = base_dim // 2
        self.stem_main = nn.Sequential(
            nn.Conv2d(1, main_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(main_ch),
            nn.Mish(),
        )
        edge_bank_size = 8
        sobel_h = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_v = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        filters = torch.stack([sobel_h, sobel_v]*4, dim=0) / 8.0
        self.edge_bank = nn.Conv2d(1, edge_bank_size, 3, 1, 1, bias=False)
        with torch.no_grad():
            self.edge_bank.weight.copy_(filters.unsqueeze(1))
        self.edge_bank.weight.requires_grad = True
        self.edge_proj = nn.Conv2d(edge_bank_size, edge_ch, 1, bias=False)
        self.edge_bn = nn.BatchNorm2d(edge_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, 3, 1, 1, groups=base_dim, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.Mish(),
            nn.Conv2d(base_dim, base_dim, 1, 1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.Mish(),
        )
        stage_out_channels = [base_dim, base_dim*2, base_dim*4, base_dim*8]
        stage_strides = [1, 2, 2, 2]
        stage_depths = split_depth(model_depth, 4)
        stages: List[nn.Module] = []
        in_ch = base_dim
        for stage_idx, (d, out_ch, stride) in enumerate(zip(stage_depths, stage_out_channels, stage_strides)):
            blocks: List[nn.Module] = []
            for i in range(d):
                s = stride if i == 0 else 1
                blocks.append(InceptionMBConvBlock(in_ch, out_ch, s, expansion, se_r,
                                                   drop_path_prob=drop_path_prob,
                                                   head_dropout=head_dropout,
                                                   stage_idx=stage_idx))  # [REG]
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.final_se = SqueezeExcite(stage_out_channels[-1], se_r=se_r, head_dropout=head_dropout)  # [REG]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(stage_out_channels[-1], label_num)
        self._initialize_weights()
    # [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0 or not self.use_mixup:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam
    def forward(self, *, pixel_values: torch.FloatTensor,  # [B,1,64,64] normalized input
                label: torch.LongTensor,  # [B] ground truth
                **kwargs: Any) -> Dict[str, torch.Tensor]:
        if self.training and self.use_mixup:  # [REG]
            pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)  # [REG]
        else:
            label_a, label_b, lam = label, label, 1.0  # [REG]
        main = self.stem_main(pixel_values)
        edge_map = self.edge_bank(pixel_values)
        edge_feat = self.edge_proj(edge_map)
        edge_feat = self.edge_bn(edge_feat)
        x = torch.cat([main, edge_feat], dim=1)
        x = self.fuse(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_se(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss_a = F.cross_entropy(logits, label_a)
        loss_b = F.cross_entropy(logits, label_b)
        loss = lam * loss_a + (1 - lam) * loss_b  # [REG]
        return {"logits": logits, "loss": loss}
    @torch.no_grad()
    def predict(self, *, pixel_values: torch.FloatTensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        main = self.stem_main(pixel_values)
        edge_map = self.edge_bank(pixel_values)
        edge_feat = self.edge_proj(edge_map)
        edge_feat = self.edge_bn(edge_feat)
        x = torch.cat([main, edge_feat], dim=1)
        x = self.fuse(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_se(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return {"logits": logits}
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.edge_bank:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02); nn.init.zeros_(m.bias)