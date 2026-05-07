import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np  # [REG] added numpy import
import random       # [REG] added random import


def split_depth(total_depth: int, stages: int) -> List[int]:
    base = total_depth // stages
    rem = total_depth - base * stages
    return [base + (1 if i < rem else 0) for i in range(stages)]


# [REG] DropPath utility
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class ECABlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, 1, C)
        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1)
        y = y.view(B, C, 1, 1)
        return x * y


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv_h = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_h = self.conv1(x_h)
        x_w = self.conv1(x_w)
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        a_h = torch.sigmoid(a_h)
        a_w = torch.sigmoid(a_w)
        return x * a_h * a_w


class BlurPool2d(nn.Module):
    def __init__(self, channels: int, stride: int = 2, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        kernel = torch.tensor([[[[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]]]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        weight = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer('weight', weight)
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, weight=self.weight, bias=None,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(B, C, H, W)


class ShuffleDepthwiseBottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        expansion: int = 4,
        use_attn: bool = False,
    ):
        super().__init__()
        mid_ch = in_ch * expansion
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups
        self.use_attn = use_attn

        self.pre_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1, groups=groups, bias=False)
        self.pre_bn = nn.BatchNorm2d(mid_ch)
        self.pre_act = nn.SiLU()

        self.dw_conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1,
                                 padding=1, groups=mid_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(mid_ch)
        self.dw_act = nn.SiLU()

        if use_attn:
            self.eca = ECABlock(mid_ch)
            self.ca = CoordinateAttention(mid_ch)

        self.out_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_ch)

        if stride == 2 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.proj = None

        self.blur_pool = BlurPool2d(out_ch, stride=stride) if stride == 2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.pre_conv(x)
        out = self.pre_bn(out)
        out = self.pre_act(out)

        if self.groups > 1:
            out = channel_shuffle(out, self.groups)

        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.dw_act(out)

        if self.use_attn:
            out = self.eca(out)
            out = self.ca(out)

        out = self.out_conv(out)
        out = self.out_bn(out)

        if self.blur_pool is not None:
            out = self.blur_pool(out)

        if self.proj is not None:
            residual = self.proj(x)

        return out + residual


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        return x + self.pe[:, :N, :]


class LightweightTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0, head_dropout: float = 0.05):  # [REG] added head_dropout
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(head_dropout),  # [REG] applied head_dropout between fc1 and fc2
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


class StageTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        h: int,
        w: int,
        transformer_heads: int = 2,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        head_dropout: float = 0.05  # [REG] propagate head_dropout
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.transformer = LightweightTransformerBlock(
            dim,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
            head_dropout=head_dropout  # [REG]
        )
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len=h * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_flat = self.pos_enc(x_flat)
        x_flat = self.transformer(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class DynamicRoutingBlock(nn.Module):
    def __init__(self, block_a: nn.Module, block_b: nn.Module, router_hidden: int, drop_path_prob: float = 0.2):  # [REG]
        super().__init__()
        self.block_a = block_a
        self.block_b = block_b
        self.router = nn.Sequential(
            nn.Linear(block_a.in_ch, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.drop_path_prob = drop_path_prob  # [REG]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pooled = x.mean(dim=[2, 3])  # [B, C]
        logits = self.router(pooled)  # [B, 2]
        weights = self.softmax(logits)  # [B, 2]
        out_a = drop_path(self.block_a(x), self.drop_path_prob, self.training)  # [REG]
        out_b = drop_path(self.block_b(x), self.drop_path_prob, self.training)  # [REG]
        w0 = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w1 = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return w0 * out_a + w1 * out_b


class ImageClfModel(nn.Module):
    def __init__(self, label_num: int, base_dim: int, model_depth: int,
                 drop_path_prob: float = 0.2, head_dropout: float = 0.05,  # [REG]
                 mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0, aug_prob: float = 0.9):  # [REG]
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout      # [REG]
        self.mixup_alpha = mixup_alpha        # [REG]
        self.cutmix_alpha = cutmix_alpha      # [REG]
        self.aug_prob = aug_prob              # [REG]

        embed_dim = base_dim * 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        stage_channels = [embed_dim, embed_dim * 2, embed_dim * 4]
        stages = 3
        depth_per_stage = split_depth(model_depth, stages)

        self.stages_blocks = nn.ModuleList()
        in_ch = embed_dim
        for stage_idx, (out_ch, num_pairs) in enumerate(zip(stage_channels, depth_per_stage)):
            blocks = nn.ModuleList()
            stage_drop_prob = self.drop_path_prob * (0.2 * (stage_idx + 1))  # [REG] progressive scheduling
            for pair_idx in range(num_pairs):
                stride = 2 if (stage_idx > 0 and pair_idx == 0) else 1
                groups = 1 if stage_idx == 0 else (2 if stage_idx == 1 else 4)
                block_a = ShuffleDepthwiseBottleneck(
                    in_ch, out_ch, stride=stride, groups=groups,
                    expansion=4, use_attn=False
                )
                block_b = ShuffleDepthwiseBottleneck(
                    in_ch, out_ch, stride=stride, groups=groups,
                    expansion=4, use_attn=True
                )
                router_hidden = max(1, in_ch // 4)
                dyn_block = DynamicRoutingBlock(block_a, block_b, router_hidden, drop_path_prob=stage_drop_prob)  # [REG]
                blocks.append(dyn_block)
                in_ch = out_ch
            self.stages_blocks.append(blocks)

        spatial_sizes = [32, 16, 8]
        self.stage_transformers = nn.ModuleList()
        for stage_idx, (out_ch, spatial) in enumerate(zip(stage_channels, spatial_sizes)):
            if stage_idx < 2:
                self.stage_transformers.append(
                    StageTransformer(
                        dim=out_ch,
                        h=spatial,
                        w=spatial,
                        transformer_heads=2,
                        transformer_mlp_ratio=4.0,
                        transformer_dropout=0.0,
                        head_dropout=self.head_dropout  # [REG]
                    )
                )

        self.stage_proj = nn.ModuleList()
        for in_c in stage_channels:
            if in_c != stage_channels[-1]:
                self.stage_proj.append(nn.Conv2d(in_c, stage_channels[-1], kernel_size=1, bias=False))
            else:
                self.stage_proj.append(nn.Identity())

        self.fuse_weight_self = nn.Parameter(torch.tensor(1.0))
        self.fuse_weight_prev = nn.Parameter(torch.tensor(0.5))
        self.fuse_weight_next = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Linear(stage_channels[-1], label_num)
        self.loss_fct = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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

    def forward(self, pixel_values: torch.FloatTensor, label: torch.LongTensor, **kwargs) -> Dict[str, torch.Tensor]:
        if self.training and random.random() < self.aug_prob:  # [REG]
            if random.random() < 0.5:
                pixel_values, label_a, label_b, lam = self._cutmix(pixel_values, label, self.cutmix_alpha)
            else:
                pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)
        else:
            label_a, label_b, lam = label, label, 1.0

        x = self.stem(pixel_values)
        stage_outputs = []
        for stage_idx, stage_blocks in enumerate(self.stages_blocks):
            for block in stage_blocks:
                x = block(x)
            if stage_idx < len(self.stage_transformers):
                x = self.stage_transformers[stage_idx](x)
            stage_outputs.append(x)

        projected = []
        for idx, feat in enumerate(stage_outputs):
            proj = self.stage_proj[idx](feat)
            if feat.shape[2] != 32 or feat.shape[3] != 32:
                proj = F.interpolate(proj, size=(32, 32), mode="bilinear", align_corners=False)
            projected.append(proj)

        fused = []
        num_stages = len(projected)
        for i in range(num_stages):
            cur = projected[i]
            f = self.fuse_weight_self * cur
            if i > 0:
                f += self.fuse_weight_prev * projected[i - 1]
            if i < num_stages - 1:
                f += self.fuse_weight_next * projected[i + 1]
            fused.append(f)

        pooled = [f.mean(dim=[2, 3]) for f in fused]
        x = torch.stack(pooled, dim=0).mean(dim=0)

        logits = self.classifier(x)
        loss_a = self.loss_fct(logits, label_a)  # [REG]
        loss_b = self.loss_fct(logits, label_b)  # [REG]
        loss = lam * loss_a + (1 - lam) * loss_b  # [REG]
        return {"logits": logits, "loss": loss}

    def predict(self, pixel_values: torch.FloatTensor, **kwargs) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            x = self.stem(pixel_values)
            stage_outputs = []
            for stage_idx, stage_blocks in enumerate(self.stages_blocks):
                for block in stage_blocks:
                    x = block(x)
                if stage_idx < len(self.stage_transformers):
                    x = self.stage_transformers[stage_idx](x)
                stage_outputs.append(x)

            projected = []
            for idx, feat in enumerate(stage_outputs):
                proj = self.stage_proj[idx](feat)
                if feat.shape[2] != 32 or feat.shape[3] != 32:
                    proj = F.interpolate(proj, size=(32, 32), mode="bilinear", align_corners=False)
                projected.append(proj)

            fused = []
            num_stages = len(projected)
            for i in range(num_stages):
                cur = projected[i]
                f = self.fuse_weight_self * cur
                if i > 0:
                    f += self.fuse_weight_prev * projected[i - 1]
                if i < num_stages - 1:
                    f += self.fuse_weight_next * projected[i + 1]
                fused.append(f)

            pooled = [f.mean(dim=[2, 3]) for f in fused]
            x = torch.stack(pooled, dim=0).mean(dim=0)
            logits = self.classifier(x)
        return {"logits": logits}