import math
import random
from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.BatchNorm3d, nn.LayerNorm)):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


# [REG] DropPath implementation
class DropPath(nn.Module):  # [REG] 
    def __init__(self, drop_prob: float = 0.0):  # [REG] 
        super().__init__()  # [REG] 
        self.drop_prob = drop_prob  # [REG] 

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [REG] 
        if self.drop_prob == 0.0 or not self.training:  # [REG] 
            return x  # [REG] 
        keep_prob = 1 - self.drop_prob  # [REG] 
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [REG] 
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # [REG] 
        random_tensor.floor_()  # [REG] 
        return x.div(keep_prob) * random_tensor  # [REG] 


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, out_ch), out_ch)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_prob: float = 0.0, drop_path_prob: float = 0.0):  # [REG] added drop_path_prob
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv3d(out_ch, out_ch)
        self.drop = nn.Dropout3d(drop_prob) if drop_prob > 0 else nn.Identity()
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()  # [REG] DropPath
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(math.gcd(8, out_ch), out_ch)
            )
        else:
            self.shortcut = nn.Identity()
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.drop_path(x)  # [REG] Apply drop_path to main branch before adding residual
        return self.act(x + res)


class Downsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.norm = nn.GroupNorm(math.gcd(8, out_ch), out_ch)
        self.act = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Image3DClfModel(nn.Module):
    '''
    Summary & Reflections: DS-Conv residual encoder with input-level volume mixup for small-sample medical 3D classification.
    Input Utilisation: Forward applies online mixup to pixel_values/labels during train, then DS-Conv stem→residual encoder→GAP→linear classifier.
    Unchanged: Encoder macro-architecture from DS-Conv residual-based prior strong model retained.
    Upgrade vs. Previous: Added Mixup(alpha=0.4) in forward; target regularization and feature interpolation to address overfitting and improve generalization.
    '''
    def __init__(
        self,
        base_dim: int = 32,
        model_depth: int = 15,
        mixup_alpha: float = 0.2,
        drop_path_prob: float = 0.2,  # [REG] new arg
        head_dropout: float = 0.05    # [REG] new arg
    ):
        super().__init__()
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.mixup_alpha = mixup_alpha
        self.drop_path_prob = drop_path_prob  # [REG] store
        self.head_dropout = head_dropout      # [REG] store
        self.num_stages = 3

        self.stem = nn.Sequential(
            nn.Conv3d(1, base_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(math.gcd(8, base_dim), base_dim),
            GELU()
        )  # [B, base_dim, 32,32,32]

        stage_channels = [base_dim, base_dim * 2, base_dim * 4]
        stage_depths = self._split_depth(model_depth, self.num_stages)
        dp_inc = 0.05 / max(1, sum(stage_depths) - 1)

        drop_path_stage_probs = [0.3 * drop_path_prob, 0.6 * drop_path_prob, drop_path_prob]  # [REG] stage-wise probs

        blocks: List[nn.Module] = []
        in_ch = base_dim
        block_idx = 0
        for stage_idx in range(self.num_stages):
            if stage_idx > 0:
                blocks.append(Downsample3D(in_ch, stage_channels[stage_idx]))
                in_ch = stage_channels[stage_idx]
            for _ in range(stage_depths[stage_idx]):
                blocks.append(ResidualBlock3D(
                    in_ch,
                    stage_channels[stage_idx],
                    drop_prob=block_idx * dp_inc,
                    drop_path_prob=drop_path_stage_probs[stage_idx]  # [REG] pass stage-level prob
                ))
                block_idx += 1

        self.encoder = nn.Sequential(*blocks)
        self.head_norm = nn.GroupNorm(math.gcd(8, stage_channels[-1]), stage_channels[-1])
        self.head_dropout_layer = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()  # [REG] dropout before classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(stage_channels[-1], 1)

        self.apply(_init_weights)

    @staticmethod
    def _split_depth(total: int, num_stages: int) -> List[int]:
        base = total // num_stages
        depths = [base] * num_stages
        for i in range(total - base * num_stages):
            depths[i] += 1
        return depths

    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        x_mixed = lam * x + (1 - lam) * x[index, :]
        y_mixed = lam * y + (1 - lam) * y[index]
        return x_mixed, y_mixed

    def forward(
        self,
        pixel_values: torch.FloatTensor,  # [B,1,64,64,64] – normalized-preprocessed 3D medical image
        label: torch.FloatTensor,         # [B] – binary target label
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        if self.training and self.mixup_alpha > 0:
            pixel_values, label = self._mixup(pixel_values, label)
        x = self.stem(pixel_values)  # [B, base_dim, 32,32,32]
        x = self.encoder(x)          # [B, stage_channels[-1], D',H',W']
        x = self.head_norm(x)
        x = self.pool(x).flatten(1)  # [B, stage_channels[-1]]
        x = self.head_dropout_layer(x)  # [REG] dropout before classifier
        logits = self.classifier(x).squeeze(-1)  # [B]
        loss = F.binary_cross_entropy_with_logits(logits, label)
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.FloatTensor,  # [B,1,64,64,64] – normalized-preprocessed 3D medical image
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        x = self.stem(pixel_values)  # [B, base_dim, 32,32,32]
        x = self.encoder(x)          # [B, stage_channels[-1], D',H',W']
        x = self.head_norm(x)
        x = self.pool(x).flatten(1)  # [B, stage_channels[-1]]
        x = self.head_dropout_layer(x)  # [REG] dropout before classifier
        logits = self.classifier(x).squeeze(-1)  # [B]
        return {"logits": logits}