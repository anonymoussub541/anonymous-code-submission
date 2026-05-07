import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # [REG] import numpy
from typing import Any, Dict, List, Tuple


# [REG] DropPath implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def split_depth(total: int, ratios: Tuple[float, ...]) -> List[int]:
    raw = [max(1, int(round(r * total))) for r in ratios]
    diff = total - sum(raw)
    idx = 0
    while diff != 0:
        raw[idx % len(raw)] += 1 if diff > 0 else -1
        diff = total - sum(raw)
        idx += 1
    return raw


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, h, w = x.size()
    group_channels = channels // groups
    x = x.view(batch, groups, group_channels, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(batch, channels, h, w)


class ConvBnAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 8, head_dropout: float = 0.0):  # [REG] add head_dropout arg
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, 1, bias=True)
        self.dropout = nn.Dropout(head_dropout)  # [REG] dropout layer between fc1 and fc2
        self.fc2 = nn.Conv2d(in_ch // reduction, in_ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = self.dropout(s)  # [REG] apply dropout
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BlurPool(nn.Module):
    def __init__(self, in_ch: int, stride: int = 2):
        super().__init__()
        kernel = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ]
        ) / 16.0
        kernel = kernel.repeat(in_ch, 1, 1, 1)
        self.register_buffer("weight", kernel)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, weight=self.weight, stride=self.stride, padding=1, groups=x.size(1)
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion: int,
        use_se: bool = False,
        head_dropout: float = 0.0,  # [REG] head_dropout arg
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        hidden_dim = in_ch * expansion

        layers: List[nn.Module] = []

        layers.append(ConvBnAct(in_ch, hidden_dim, 1))
        layers.append(nn.Dropout2d(head_dropout))  # [REG] dropout after fc1 act

        if stride > 1:
            layers.append(BlurPool(hidden_dim, stride=stride))

        layers.append(ConvBnAct(hidden_dim, hidden_dim, 3, stride=1, groups=hidden_dim))

        if use_se:
            layers.append(SqueezeExcite(hidden_dim, head_dropout=head_dropout))

        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class InvertedResidualGate(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion: int,
        use_se: bool = False,
        channel_groups: int = 4,
        drop_path_prob: float = 0.0,  # [REG] add drop_path arg
        head_dropout: float = 0.0,    # [REG] head_dropout arg
    ):
        super().__init__()
        self.main = InvertedResidual(in_ch, out_ch, stride, expansion, use_se=use_se, head_dropout=head_dropout)
        self.attn_se = SqueezeExcite(in_ch, head_dropout=head_dropout)
        self.attn = nn.Sequential(
            BlurPool(in_ch, stride=stride) if stride > 1 else nn.Identity(),
            ConvBnAct(in_ch, out_ch, 1),
        )
        self.gate_fc = nn.Linear(in_ch, out_ch)
        self.use_res_connect = stride == 1 and in_ch == out_ch
        self.channel_groups = channel_groups
        self.drop_path_prob = drop_path_prob  # [REG]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        attn_out = self.attn(self.attn_se(x))
        gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        gate = torch.sigmoid(self.gate_fc(gap)).view(x.size(0), -1, 1, 1)
        out_main = drop_path(gate * main_out, self.drop_path_prob, self.training)  # [REG] apply drop_path
        out_attn = drop_path((1 - gate) * attn_out, self.drop_path_prob, self.training)  # [REG]
        out = out_main + out_attn
        if self.use_res_connect:
            out = out + x
        return out


class DCTStem(nn.Module):
    def __init__(self, base_dim: int):
        super().__init__()
        half_dim = base_dim // 2
        self.conv_rgb = ConvBnAct(3, half_dim, 3, stride=1)
        dct_1d = torch.zeros((8, 8))
        for k in range(8):
            for n in range(8):
                coeff = math.sqrt(1 / 8) if k == 0 else math.sqrt(2 / 8)
                dct_1d[k, n] = coeff * math.cos(math.pi * (n + 0.5) * k / 8)

        low_pairs = [
            (0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1),
            (0, 3), (3, 0), (1, 2), (2, 1),
        ]
        kernel_list = []
        for k, l in low_pairs:
            basis = torch.ger(dct_1d[k], dct_1d[l])
            kernel_list.append(basis)

        weight = torch.zeros((30, 1, 8, 8))
        for c in range(3):
            for idx, mat in enumerate(kernel_list):
                weight[10 * c + idx, 0] = mat

        self.conv_dct = nn.Conv2d(
            3, 30, kernel_size=8, stride=8, bias=False, groups=3
        )
        self.conv_dct.weight = nn.Parameter(weight, requires_grad=False)
        self.proj = ConvBnAct(30, half_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.conv_rgb(x)
        dct_feat = self.conv_dct(x)
        dct_feat = self.proj(dct_feat)
        dct_feat = F.interpolate(dct_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([rgb_feat, dct_feat], dim=1)


class SpectralMixerBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden: int, head_dropout: float = 0.0):  # [REG] head_dropout arg
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden = hidden
        self.mlp1 = nn.Conv2d(in_ch * 2, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(head_dropout)  # [REG]
        self.mlp2 = nn.Conv2d(hidden, out_ch, 1)
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_real = torch.real(x_fft)
        x_imag = torch.imag(x_fft)
        x_freq = torch.cat([x_real, x_imag], dim=1)
        out = self.act(self.mlp1(x_freq))
        out = self.dropout(out)  # [REG]
        out = self.mlp2(out)
        return out + self.proj(x)


class ImageClfModel(nn.Module):
    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 12,
        channel_groups: int = 4,
        drop_path_prob: float = 0.2,  # [REG]
        head_dropout: float = 0.05,   # [REG]
        mixup_alpha: float = 0.2      # [REG]
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.channel_groups = channel_groups
        self.drop_path_prob = drop_path_prob
        self.head_dropout = head_dropout
        self.mixup_alpha = mixup_alpha  # [REG]

        self.stem = DCTStem(base_dim)

        stage_out_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        stage_strides = [1, 2, 2, 2]
        stage_ratios = (0.2, 0.25, 0.25, 0.3)
        stage_depths = split_depth(model_depth, stage_ratios)
        drop_path_probs = [drop_path_prob * 0.3, drop_path_prob * 0.6, drop_path_prob * 0.9, drop_path_prob]  # [REG]
        self.stages = nn.ModuleList()
        in_ch = base_dim
        for depth, out_ch, stride, dp_prob in zip(stage_depths, stage_out_channels, stage_strides, drop_path_probs):
            blocks = []
            for i in range(depth):
                s = stride if i == 0 else 1
                use_se = out_ch >= base_dim * 4
                if out_ch == base_dim * 8 and s == 2:
                    blocks.append(
                        SpectralMixerBlock(
                            in_ch=in_ch,
                            out_ch=out_ch,
                            hidden=base_dim * 16,
                            head_dropout=head_dropout
                        )
                    )
                else:
                    blocks.append(
                        InvertedResidualGate(
                            in_ch=in_ch,
                            out_ch=out_ch,
                            stride=s,
                            expansion=4,
                            use_se=use_se,
                            channel_groups=channel_groups,
                            drop_path_prob=dp_prob,
                            head_dropout=head_dropout
                        )
                    )
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        self.fpn_conv = nn.Conv2d(sum(stage_out_channels), base_dim * 4, kernel_size=1, bias=False)
        self.bn_fpn = nn.BatchNorm2d(base_dim * 4)
        self.act_fpn = nn.Hardswish(inplace=True)
        self.se_fpn = SqueezeExcite(base_dim * 4, reduction=4, head_dropout=head_dropout)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(base_dim * 4, label_num)
        self.loss_fn = nn.CrossEntropyLoss()
        self._init_weights()

    # [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not m.weight.requires_grad:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *, pixel_values: torch.FloatTensor, label: torch.LongTensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        x = pixel_values
        if self.training:  # [REG] apply MixUp during training
            x, label_a, label_b, lam = self._mixup(x, label)
        else:
            label_a, label_b, lam = label, label, 1.0
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        target_size = features[0].shape[2:]
        up_features = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
        concat_feat = torch.cat(up_features, dim=1)
        x = self.act_fpn(self.bn_fpn(self.fpn_conv(concat_feat)))
        x = self.se_fpn(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        logits = self.classifier(x)
        if self.training:  # [REG] mixup loss
            loss = lam * self.loss_fn(logits, label_a) + (1 - lam) * self.loss_fn(logits, label_b)
        else:
            loss = self.loss_fn(logits, label)
        return {"logits": logits, "loss": loss}

    def predict(self, *, pixel_values: torch.FloatTensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        x = pixel_values
        x = self.stem(x)
        features: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        target_size = features[0].shape[2:]
        up_features = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
        concat_feat = torch.cat(up_features, dim=1)
        x = self.act_fpn(self.bn_fpn(self.fpn_conv(concat_feat)))
        x = self.se_fpn(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return {"logits": logits}