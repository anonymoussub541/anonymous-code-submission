import torch
import torch.nn as nn
import torch.nn.functional as F
import random  # [REG] for augmentation random selection
import numpy as np  # [REG] for augmentation
from typing import Dict, Tuple, List

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

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3, head_dropout: float = 0.0):  # [REG] head_dropout param
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(head_dropout)  # [REG] Dropout in attention head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = self.dropout(y)  # [REG] Apply dropout here
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y

class MixDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernels: Tuple[int, ...]):
        super().__init__()
        self.groups = len(kernels)
        splits = self._split_channels(channels, self.groups)
        self.convs = nn.ModuleList(
            [nn.Conv2d(splits[i], splits[i], kernels[i], padding=kernels[i] // 2, groups=splits[i], bias=False)
             for i in range(self.groups)]
        )
        self.channel_splits = splits

    @staticmethod
    def _split_channels(channels: int, num_groups: int) -> List[int]:
        base = channels // num_groups
        splits = [base] * num_groups
        for i in range(channels - base * num_groups):
            splits[i] += 1
        return splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, self.channel_splits, dim=1)
        ys = [conv(t) for conv, t in zip(self.convs, xs)]
        return torch.cat(ys, dim=1)

class BlurPool(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()
        assert kernel_size in [3, 5, 7], "Only supports 3,5,7 kernel"
        self.register_buffer("filt", self._build_filter(kernel_size))
        self.channels = channels
        self.stride = stride
        self.pad = kernel_size // 2

    @staticmethod
    def _build_filter(kernel_size: int) -> torch.Tensor:
        if kernel_size == 3:
            a = torch.tensor([1., 2., 1.])
        elif kernel_size == 5:
            a = torch.tensor([1., 4., 6., 4., 1.])
        else:
            a = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        filt2d = a[:, None] * a[None, :]
        return filt2d / torch.sum(filt2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = self.filt[None, None].repeat(self.channels, 1, 1, 1)
        return F.conv2d(x, filt, stride=self.stride, padding=self.pad, groups=self.channels)

class MixMBConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion: int,
        kernels: Tuple[int, ...] = (3, 5, 7),
        blur_kernel: int = 5,
        layerscale_init: float = 1e-6,
        eca_kernel: int = 3,
        head_dropout: float = 0.0,  # [REG]
        drop_path_prob: float = 0.0  # [REG]
    ):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        self.drop_path_prob = drop_path_prob  # [REG]
        mid_ch = in_ch * expansion
        layers = []
        if expansion != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.SiLU()]
        layers += [MixDepthwiseConv(mid_ch, kernels), nn.BatchNorm2d(mid_ch), nn.SiLU()]
        if stride == 2:
            layers.append(BlurPool(mid_ch, kernel_size=blur_kernel, stride=stride))
            stride = 1
        layers.append(EfficientChannelAttention(mid_ch, k_size=eca_kernel, head_dropout=head_dropout))  # [REG]
        layers += [nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        self.gamma = nn.Parameter(layerscale_init * torch.ones((out_ch)), requires_grad=True) if self.use_res else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        if self.use_res:
            h = self.gamma.view(1, -1, 1, 1) * h
            h = drop_path(h, self.drop_path_prob, self.training)  # [REG] DropPath in residual
            h = h + x
        return h

class FFTContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_freq = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        fft = torch.fft.rfft2(x, norm="ortho")
        real = fft.real
        imag = fft.imag
        fft_ri = torch.cat([real, imag], dim=1)
        y_freq = self.conv_freq(fft_ri)
        y = torch.fft.irfft2(torch.complex(y_freq, torch.zeros_like(y_freq)), s=(H, W), norm="ortho")
        return x + self.gate * y

class ODEFunc(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.bn2(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, steps: int = 6, step_size: float = 1.0/6):
        super().__init__()
        self.odefunc = odefunc
        self.steps = steps
        self.h = step_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tensor(0.0, device=x.device)
        for _ in range(self.steps):
            x = x + self.h * self.odefunc(t, x)
            t = t + self.h
        return x

class ImageClfModel(nn.Module):
    def __init__(
        self,
        label_num: int,
        base_dim: int = 32,
        model_depth: int = 18,
        blur_kernel: int = 5,
        channel_multiplier: float = 1.0,
        label_smooth: float = 0.1,
        dropout: float = 0.3,
        layerscale_init: float = 1e-6,
        eca_kernel: int = 3,
        drop_path_prob: float = 0.3,  # [REG]
        head_dropout: float = 0.05,  # [REG]
        aug_prob: float = 0.9,  # [REG]
        mixup_alpha: float = 1.0,  # [REG]
        cutmix_alpha: float = 1.0   # [REG]
    ):
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.num_stages = 3
        self.blur_kernel = blur_kernel
        self.channel_multiplier = channel_multiplier
        self.label_smooth = label_smooth
        self.dropout_rate = dropout
        self.eca_kernel = eca_kernel
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.aug_prob = aug_prob  # [REG]
        self.mixup_alpha = mixup_alpha  # [REG]
        self.cutmix_alpha = cutmix_alpha  # [REG]

        stem_out = int(base_dim * channel_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.SiLU(),
        )

        stage_depths = self._split_depth(model_depth, self.num_stages)
        stage_channels = [
            int(base_dim * 4 * channel_multiplier),
            int(base_dim * 8 * channel_multiplier),
            int(base_dim * 12 * channel_multiplier),
        ]
        self.stage1_out, self.stage2_out, self.stage3_out = stage_channels

        self.stage1 = nn.Sequential(*[
            MixMBConv(
                in_ch=stem_out if b == 0 else self.stage1_out,
                out_ch=self.stage1_out,
                stride=1,
                expansion=2,
                kernels=(3, 5, 7),
                blur_kernel=self.blur_kernel,
                layerscale_init=layerscale_init,
                eca_kernel=self.eca_kernel,
                head_dropout=self.head_dropout,  # [REG]
                drop_path_prob=self.drop_path_prob * 0.3  # [REG]
            ) for b in range(stage_depths[0])
        ])
        self.fft_block = FFTContextBlock(self.stage1_out)

        stage2_blocks = []
        for b in range(stage_depths[1]):
            if b == stage_depths[1] // 2:
                stage2_blocks.append(ODEBlock(ODEFunc(self.stage2_out), steps=6))
            else:
                stride = 2 if b == 0 else 1
                in_ch = self.stage1_out if b == 0 else self.stage2_out
                stage2_blocks.append(
                    MixMBConv(
                        in_ch=in_ch,
                        out_ch=self.stage2_out,
                        stride=stride,
                        expansion=3,
                        kernels=(3, 5, 7),
                        blur_kernel=self.blur_kernel,
                        layerscale_init=layerscale_init,
                        eca_kernel=self.eca_kernel,
                        head_dropout=self.head_dropout,  # [REG]
                        drop_path_prob=self.drop_path_prob * 0.6  # [REG]
                    )
                )
        self.stage2 = nn.Sequential(*stage2_blocks)

        self.stage3 = nn.Sequential(*[
            MixMBConv(
                in_ch=self.stage2_out if b == 0 else self.stage3_out,
                out_ch=self.stage3_out,
                stride=2 if b == 0 else 1,
                expansion=4,
                kernels=(3, 5, 7),
                blur_kernel=self.blur_kernel,
                layerscale_init=layerscale_init,
                eca_kernel=self.eca_kernel,
                head_dropout=self.head_dropout,  # [REG]
                drop_path_prob=self.drop_path_prob * 0.9  # [REG]
            ) for b in range(stage_depths[2])
        ])

        self.proj2 = nn.Conv2d(self.stage1_out + stem_out, self.stage1_out, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(self.stage2_out + self.stage1_out + stem_out, self.stage2_out, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.stage3_out, label_num))

        self._initialize_weights()

    @staticmethod
    def _split_depth(total: int, parts: int) -> List[int]:
        base = total // parts
        arr = [base] * parts
        for i in range(total - base * parts):
            arr[i] += 1
        return arr

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def forward(
        self,
        *,
        pixel_values: torch.FloatTensor,
        label: torch.LongTensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.training and random.random() < self.aug_prob:  # [REG] apply augmentation
            if random.random() < 0.5:
                pixel_values, label_a, label_b, lam = self._cutmix(pixel_values, label, self.cutmix_alpha)
            else:
                pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)
        else:
            label_a, label_b, lam = label, label, 1.0

        x0 = self.stem(pixel_values)
        x1 = self.stage1(x0)
        x1_fft = self.fft_block(x1)
        x1_down = F.avg_pool2d(x1_fft, 2)
        x0_down = F.avg_pool2d(x0, 2)

        x2_in = torch.cat([x1_fft, x0], dim=1)
        x2_in = self.proj2(x2_in)
        x2 = self.stage2(x2_in)

        x3_in = torch.cat([x2, x1_down, x0_down], dim=1)
        x3_in = self.proj3(x3_in)
        x3 = self.stage3(x3_in)

        x = self.dropout(x3)
        logits = self.head(x)
        loss_a = F.cross_entropy(logits, label_a, label_smoothing=self.label_smooth)
        loss_b = F.cross_entropy(logits, label_b, label_smoothing=self.label_smooth)
        loss = lam * loss_a + (1 - lam) * loss_b  # [REG] augmented loss
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def predict(
        self,
        *,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        x0 = self.stem(pixel_values)
        x1 = self.stage1(x0)
        x1_fft = self.fft_block(x1)
        x1_down = F.avg_pool2d(x1_fft, 2)
        x0_down = F.avg_pool2d(x0, 2)

        x2_in = torch.cat([x1_fft, x0], dim=1)
        x2_in = self.proj2(x2_in)
        x2 = self.stage2(x2_in)

        x3_in = torch.cat([x2, x1_down, x0_down], dim=1)
        x3_in = self.proj3(x3_in)
        x3 = self.stage3(x3_in)

        x = self.dropout(x3)
        logits = self.head(x)
        return {"logits": logits}