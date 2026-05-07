import math
import random  # [REG] for augmentation probability
import numpy as np  # [REG] required for augmentations
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# [REG] DropPath Implementation
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def _make_divisible(v: int, divisor: int = 8) -> int:
    return int(math.ceil(v / divisor) * divisor)

class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6

class BlurPool(nn.Module):
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        if filt_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise ValueError("Unsupported filt_size")
        kernel2d = kernel[:, None] * kernel[None, :]
        kernel2d = kernel2d / kernel2d.sum()
        self.register_buffer('kernel', kernel2d[None, None, :, :].repeat(channels, 1, 1, 1))
        self.stride = stride
        self.pad = (filt_size - 1) // 2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.pad, groups=x.shape[1])

class RCAB(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, head_dropout: float = 0.05):  # [REG] head_dropout arg
        super().__init__()
        reduced_ch = _make_divisible(channels // reduction, 1)
        self.attn_fc1 = nn.Conv2d(channels, reduced_ch, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(head_dropout)  # [REG] Dropout between fc1 and fc2
        self.attn_fc2 = nn.Conv2d(reduced_ch, channels, 1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        w = self.pool(x)
        w = self.attn_fc1(w)
        w = self.relu(w)
        w = self.dropout(w)  # [REG] Dropout applied
        w = self.attn_fc2(w)
        w = self.sigmoid(w)
        return res * w + res

class GLUExpand(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gate_type: str = "sigmoid"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        if gate_type == "sigmoid":
            self.gate_act = nn.Sigmoid()
        elif gate_type == "hswish":
            self.gate_act = HSwish()
        else:
            raise ValueError("Unsupported gate_type")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        val, gate = torch.chunk(x, 2, dim=1)
        return val * self.gate_act(gate)

class DepthwiseDeformConv(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size,
                                     kernel_size=3, stride=stride, padding=1)
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=in_ch, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w

class Res2NetRCABDeformSEBlock(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int,
                 expansion: int = 4, splits: int = 4, attn_reduction: int = 8,
                 blur_kernel_size: int = 3, bias_init: float = 0.0, gate_type: str = "sigmoid",
                 drop_path_prob: float = 0.2, head_dropout: float = 0.05):  # [REG] new args
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.drop_path_prob = drop_path_prob  # [REG]
        hidden_dim = inp * expansion
        self.splits = splits
        split_dim = hidden_dim // splits
        self.expand_glu = GLUExpand(inp, hidden_dim, gate_type=gate_type)
        self.hswish = HSwish()
        self.split_paths = nn.ModuleList()
        self.split_bns = nn.ModuleList()
        for _ in range(splits):
            if stride == 2:
                path = nn.Sequential(
                    BlurPool(split_dim, filt_size=blur_kernel_size, stride=2),
                    DepthwiseDeformConv(split_dim, kernel_size=3, stride=1, padding=1),
                    SEBlock(split_dim, reduction=4)
                )
            else:
                path = nn.Sequential(
                    DepthwiseDeformConv(split_dim, kernel_size=3, stride=1, padding=1),
                    SEBlock(split_dim, reduction=4)
                )
            self.split_paths.append(path)
            self.split_bns.append(nn.BatchNorm2d(split_dim))
        self.spatial_bias = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        nn.init.constant_(self.spatial_bias, bias_init)
        self.rcab = RCAB(hidden_dim, reduction=attn_reduction, head_dropout=head_dropout)  # [REG]
        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand_glu(x)
        splits = torch.chunk(out, self.splits, dim=1)
        processed = []
        for i, sp in enumerate(splits):
            y = self.split_paths[i](sp)
            y = self.split_bns[i](y)
            y = self.hswish(y)
            y = drop_path(y, self.drop_path_prob, self.training)  # [REG] drop_path per branch
            processed.append(y)
        out = torch.cat(processed, dim=1) + self.spatial_bias
        out = self.rcab(out)
        out = self.bn2(self.conv2(out))
        if self.use_res_connect:
            out = out + x
        return out

class ImageClfModel(nn.Module):
    def __init__(self, label_num: int, base_dim: int = 32, model_depth: int = 15,
                 expansion: int = 4, splits: int = 4, attn_reduction: int = 8,
                 blur_kernel_size: int = 3, bias_init: float = 0.0,
                 gate_type: str = "sigmoid", rotation_loss_weight: float = 0.15,
                 drop_path_prob: float = 0.2, head_dropout: float = 0.05,  # [REG] new args
                 aug_prob: float = 0.9, mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0):  # [REG] augmentation args
        super().__init__()
        self.label_num = label_num
        self.base_dim = base_dim
        self.model_depth = model_depth
        self.rotation_loss_weight = rotation_loss_weight
        self.drop_path_prob = drop_path_prob  # [REG]
        self.head_dropout = head_dropout  # [REG]
        self.aug_prob = aug_prob  # [REG]
        self.mixup_alpha = mixup_alpha  # [REG]
        self.cutmix_alpha = cutmix_alpha  # [REG]
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            HSwish(),
        )
        stage_depth1 = max(1, model_depth // 2)
        stage_depth2 = max(1, model_depth // 3)
        stage_depth3 = max(1, model_depth - stage_depth1 - stage_depth2)
        self.stage1 = self._make_stage(base_dim, base_dim*2, stage_depth1, first_stride=1,
                                       expansion=expansion, splits=splits,
                                       attn_reduction=attn_reduction, blur_kernel_size=blur_kernel_size,
                                       bias_init=bias_init, gate_type=gate_type,
                                       drop_path_prob=self.drop_path_prob*0.2, head_dropout=self.head_dropout)  # [REG] stage-specific drop_path
        self.stage2 = self._make_stage(base_dim*2, base_dim*4, stage_depth2, first_stride=2,
                                       expansion=expansion, splits=splits,
                                       attn_reduction=attn_reduction, blur_kernel_size=blur_kernel_size,
                                       bias_init=bias_init, gate_type=gate_type,
                                       drop_path_prob=self.drop_path_prob*0.6, head_dropout=self.head_dropout)
        self.stage3 = self._make_stage(base_dim*4, base_dim*8, stage_depth3, first_stride=2,
                                       expansion=expansion, splits=splits,
                                       attn_reduction=attn_reduction, blur_kernel_size=blur_kernel_size,
                                       bias_init=bias_init, gate_type=gate_type,
                                       drop_path_prob=self.drop_path_prob*1.0, head_dropout=self.head_dropout)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_dim*8, label_num)
        self.rot_dropout = nn.Dropout(0.2)
        self.rot_classifier = nn.Linear(base_dim*8, 4)
        self._initialize_weights()
    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, first_stride: int,
                    expansion: int, splits: int, attn_reduction: int, blur_kernel_size: int,
                    bias_init: float, gate_type: str, drop_path_prob: float, head_dropout: float) -> nn.ModuleList:
        mods: List[nn.Module] = []
        mods.append(Res2NetRCABDeformSEBlock(in_ch, out_ch, first_stride, expansion, splits,
                                             attn_reduction, blur_kernel_size, bias_init, gate_type,
                                             drop_path_prob=drop_path_prob, head_dropout=head_dropout))
        for _ in range(1, blocks):
            mods.append(Res2NetRCABDeformSEBlock(out_ch, out_ch, 1, expansion, splits,
                                                 attn_reduction, blur_kernel_size, bias_init, gate_type,
                                                 drop_path_prob=drop_path_prob, head_dropout=head_dropout))
        return nn.ModuleList(mods)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    # [REG] Augmentation: CutMix
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
    def _cutmix(self, images, labels, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(images.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam
    # [REG] Augmentation: MixUp
    def _mixup(self, x, y):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam
    def forward(self, *, pixel_values: torch.FloatTensor,
                label: torch.LongTensor,
                rotation_label: Optional[torch.LongTensor] = None,
                **kwargs: Any) -> Dict[str, torch.Tensor]:
        # [REG] Apply augmentation during training
        if self.training and random.random() < self.aug_prob:
            if random.random() < 0.5:
                pixel_values, label_a, label_b, lam = self._cutmix(pixel_values, label, self.cutmix_alpha)
            else:
                pixel_values, label_a, label_b, lam = self._mixup(pixel_values, label)
        else:
            label_a, label_b, lam = label, label, 1.0
        x = self.stem(pixel_values)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        spatial_feat = self.global_pool(x).flatten(1)
        logits = self.classifier(spatial_feat)
        loss_a = F.cross_entropy(logits, label_a)
        loss_b = F.cross_entropy(logits, label_b)
        loss = lam * loss_a + (1 - lam) * loss_b  # [REG] combined loss
        outputs: Dict[str, torch.Tensor] = {"logits": logits, "loss": loss}
        if rotation_label is not None:
            rot_feat = self.rot_dropout(spatial_feat)
            rot_logits = self.rot_classifier(rot_feat)
            rot_loss = F.cross_entropy(rot_logits, rotation_label)
            loss = loss + self.rotation_loss_weight * rot_loss
            outputs["rot_logits"] = rot_logits
            outputs["rot_loss"] = rot_loss
            outputs["loss"] = loss
        return outputs
    @torch.no_grad()
    def predict(self, *, pixel_values: torch.FloatTensor,
                **kwargs: Any) -> Dict[str, torch.Tensor]:
        self.eval()
        x = self.stem(pixel_values)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        spatial_feat = self.global_pool(x).flatten(1)
        logits = self.classifier(spatial_feat)
        return {"logits": logits}