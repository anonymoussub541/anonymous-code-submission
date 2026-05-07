import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100.0)/batch_size)
    return res


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save_dir):
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))

def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(os.path.join(path))        
    print('Experiment dir : {}'.format(path))


import torch.nn as nn

class DropScheduler:
    def __init__(self, model, total_epochs):
        self.model = model
        self.total_epochs = total_epochs

        # attr_path -> (module, original_value)
        self.original_values = {}

        # Start collecting from model root
        self._collect(model, prefix="")

    def _collect(self, module, prefix):
        """
        Recursively walk through nn.Module hierarchy,
        recording all attributes containing 'drop' and numeric.
        """
        for name, value in module.__dict__.items():
            attr_path = f"{prefix}.{name}" if prefix else name

            # detect drop attributes
            if "drop" in name.lower() and isinstance(value, (float, int)):
                self.original_values[attr_path] = (module, value)

        # recurse into children modules
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._collect(child, child_prefix)

    def schedule(self, epoch):
        factor = min(max(epoch * 1.5 / self.total_epochs, 0.0), 1.0)

        for attr_path, (module, orig) in self.original_values.items():
            attr_name = attr_path.split(".")[-1]
            setattr(module, attr_name, orig * factor)

    def update(self, attr_path, value, update_original=True):
        if attr_path not in self.original_values:
            raise ValueError(f"{attr_path} is not a tracked drop attribute.")

        module, orig = self.original_values[attr_path]
        attr_name = attr_path.split(".")[-1]

        setattr(module, attr_name, value)

        if update_original:
            self.original_values[attr_path] = (module, value)

    def reset(self):
        """Restore all original drop values."""
        for attr_path, (module, orig) in self.original_values.items():
            attr_name = attr_path.split(".")[-1]
            setattr(module, attr_name, orig)



import numpy as np
import torch
import random


class Transform3DTrain:
    """
    Transform for 3D images with shape (1, D, H, W), typically used for MedMNIST 3D.
    Supports optional data augmentation and normalization.
    """

    def __init__(
        self,
        mul: float | None = None,                  
        to_tensor: bool = True,                    
        augment: bool = True,                      
        noise_std: float = 0.01,                   
        scale_range: tuple[float, float] = (0.95, 1.05),  
        normalize: bool = True,                   
    ):
        self.mul = mul
        self.to_tensor = to_tensor
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.normalize = normalize

    def random_intensity_scale(self, x: np.ndarray) -> np.ndarray:
        scale = random.uniform(*self.scale_range)
        return x * scale

    def random_noise(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
        return x + noise

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if sample.ndim != 4:
            raise ValueError("Expected shape (C, D, H, W)")

        x = sample.astype(np.float32)

        # Apply scalar multiplier if specified
        if self.mul is not None:
            x *= self.mul

        # Apply augmentation if enabled
        if self.augment:
            x = self.random_intensity_scale(x)
            x = self.random_noise(x)

        x = np.ascontiguousarray(x)

        # Normalize to [-1, 1] if enabled
        if self.normalize:
            x = (x - 0.5) / 0.5

        # Convert to PyTorch tensor
        if self.to_tensor:
            x = torch.from_numpy(x)
        return x

class Transform3DTest:
    """
    Deterministic transform for 3D validation/test images.
    Does not apply augmentation, only normalization and tensor conversion.
    """

    def __init__(
        self,
        mul: float | None = None,
        to_tensor: bool = True,
        normalize: bool = True,
    ):
        self.mul = mul
        self.to_tensor = to_tensor
        self.normalize = normalize

    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if sample.ndim != 4:
            raise ValueError("Expected shape (C, D, H, W)")

        x = sample.astype(np.float32)

        # Same normalization as training
        if self.mul is not None:
            x *= self.mul

        x = np.ascontiguousarray(x)

        if self.normalize:
            x = (x - 0.5) / 0.5

        if self.to_tensor:
            x = torch.from_numpy(x)

        return x


