import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import medmnist
tempdir = '/home/DATA/'
from torch.utils.data import TensorDataset
from medmnist import INFO
data_flag = 'tissuemnist'


info = INFO[data_flag]
task, n_channels, n_classes = info['task'], info['n_channels'], len(info['label'])
is_3d = '3d' in data_flag
DataClass = getattr(medmnist, info['python_class'])


if data_flag == 'tissuemnist':
    MEAN = [0.1030]
    STD = [0.0986]
elif data_flag == 'pathmnist':
    MEAN = [0.7405, 0.5329, 0.7058]
    STD = [0.1404, 0.1952, 0.1388]
elif data_flag == 'octmnist':
    MEAN = [0.1894]
    STD = [0.2071]
else:
    MEAN = [0.5] * n_channels
    STD = [0.5] * n_channels

transform = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


    
# Define file paths to save preprocessed tensors
preprocessed_dir = f'/home/DATA/preprocessed_{data_flag}_ms/'
os.makedirs(preprocessed_dir, exist_ok=True)
train_tensor_path = os.path.join(preprocessed_dir, 'train_data.pt')
train_label_path = os.path.join(preprocessed_dir, 'train_labels.pt')
test_tensor_path = os.path.join(preprocessed_dir, 'test_data.pt')
test_label_path = os.path.join(preprocessed_dir, 'test_labels.pt')

train_set = DataClass(split='train', transform=None, download=True, root=tempdir, size = 64)

# Preprocess and save only once
if not all(os.path.exists(p) for p in [train_tensor_path, train_label_path, test_tensor_path, test_label_path]):
    train_set = DataClass(split='train', transform=transform, download=True, root=tempdir, size = 64)
    test_set = DataClass(split='val', transform=transform, download=True, root=tempdir, size = 64)
    # Stack images and labels
    train_data = torch.stack([img for img, _ in train_set])
    train_labels = torch.tensor([label for _, label in train_set]).squeeze()
    test_data = torch.stack([img for img, _ in test_set])
    test_labels = torch.tensor([label for _, label in test_set]).squeeze()
    print('Preprocessing done, saving tensors...')
    # Save to disk
    torch.save(train_data, train_tensor_path)
    torch.save(train_labels, train_label_path)
    torch.save(test_data, test_tensor_path)
    torch.save(test_labels, test_label_path)


# Load preprocessed tensors
train_data = torch.load(train_tensor_path)
train_labels = torch.load(train_label_path)
test_data = torch.load(test_tensor_path)
test_labels = torch.load(test_label_path)

print('Loaded preprocessed data:')
print('Train data shape:', train_data.shape)
print('Train labels shape:', train_labels.shape)
print('Test data shape:', test_data.shape)
print('Test labels shape:', test_labels.shape)
print('label num:', len(torch.unique(train_labels)))

# Wrap into TensorDatasets
train_dataset00 = TensorDataset(train_data, train_labels)
test_dataset00 = TensorDataset(test_data, test_labels)


from torch.utils.data import Dataset, DataLoader

class ImageDictDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {"pixel_values": image, "label": label}

# Instantiate the custom dataset
train_dataset = ImageDictDataset(train_dataset00)
test_dataset = ImageDictDataset(test_dataset00)


import torch.nn.functional as F
from typing import List, Dict

def make_loader(dataset, batch_size: int, shuffle: bool, drop_last = True, pin_memory = True, num_workers = 0, collate_fn = None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


from collections import defaultdict
import torch.nn as nn

# Evaluation function
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device=None):
    model.eval()
    totals = defaultdict(float)
    correct = 0
    total = 0
    ii = 0
    for x in loader:
        inputs = {k: v.to(device) for k, v in x.items() if k != 'label'}
        labels = x["label"].to(device)
        output = model.predict(**inputs)
        loss = F.cross_entropy(output["logits"], labels, reduction="sum")
        preds = output["logits"].argmax(dim=1)
        totals["loss"] += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        ii += 1
    final_dict = {
        "eval_loss": round(totals["loss"] / total, 5),
        "eval_acc": round(correct / total, 4)
    }
    return final_dict

per_device_train_batch_size = 32
per_device_eval_batch_size = 32
batch_size = per_device_train_batch_size

train_loader = make_loader(train_dataset, per_device_train_batch_size, shuffle=True)
test_loader = make_loader(test_dataset, per_device_eval_batch_size, shuffle=False)
test_loader_small = make_loader(test_dataset, 2, shuffle=False)


## config
print("Need to customize the config:")

EXPECT_METRIC_POINT = 0.6
LIMIT_METRIC = 0.75
key_metrics_name = "eval_acc"
print('EXPECT_METRIC_POINT: ', EXPECT_METRIC_POINT, 'LIMIT_METRIC', LIMIT_METRIC, 'key_metrics_name: ', key_metrics_name)


GREATER_IS_BETTER = LIMIT_METRIC>EXPECT_METRIC_POINT
if GREATER_IS_BETTER:
    print('greater is better')
else:
    print('smaller is better')
    
# User-defined required functions
def get_best_eval(metrics: List[Dict]):
    best_score = None
    if metrics:
        values = [m[key_metrics_name] for m in metrics if key_metrics_name in m]
        if values:
            best_score = max(values) if GREATER_IS_BETTER else min(values)
            if np.isnan(best_score) or np.isinf(best_score):
                best_score = None
                return {'best_score': best_score}
        loss_values = [m['eval_loss'] for m in metrics if 'eval_loss' in m]
        if loss_values and len(loss_values)>=2:
            temp = np.array(loss_values)
            temp = temp[1:]-temp[:-1]
            temp = temp[temp>0]
            evalloss_bad = 0.0
            if len(temp) > 0:
                evalloss_bad = sum(temp)/len(temp) * 0.05
            best_score = best_score - evalloss_bad
    return {'best_score': best_score}

print('use the best one as best score')

def process_eval_log(metrics: List[Dict]) -> List[Dict]:
    return metrics[-30:] if len(metrics) > 30 else metrics


init_paras_used ={'label_num': n_classes, 'base_dim': 32, 'model_depth': 12}
print("init_paras_used: ", init_paras_used)

task_specific_instruction = "Note: no pretrain, no finetune, model will be trained from scratch. Must use model_depth to allocate main blocks/layers depth. " 
