import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

tempdir = '/home/DATA/'
from torch.utils.data import TensorDataset

# Define file paths to save preprocessed tensors
preprocessed_dir = '/home/DATA/preprocessed_evotreenad_fds1/' # the folder save the preprocessed data. Then, before the evolution, will load the saved preprocessed data in one time. all samples data for short-budget training will be in-memory, no transformer, no workers for efficiency. 
os.makedirs(preprocessed_dir, exist_ok=True)
train_tensor_path = os.path.join(preprocessed_dir, 'train_data.pt')
train_label_path = os.path.join(preprocessed_dir, 'train_labels.pt')
test_tensor_path = os.path.join(preprocessed_dir, 'test_data.pt')
test_label_path = os.path.join(preprocessed_dir, 'test_labels.pt')


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

# Preprocess and save only once
if not all(os.path.exists(p) for p in [train_tensor_path, train_label_path, test_tensor_path, test_label_path]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    # Download and transform CIFAR-10
    train_set_full = torchvision.datasets.CIFAR10(root=tempdir, train=True, transform=train_transform, download=True)
    set_no_transform = torchvision.datasets.CIFAR10(root=tempdir, train = True, transform=transform, download=True)
    seed = 42
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    num_train_samples = 42000
    random.seed(74364)

    # split train and val from train_set_full
    all_sample_size = len(train_set_full)  
    all_indices = list(range(all_sample_size))  
    random.shuffle(all_indices)  
    train_indices = all_indices[:num_train_samples]
    val_indices = all_indices[num_train_samples:]
    print('intersection between train and val indices (should be empty): ')
    print(set(train_indices).intersection(set(val_indices)))  # Should be empty set


    from torch.utils.data import Subset
    train_set = Subset(train_set_full, train_indices)
    test_set  = Subset(set_no_transform, val_indices)


    # Stack images and labels
    train_data = torch.stack([img for img, _ in train_set]+[img for img, _ in train_set])
    train_labels = torch.tensor([label for _, label in train_set]+[label for _, label in train_set])
    test_data = torch.stack([img for img, _ in test_set])
    test_labels = torch.tensor([label for _, label in test_set])

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

print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

# Wrap into TensorDatasets
train_dataset00 = TensorDataset(train_data, train_labels)
test_dataset00 = TensorDataset(test_data, test_labels)


from torch.utils.data import Dataset, DataLoader

class CIFAR10DictDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {"pixel_values": image, "label": label}

# Instantiate the custom dataset
train_dataset = CIFAR10DictDataset(train_dataset00)
test_dataset = CIFAR10DictDataset(test_dataset00)


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




# Evaluation function
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device=None):
    model.eval()
    totals = defaultdict(float)
    n_items = 0
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
        "eval_acc": round(correct / total, 5)
    }
    return final_dict


## config
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
batch_size = per_device_train_batch_size

train_loader = make_loader(train_dataset, per_device_train_batch_size, shuffle=True)
test_loader = make_loader(test_dataset, per_device_eval_batch_size, shuffle=False)
test_loader_small = make_loader(test_dataset, 2, shuffle=False)





print("Need to customize the config:")

EXPECT_METRIC_POINT = 0.8
LIMIT_METRIC = 0.92
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
            # best_score = np.mean(values[-2:])
            if np.isnan(best_score) or np.isinf(best_score):
                best_score = None
                return {'best_score': best_score}
        loss_values = [m['eval_loss'] for m in metrics if 'eval_loss' in m]
        if loss_values and len(loss_values)>=2:
            temp = np.array(loss_values)
            temp = temp[1:]-temp[:-1]
            temp = temp[temp>0]
            evalloss_bad = sum(temp)/10
            best_score = best_score - evalloss_bad
    return {'best_score': best_score}

print('use the best one as best score')

def process_eval_log(metrics: List[Dict]) -> List[Dict]:
    return metrics[-30:] if len(metrics) > 30 else metrics



init_paras_used ={'label_num': 10, 'base_dim': 32, 'model_depth': 15}
print("init_paras_used: ", init_paras_used)

task_specific_instruction = "Note: no pretrain, no finetune, model will be trained from scratch. Must use model_depth to allocate main blocks/layers depth. " 






