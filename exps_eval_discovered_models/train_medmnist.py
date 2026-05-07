import argparse
import os
import numpy as np
import sys
sys.path.append('/home/')
sys.path.append('../')

SAVE_DIR = './'



parser = argparse.ArgumentParser("medmnist_train")
parser.add_argument('--modelclass', type=str, default='ImageClfModel', help='Model class name')
parser.add_argument('--modulepath', type=str, default='discoveredmodel.model1', help='Full python import path of model module')
parser.add_argument('--modelname', type=str, default='modelname', help='Model name')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--data', type=str, default='/home/DATA/', help='data corpus path')
parser.add_argument('--data_flag', type=str, default='pathmnist', help='which dataset to use')
parser.add_argument('--model_dim', type=int, default=32, help='model dim')
parser.add_argument('--model_depth', type=int, default=12, help='model depth')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--eval_log_ep', type=int, default=1, help='eval report frequency')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
data_flag = args.data_flag

import medmnist
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
from medmnist import INFO, Evaluator
import torch.nn as nn
from exp_util import Transform3DTrain, Transform3DTest, DropScheduler, count_parameters_in_MB, create_exp_dir

info = INFO[data_flag]
print(info)
task, n_channels, n_classes = info['task'], info['n_channels'], len(info['label'])
label_num = n_classes
is_3d = '3d' in data_flag
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
if is_3d:
    train_transform = Transform3DTrain()
    valid_transform = Transform3DTest()
    print('use 3d transformers')
else:
    if n_channels == 1:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

DataClass = getattr(medmnist, info['python_class'])
train_data = DataClass(split='train', transform=train_transform, download=True, root=args.data, size = 64)
test_data = DataClass(split='test', transform=valid_transform, download=True, root=args.data, size = 64)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


import importlib
try:
    model_module = importlib.import_module(args.modulepath)
    ImageClfModel = getattr(model_module, args.modelclass)
except Exception as e:
    print('Error in importing model:', e)
    raise e

import logging
import time

modelname0 = args.modelname
savedir = os.path.join(SAVE_DIR, './{}/{}-{}'.format(data_flag, modelname0, time.strftime("%Y%m%d-%H%M%S")))
create_exp_dir(savedir)


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(savedir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)




def train_and_evaluate(model, train_loader, test_loader, epochs, device, dataset_flag, lr, weight_decay, log_interval=10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    max_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(max_steps))
    try:
        dropscheduler = DropScheduler(model, epochs)
        dropscheduler.update('drop_path_prob', 0.2, update_original=True)
    except:
        pass

    logging.info("param size = %fMB", count_parameters_in_MB(model))    
    logging.info(f"Starting training on {dataset_flag} for {epochs} epochs...")

    best_acc = 0.0
    best_auc = 0.0
    auc_cor_bacc = 0.0
    acc_cor_bauc = 0.0

    for epoch in range(epochs):
        try:
            dropscheduler.schedule(epoch)
        except:
            pass
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            if dataset_flag in ['synapsemnist3d', 'vesselmnist3d']:
                y = y.squeeze().float().to(device)
            else:
                y = y.squeeze().long().to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=x, label=y)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        logging.info(f" Epoch {epoch+1}. Training Loss: {total_loss/len(train_loader):.4f}")
        if (epoch + 1) % log_interval == 0 or epoch == 0: #every 10 epochs / at start
            logging.info(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = model.predict(pixel_values=x)
                    logits = out["logits"]

                    if dataset_flag in ['synapsemnist3d', 'vesselmnist3d']:
                        y = y.squeeze().float()
                        preds = torch.sigmoid(logits).cpu().numpy()
                    else:
                        y = y.squeeze().long()
                        preds = torch.softmax(logits, dim=1).cpu().numpy()
                    y_pred.append(preds)
                    y_true.append(y.numpy())

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            evaluator = Evaluator(dataset_flag, split='test', root = args.data, size=64)
            
            try:
                metrics = evaluator.evaluate(y_pred)
                if isinstance(metrics, (list, tuple)):
                    auc = metrics[0]
                    acc = metrics[1]
                elif isinstance(metrics, dict):
                    auc = metrics["auc"]
                    acc = metrics["acc"]
                else:
                    auc = metrics
                    acc = 0.0
            except Exception as e:
                logging.error("Evaluation error: %s", e)
                pred_labels = y_pred.argmax(axis=1)
                acc = (y_true == pred_labels).mean()
                auc = 0.0 
            
            if acc > best_acc:
                best_acc = acc
                auc_cor_bacc = auc
                logging.info(' ---- New best accuracy achieved! ---- ')
            if auc > best_auc:
                best_auc = auc
                acc_cor_bauc = acc
            logging.info(f"     Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            logging.info(f"           Best ACC: {best_acc:.4f}, Corresponding AUC: {auc_cor_bacc:.4f}")
    
    return best_acc, best_auc, auc_cor_bacc, acc_cor_bauc, acc, auc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

import inspect
sig = inspect.signature(ImageClfModel.__init__)
print(sig)


if label_num <= 2:
    init_paras = {"base_dim": args.model_dim, "model_depth": args.model_depth, }
else:
    init_paras = {"label_num": label_num, "base_dim": args.model_dim, "model_depth": args.model_depth}

print('init_paras:', init_paras)
try:
    model = ImageClfModel(**init_paras)
    print('Init with model_depth ', args.model_depth)
except Exception as e:
    print('Error in initializing model:', e)
    raise e
    
acc, auc, auc_cor_bacc, acc_cor_bauc, last_acc, last_auc = train_and_evaluate(model, train_loader, test_loader, args.epochs, device, data_flag, args.learning_rate, args.weight_decay, args.eval_log_ep)
logging.info(f" Finished Training. Best: ACC={acc:.4f}, corresponding AUC={auc_cor_bacc:.4f}.")
logging.info(' Experiment done! ')
