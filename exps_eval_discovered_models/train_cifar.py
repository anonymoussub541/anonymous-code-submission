import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from exp_util import *

sys.path.append('/home/')

SAVE_DIR = './'

parser = argparse.ArgumentParser("cifartasks")

parser.add_argument('--modelclass', type=str, default='ImageClfModel', help='Model class name')
parser.add_argument('--modulepath', type=str, default='discoveredmodels.fds1run1m5', help='Full python import path of model module')
parser.add_argument('--modelname', type=str, default='modelname', help='Trained model name')

parser.add_argument('--data', type=str, default='/home/DATA', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--eval_freq_step', type=float, default=3000, help='eval by steps')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--data_flag', type=str, default='cifar100', help='which dataset to use')
parser.add_argument('--model_dim', type=int, default=32, help='model dim')
parser.add_argument('--model_depth', type=int, default=15, help='model depth')
parser.add_argument('--workers', type=int, default=0, help='number of workers to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume')
parser.add_argument('--grad_clip', type=float, default=3, help='gradient clipping')
parser.add_argument('--dropout_prob', type=float, default=0.0, help='drop out probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--eval_log_ep', type=int, default=2, help='evaluation log every n epochs')
parser.add_argument('--fixtrain_eval_log_ep', type=int, default=20, help='fixed training set evaluation log every n epochs')

args = parser.parse_args()
print('weight_decay and gpu', args.weight_decay, args.gpu)


eval_log_ep = args.eval_log_ep
fixtrain_eval_log_ep = args.fixtrain_eval_log_ep
eval_freq_step = args.eval_freq_step


import importlib
try:
    model_module = importlib.import_module(args.modulepath)
    ImageClfModel = getattr(model_module, args.modelclass)
except Exception as e:
    print('Error in importing model:', e)
    raise e


data_flag = args.data_flag
print(data_flag)

if data_flag == 'cifar100':
    print('Using CIFAR100 dataset')
    label_num = 100
    CIFAR100_MEAN=[0.5071, 0.4867, 0.4408]
    CIFAR100_STD=[0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                )
            ], p=0.7),
            transforms.RandomApply([
                transforms.RandomRotation(10)
            ], p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    
    train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    print(f'Train data size: {len(train_data)}, Valid data size: {len(valid_data)}')
    test_data = None

    fix_train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)
elif data_flag == 'cifar10':
    print('Using CIFAR10 dataset')
    label_num = 10
    CIFAR10_MEAN=[0.4914, 0.4822, 0.4465]
    CIFAR10_STD=[0.2023, 0.1994, 0.2010]
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                )
            ], p=0.7),
            transforms.RandomApply([
                transforms.RandomRotation(10)
            ], p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    print(f'Train data size: {len(train_data)}, Valid data size: {len(valid_data)}')
    test_data = None
    fix_train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
else:
    raise ValueError('data_flag must be either cifar10 or cifar100!')


modelname0 = args.modelname
savedir = os.path.join(SAVE_DIR, './{}/{}-{}'.format(data_flag, modelname0,  time.strftime("%Y%m%d-%H%M%S")))
create_exp_dir(savedir)



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(savedir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    print('data flag: ', data_flag)

    import inspect

    sig = inspect.signature(ImageClfModel.__init__)
    print(sig)

    if label_num <= 2:
        init_paras = {"base_dim": args.model_dim, "model_depth": args.model_depth, }
    else:
        init_paras = {"label_num": label_num, "base_dim": args.model_dim, "model_depth": args.model_depth}
    
    if 'mnist' in data_flag and 'mixup_alpha' in sig.parameters:
        init_paras['mixup_alpha'] = 0.2
    print('init_paras:', init_paras)
    
    try:
        model = ImageClfModel(**init_paras, drop_path_prob = args.drop_path_prob)
        print('Init with drop_path_prob ', args.drop_path_prob)
        print('Init with model_depth ', args.model_depth)
    except:
        try:
            model = ImageClfModel(**init_paras)
            print('Init with model_depth ', args.model_depth)
        except:
            try:
               model = ImageClfModel(**init_paras, drop_path_prob = args.drop_path_prob)
               print('Init with drop_path_prob ', args.drop_path_prob)
            except:
                print('no drop_path_prob ')
                model = ImageClfModel(**init_paras)

    if args.resume:
        print(f'Load model {args.resume}')
        model.load_state_dict(torch.load(args.resume))
        
    model = model.cuda()
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    optimizer       = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if test_data is not None:
        test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,  num_workers=args.workers)
    if fix_train_data is not None:
        fixtrain_queue = torch.utils.data.DataLoader(fix_train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    max_train_steps = args.epochs * len(train_queue)
    print('max_train_steps:', max_train_steps)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = float(max_train_steps))

    dropscheduler = DropScheduler(model, args.epochs)
    try:
        dropscheduler.update('drop_path_prob', args.drop_path_prob, update_original=True)
    except:
        pass
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    train_fun = train_top5
    infer_fun = infer_top5
    criterion = nn.CrossEntropyLoss()

    best_score_dict_val = {}
    best_score_dict_test = {}

    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        dropscheduler.schedule(epoch)
        
        train_output = train_fun(train_queue, model, optimizer, scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        train_temp = format_eval_output(train_output)
        logging.info('TRAIN: ep %d %s', epoch, train_temp)
        print('current_lr:', current_lr)

        if (epoch+1) % eval_log_ep == 0:
            val_output = infer_fun(valid_queue, model, criterion)
            key = [kk for kk in val_output if kk != 'LOSS']
            mainkey = key[0]

            is_best = val_output[mainkey] > best_score_dict_val.get(mainkey, 0)
            if is_best:
                best_score_dict_val[mainkey] = val_output[mainkey]
                save_model(model, os.path.join(savedir, 'best_weights.pt'))
                logging.info('   ---------- New best model based on val!')

            for xx in key:
                if xx not in best_score_dict_val:
                    best_score_dict_val[xx] = 0
                best_score_dict_val[xx] = max(best_score_dict_val[xx], val_output[xx])
            temp = format_eval_output(val_output)
            best_temp = format_eval_output(best_score_dict_val)
            logging.info('      VAL: ep %d %s', epoch, temp)
            logging.info('                 VAL: BEST: %s', best_temp)


            if test_data is not None:
                test_output = infer_fun(test_queue, model, criterion)

                for xx in key:
                    if xx not in best_score_dict_test:
                        best_score_dict_test[xx] = 0
                    best_score_dict_test[xx] = max(best_score_dict_test[xx], test_output[xx])
                temp = format_eval_output(test_output)
                best_temp = format_eval_output(best_score_dict_test)
                logging.info('      TEST: ep %d %s', epoch, temp)
                logging.info('                 TEST: BEST: %s', best_temp)

                
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_score_dict_val': best_score_dict_val,
                'best_score_dict_test': best_score_dict_test,
                'log': logging.getLogger(),
                'optimizer': optimizer.state_dict(),
            }, is_best, save_dir=savedir)

            
        if (epoch+1) % fixtrain_eval_log_ep == 0 and fix_train_data is not None:
            fixtrain_output = infer_fun(fixtrain_queue, model, criterion)
            temp = format_eval_output(fixtrain_output)
            logging.info('FIXTRAIN (eval on train data): ep %d %s', epoch, temp)
        

        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    print("finish! ", modelname0)
    

def format_eval_output(val_output):
    return ' '.join(['{} {:.7f}'.format(k, v) for k, v in val_output.items()])


def train_top5(train_queue, model, optimizer, scheduler, valid_queue = None, criterion = None):
    loss_obj = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.squeeze()
        optimizer.zero_grad()
        
        output00 = model(pixel_values = input.cuda(), label = target.cuda())
        loss = output00['loss']
        logits = output00['logits'].data.cpu()
        loss.backward()
        
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        prec1, prec5 = compute_accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        loss_obj.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if (step+1) % args.report_freq == 0:
            logging.info('train step %03d loss %e top1 %f top5 %f', step, loss_obj.avg, top1.avg, top5.avg)
        if (step+1) % eval_freq_step == 0:
            # logging.info('Intermediate eval at step %d', step+1)
            val_output = infer_top5(valid_queue, model, criterion)
            temp = format_eval_output(val_output)
            logging.info('      intermediate VAL: step %d %s', step+1, temp)
    
    return {'LOSS': loss_obj.avg, 'TOP1': top1.avg, 'TOP5': top5.avg}

def infer_top5(valid_queue, model, criterion):
    loss_obj = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            target = target.squeeze()
            logits = model.predict(pixel_values = input.cuda())['logits'].cpu()
            loss = criterion(logits, target)

            prec1, prec5 = compute_accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            loss_obj.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return {'LOSS': loss_obj.avg, 'TOP1': top1.avg, 'TOP5': top5.avg}

if __name__ == '__main__':
    main()
