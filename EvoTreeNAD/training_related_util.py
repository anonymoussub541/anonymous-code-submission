PROMPT_TRAIN_LONGTIME = f"""**Training Halted - Excessive Runtime**
The current model trains too slowly because of computational inefficiencies.
Please,
1. **Redesign the architecture** to reduce training time.
2. Provide either
  * A *slimmed‑down* version of the existing network, **or**
  * Cut or replace inefficient modules with lighter and effective alternatives.
"""

PROMPT_TRAIN_NOTGOOD = f"""**Early Training Termination - Sub-par Performance**
The model is failing to meet expected accuracy or loss targets.
Please,
1. **Diagnose the shortfall**
  * Identify likely causes (e.g., fusion strategy, underfitting feature flow, or ill-suited modules/architecture).
2. **Propose improvements**
  * Suggest specific changes and briefly explain how each could enhance performance.
"""

PROMPT_EXCEED_PARAS = "The model surpasses the parameter cap, and parts or even the entire architecture are bloated and inefficient, which is clearly in need of redesign! Rethink and redesign it to provide a smarter and more effective architecture!"


PROMPT_SMALL_PARAS = "The model has extremely few parameters, might lead to underfitting issues or unfair performance for the architecture evaluation! Please adjust internal design arguments, such as scaling factors, intermediate dimensions, expansion ratio, or growth rates, to create a more reasonable architecture, or drop efficient-priority mechanisms/components (e.g., set conv without groups.). Note: Do not modify any user-specified initialization parameters. You have room to pursue the performance. Do not increase the model size dramatically, just moderately increase some ratios, scaling, or module configurations, or remove some efficiency-focused components."


from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Optional, Callable, Dict, List, Any

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import math
import time
import numpy as np




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


def train_loop(
    model: nn.Module,
    *,
    evaluate: Optional[Callable[[Any], Any]] = None,
    compute_loss: Optional[Callable[[Any], Any]] = None,
    train_dataset: Dataset | None = None,
    test_dataset: Dataset | None = None,
    train_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
    scheduler_type: str = 'linear',
    collate_fn = None,
    epochs: int | None = None,
    max_steps_train: int | None = None,     
    max_steps: int | None = None,         
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size : int = 1,
    num_workers: int = 0,
    batch_size: int = 32,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-8,
    device: str | None = None,
    log_step: int = 100,
    eval_step: int = 100,                    
    eval_start_step: int = 0,
    max_norm: float = 1.0,
    total_talency_limit: float = 3600,
    early_stop_fn: Optional[Callable[[int, Dict], bool]] = None,
    print_metrics: bool = True,
    opt_type: str = 'adamw',
    lr_min_coef: float = 0.2,
):

    compute_loss_flag = False
    if compute_loss is not None:
        compute_loss_flag = True
        
    do_eval = False

    if test_dataset or test_loader:
        do_eval = True

    if train_loader is None:
        train_loader = make_loader(train_dataset, per_device_train_batch_size, shuffle=True,  num_workers = num_workers, collate_fn = collate_fn)
        
    if do_eval:
        if test_loader is None:
            test_loader  = make_loader(test_dataset , per_device_eval_batch_size , shuffle=False,  num_workers = num_workers, collate_fn = collate_fn)

    total_params = sum(p.numel() for p in model.parameters())
    param_str = f"The model has {total_params / 1e6:.2f}M parameters.\n"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if batch_size:
        assert batch_size%per_device_train_batch_size == 0, "batch size must be divided by per_device_train_batch_size"
        gradient_accumulation_steps = int(batch_size/per_device_train_batch_size)
    else:
        raise ValueError('must specify batch size!')

    steps_per_epoch = math.floor(len(train_loader) / gradient_accumulation_steps)

    if max_steps is None:
        assert epochs is not None, "Set either epochs or max_steps"
        max_steps = epochs * steps_per_epoch
        
    if epochs is None:
        epochs = math.ceil(max_steps/steps_per_epoch)

    
    total_update_steps = max_steps

    if scheduler_type not in ['linear', 'cosine', 'constant']:
        raise ValueError('scheduler_type must be either linear or cosine!')

    if opt_type == 'adamw':
        opt       = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adam':
        opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        opt       = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'momentum':
        opt       = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        raise ValueError('opt_type must be either adamw, adam, sgd or momentum!')
    print('opt type is ', opt_type)
    if warmup_ratio > 0:
        warmup_steps = int(warmup_ratio * total_update_steps)
    
    eta_min = lr * lr_min_coef
    if warmup_steps > 0:
        rest_steps = total_update_steps - warmup_steps
        warmup_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        if scheduler_type == 'linear':
            rest_scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=lr_min_coef, total_iters=rest_steps)
        elif scheduler_type == 'cosine':
            rest_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rest_steps, eta_min=eta_min)
        elif scheduler_type == 'constant':
            rest_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
        scheduler = optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_scheduler, rest_scheduler], milestones=[warmup_steps])
    else:
        if scheduler_type == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=lr_min_coef, total_iters=total_update_steps)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_update_steps, eta_min=eta_min)
        elif scheduler_type == 'constant':
            scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 1.0)
    
    remark_info = param_str
    history, update_step, running_loss = [], 0, 0.0
    t0 = time.time(); end_flag = False
    early_stop = False
    n_loss = 0
    n_minibatch = 0

    training_talency = 0.0
    st_each_step_time = time.time()

    if max_steps_train is None:
        max_steps_train = max_steps
    if max_steps_train > max_steps:
        max_steps_train = max_steps

    talency_limit_each_step = total_talency_limit/max_steps_train

    total_update_steps_80 = int(total_update_steps * 0.8)
    
    # ----- training ---------------------------------------------------------
    while update_step <= total_update_steps:
        model.train()
        for batch_idx, batch in enumerate(train_loader, 1):
            # forward --------------------------------------------------------
            if compute_loss_flag:
                loss_out = compute_loss(batch, model, device)
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                if compute_loss:
                    out = compute_loss(batch, model, 'cpu')
                else:
                    out   =  model(**batch)
                loss_out = out["loss"]
            loss  = loss_out / gradient_accumulation_steps  # scale
            if torch.isnan(loss):
                return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": "Error: Training Stop: NaN loss encountered!", "log": "NaN loss!"})
            loss.backward()
            
            running_loss += loss.item()
            if update_step > 10 and (running_loss==float('inf') or np.isnan(running_loss) or running_loss==0.0):
                return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": "Error: Training Stop: Invalid running loss encountered!", "log": "Invalid running loss!"})
            
            n_minibatch += 1

            should_step = (n_minibatch % gradient_accumulation_steps == 0)
            
            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True)
                update_step += 1
                et_each_step_time = time.time()
                training_talency += (et_each_step_time-st_each_step_time)                

                row = {"step": update_step,}
                log_temp_flag = False
                if update_step % log_step == 0:
                    

                    train_loss = running_loss/log_step
                    row.update({"train_loss": round(train_loss, 6),})
                    
                    log_temp_flag = True
                    running_loss = 0.0  # fresh window

                if do_eval and update_step%eval_step ==0 and update_step >= eval_start_step:
                    log_temp_flag = True
                    try:
                        val_metrics = evaluate(model, test_loader, device)
                    except Exception as e:
                        print(e)
                        return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": f"Error during evaluation: {str(e)}", "log": "Failed in evaluation!"})
                    row.update(val_metrics)
                    if print_metrics:
                        print(str(row))  
                    model.train()


                st_each_step_time = time.time()

                if log_temp_flag:
                    if print_metrics and (not do_eval):
                        print(str(row))
                        
                    history.append(row)
                    if early_stop_fn:
                        early_stop = early_stop_fn(update_step, row)

                if (training_talency > (talency_limit_each_step*update_step + 100)) and update_step < total_update_steps_80:
                    time_log = f"Time consuming: Finished {update_step} update steps in {training_talency/60:.1f} min."
                    return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": PROMPT_TRAIN_LONGTIME, "log": time_log})
                    
                if early_stop:
                    break

                if update_step >= max_steps_train:
                    end_flag = True
                    break

        if end_flag:
            break
            
        if early_stop:
            time_log = f"Unsatisfied performance: Finished {update_step} update steps in {training_talency/60:.1f} min."
            return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": PROMPT_TRAIN_NOTGOOD, "log": time_log})

    elapsed = time.time() - t0
    time_log = f"Finished {update_step} update steps ({epochs} epochs) in {elapsed/60:.1f} min."
    print(time_log)
        
    return({"metrics": history, "pass_check": True, "remark_info": remark_info, "error_msg": None, "log": time_log})


