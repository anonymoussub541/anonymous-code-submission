import sys
import importlib.util
import os

def load_model_class_from_path(file_path: str, class_name: str):
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, class_name)

def count_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

modelname = 'ImageClfModel'
batch_path = '/home/project/exps_nad/fds1/batch_test.pt'
model_para_limit_B = 0.016
model_para_extreme_low_threshold_B = 0.0001
init_paras_used = {'label_num': 10, 'base_dim': 32, 'model_depth': 15}
import torch
def verify_model_from_path(model_code_path: str):
    errors = []
    try:
        modelclass = load_model_class_from_path(model_code_path, modelname)
        errors0 = []
        if errors0:
            errors += errors0
        else:
            if isinstance(init_paras_used, dict):
                model0 = modelclass(**init_paras_used)
            else:
                model0 = modelclass(*init_paras_used)
            model0_size = count_trainable_parameters(model0)/1e9
            model0_sizeM = round(1000*model0_size, 2)
            if model0_size > model_para_limit_B:
                error_oom = f"The model has {model0_sizeM}M parameters. The model surpasses the parameter cap, and parts or even the entire architecture are bloated and inefficient, which is clearly in need of redesign! Rethink and redesign it to provide a smarter and more effective architecture!"
                errors.append(error_oom)
            elif model0_size < model_para_extreme_low_threshold_B:
                error_oom = f"The model has {model0_sizeM}M parameters. The model has extremely few parameters, might lead to underfitting issues or unfair performance for the architecture evaluation! Please adjust internal design arguments, such as scaling factors, intermediate dimensions, expansion ratio, or growth rates, to create a more reasonable architecture, or drop efficient-priority mechanisms/components (e.g., set conv without groups.). Note: Do not modify any user-specified initialization parameters. You have room to pursue the performance. Do not increase the model size dramatically, just moderately increase some ratios, scaling, or module configurations, or remove some efficiency-focused components."
                errors.append(error_oom)
            else:
                if os.path.isfile(batch_path):
                    loaded_batch = torch.load(batch_path)
                    output00 = model0(**loaded_batch)
    except Exception as e:
        errors.append(str(e))
    passed = len(errors) == 0
    return (passed, errors)
