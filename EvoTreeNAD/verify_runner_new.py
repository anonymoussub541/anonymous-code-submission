import json
import subprocess
import sys
import textwrap
from training_related_util import PROMPT_EXCEED_PARAS, PROMPT_SMALL_PARAS



_HELPER_SCRIPT = textwrap.dedent(r"""
    import json, sys, traceback
    from verfun_final_new import verify_model_from_path           # ← your import

    try:
        payload = json.loads(sys.stdin.read())                # receive args
        result  = verify_model_from_path(payload["model_path"])
        json.dump({"result": result}, sys.stdout)             # send back
    except Exception:
        traceback.print_exc(file=sys.stderr)                  # forward tb
        sys.exit(1)
""")


def preprocessing_verify(model_path: str,
                         timeout = 180):
    """
    Run verfun_final.verify_model_from_path(model_path) in a *fresh* Python
    interpreter.  Kill it if it exceeds `timeout` seconds.

    Returns
    -------
    Whatever verify_model_from_path returns.

    Raises
    ------
    TimeoutError : if the call runs longer than `timeout`
    RuntimeError : if the child crashes; the child’s traceback is included.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _HELPER_SCRIPT],   # -u = unbuffered I/O
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        out, err = proc.communicate(
            json.dumps({"model_path": str(model_path)}).encode(),
            timeout=timeout
        )
    except subprocess.TimeoutExpired:     # hard kill on timeout
        print("Timeout reached. Terminating task...")
        proc.kill()
        raise Exception("Error: Something wrong in smoke testing of the model!!! Timeout reached. The model initiation was terminated.")

    if proc.returncode:
        # bubble up the child’s stderr so you see the exact reason
        # raise RuntimeError(f"Child exited with code {proc.returncode}:\n{err.decode()}")
        print(f"Child exited with code {proc.returncode}:\n{err.decode()}")
        raise Exception("Error: Something wrong in smoke testing of the model!!! Timeout reached. The model initiation was terminated.")

    return json.loads(out)["result"]



def generate_verfun_final_new_code(modelname, batch_path, model_para_limit_B, model_para_extreme_low_threshold_B, init_paras_used, verify_signature = False):
    if verify_signature:
        signature_check_str = "passed0, errors0  = verify_model_from_path_old(model_code_path)"
        raise NotImplementedError("Signature verification is not implemented yet.")
    else:
        signature_check_str = "errors0 = []"
    code_str0 = f"""import sys
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

modelname = '{modelname}'
batch_path = '{batch_path}'
model_para_limit_B = {model_para_limit_B}
model_para_extreme_low_threshold_B = {model_para_extreme_low_threshold_B}
init_paras_used = {init_paras_used}
import torch
def verify_model_from_path(model_code_path: str):
    errors = []
    try:
        modelclass = load_model_class_from_path(model_code_path, modelname)
        {signature_check_str}
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
                error_oom = f"The model has {{model0_sizeM}}M parameters. {PROMPT_EXCEED_PARAS}"
                errors.append(error_oom)
            elif model0_size < model_para_extreme_low_threshold_B:
                error_oom = f"The model has {{model0_sizeM}}M parameters. {PROMPT_SMALL_PARAS}"
                errors.append(error_oom)
            else:
                if os.path.isfile(batch_path):
                    loaded_batch = torch.load(batch_path)
                    output00 = model0(**loaded_batch)
    except Exception as e:
        errors.append(str(e))
    passed = len(errors) == 0
    return (passed, errors)
"""
    return(code_str0)
    