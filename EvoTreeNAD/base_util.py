import importlib.util
import os
import re
import json
import sys
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_py_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        str_output = f.read()
        return(str_output)
    return None

def dump_py_file(pycode, file_path, force = False):
    if os.path.isfile(file_path):
        print(f"{file_path} already exists!!!")
    else:
        force = True
    if force:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pycode)
        print(f"Sucessfully save {file_path}!!!")

def _normalize_triple_quotes(code: str) -> str:
    """
    If code ends with ''' and total occurrences of ''' is odd, strip the trailing one.
    """
    if code.endswith("'''"):
        occurrences = code.count("'''")
        if occurrences % 2 == 1:
            return code[:-3]
    return code

def extract_python_code(response: str) -> str:
    code = None
    if "```python" in response[:100]:
        # Use regex to extract code block between ```python ... ```
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        if not match:
            if response.strip().startswith("```python"):
                code = response.strip()[9:].strip()
            else:
                print("!!!!No valid Python code block found in the response!")
                print(response)
                return ("#ERROR: Something wrong with code generation! ")
            # raise ValueError("No valid Python code block found in the response.")
        else:
            code = match.group(1).strip()
    else:
        code = response
    if code:
        code = _normalize_triple_quotes(code)
        return code
    else:
        print("No code generated! Or something wrong with the output format!")
        print('===The response is shown below===')
        print(response)
        return ("#ERROR: Something wrong with code generation!")


def import_fun_from_path(file_path):
    if not os.path.isfile(file_path):
        return None, f"File not found: {file_path}"
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, None

    except Exception as e:
        return None, str(e)

def convert_to_eval_output_path(cid):
    dir_path = os.path.dirname(cid)
    original_name = os.path.basename(cid)
    name_without_ext = os.path.splitext(original_name)[0]
    
    new_filename = f"evaloutput_{name_without_ext}.txt"
    return os.path.join(dir_path, new_filename)


def import_specific_module_from_path(file_path):
    if not os.path.isfile(file_path):
        return None, f"File not found: {file_path}"
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, None

    except Exception as e:
        return None, str(e)

def count_trainable_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params