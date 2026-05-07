import sys
import argparse

def _parse_args():
    p = argparse.ArgumentParser(description="Apply strong regularization to a model file.")
    p.add_argument(
        "--file", "-m",
        help="Path to the model .py file.")
    p.add_argument('--add_mixup', action='store_true', help='Whether to add mixup augmentation.')
    p.add_argument('--add_cutmix', action='store_true', help='Whether to add cutmix augmentation.')
    p.add_argument('--add_cutout', action='store_true', help='Whether to add cutout augmentation.')
    return p.parse_args()

args = _parse_args()
model_file_name = args.file
print('model_file_name:', model_file_name)
import os
import json
import pandas as pd
from llm_api_util import *
from base_util import *


llmagent = AzureOpenAIChatClient('gpt5chat')

strong_reg_prompt = ""

PROMPT_POSTPROCESS_REG = f"""You are a precise Python refactoring assistant for PyTorch models.

## Goal
Enhance an existing PyTorch model with **regularization mechanisms** while strictly preserving:
- Model architecture and tensor shapes
- Public method names and signatures
- Input/output behavior

All changes must be **non-invasive** (regularization only) and optional (default settings keep behavior close to original).

## Output format
- **Output the full modified model code only** (no extra text, no markdown, no explanations).
- Include all original classes/functions (rewritten if necessary).
- Every addition or change must be annotated with `# [REG] ...`.
- Ensure to use `int` instead of deprecated `np.int` if needed, import necessary libraries (e.g., numpy, import numpy as np).

---

## Required Regularization Features

1. **Dropout / Dropout2d**
   - Apply it in the appropriate locations, e.g., In MLP-based modules (attention or context modulators), insert dropout between the two fully connected layers (after fc1 activation and before fc2).
   - Add configurable arguments like head_dropout: float = 0.05 to control the dropout rate.
   
2. **DropPath (Stochastic Depth)**
   - Apply drop_path independently to each non-identity path/branch before fusion. Note: In multi-branch residual structures, apply drop_path to the individual early branches rather than to the fused branch with the residual connection. 
     - Example Implementation:
        - fused = torch.cat([drop_path(out3), drop_path(out5)], dim=1)
        - fused = drop_path(out3) + drop_path(out5)
   - Note: do not apply drop_path to single-branch modules.
   - Use progressive scheduling (deeper blocks get higher probability). Note: please just set several levels of probs (e.g., each stage has one prob.) and start with 0.2 * drop_path_prob.
   - New arg, e.g., `drop_path_prob: float = 0.2`.

3. **Input Sample Augmentation (if specified)**
   - If requested, apply appropriate input sample augmentations (e.g., CutMix, MixUp) in the input samples in the training.
   - If there are multiple augmentations, apply them by setting an appropriate random probabilities or flags to control their application. By default, random probabilities should be positive to enable augmentations.
   - These augmentations must be embedded within the model logic, not dataset-level.

---

## Constructor Requirements
Top-level model class must accept and propagate new keyword-only parameters, e.g., 
- `drop_path_prob: float = 0.2`
- `head_dropout: float = 0.05`
- ...

If similar arguments already exist, reuse them (do not duplicate).

---

## Implementation Notes
- Do not alter normalization and loss logic.
- Skip irrelevant features gracefully if model lacks residuals, transformers, or auxiliary heads.
- Maintain original tensor shapes, strides, and padding.
- Do not rename public methods (`forward`, `predict`, etc.).
- Ensure all added regularization is enabled by default.
- Set top-level model attributes to store regularization settings for reference, and use self.<attribute_name> to access them within methods.
"""



PROMPT_CUTMIX_AUGMENTATION = """
Additionally, please implement CutMix augmentation in the input sample in the training. Ensure that the CutMix technique is correctly implemented. Ensure that CutMix is enabled by default (e.g., use_cutmix = True or set a positive probability) and that the augmentation is applied only during the training phase.

Implementation Example:
cutmix_alpha = 1.0 # by default
# [REG] CutMix augmentation
    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(images.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

    def _rand_bbox(self, size, lam): # Image tensor format used: [B, C, W, H]  
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

Note:When compute loss, must use lam for the loss.   
"""

PROMPT_MIXUP_AUGMENTATION = """
Additionally, please implement MixUp augmentation in the input sample in the training. Ensure that the MixUp technique is correctly implemented. Ensure that MixUp is enabled by default (e.g., use_mixup = True or set a positive probability) and that the augmentation is applied only during the training phase.

Implementation Example:
mixup_alpha = 0.2 # by default

# [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0  
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam

        
When compute loss, must use lam: loss = lam * loss_a + (1 - lam) * loss_b

"""

PROMPT_CUTOUT_AUGMENTATION = """
Additionally, please implement CutOut augmentation in the input sample in the training. Ensure that the CutOut technique is correctly implemented. Ensure that CutOut is enabled by default (e.g., use_cutout = True or set a positive probability) and that the augmentation is applied only during the training phase.
"""

PROMPT_CUTMIX_MIXUP_AUGMENTATION = """
Additionally, please implement both CutMix and MixUp augmentations in the input sample in the training. Ensure that both techniques are correctly implemented. Use random selection to apply either no augmentation, CutMix, or MixUp to input sample during training. Ensure that CutMix and MixUp are enabled by default (e.g., set positive probabilities 0.9 for augmentation) and that the augmentations are applied only during the training phase.


Implementation Example:
mixup_alpha = 1.0 # by default
cutmix_alpha = 1.0 # by default

# [REG] CutMix augmentation
    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(images.device)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam

    def _rand_bbox(self, size, lam): # Image tensor format used: [B, C, W, H] 
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

Implementation Example:
# [REG] MixUp augmentation
    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0:
            return x, y, y, 1.0  
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, y, y[index], lam


When compute loss, must use lam: loss = lam * loss_a + (1 - lam) * loss_b

    aug_prob = 0.9  # probability to apply augmentation
    ...
    if self.training and random.random() < self.aug_prob:  # [REG] apply augmentation
        if random.random() < 0.5:
            ... # apply CutMix
        else:
            ... # apply MixUp

"""


if args.add_cutmix and args.add_mixup:
    strong_reg_prompt += PROMPT_CUTMIX_MIXUP_AUGMENTATION
else:
    if args.add_cutmix:
        strong_reg_prompt += PROMPT_CUTMIX_AUGMENTATION
    if args.add_mixup:
        strong_reg_prompt += PROMPT_MIXUP_AUGMENTATION

if args.add_cutout:
    strong_reg_prompt += PROMPT_CUTOUT_AUGMENTATION

def get_modified_code(model_file_name, supple_context = ""):
    model_code_input = load_py_file(model_file_name)
    msg_list_temp = llmagent.init_msg(model_code_input, PROMPT_POSTPROCESS_REG + supple_context)
    response_temp = llmagent.request_llm_api_msgs(msg_list_temp)
    return extract_python_code(response_temp[0])

def extract_items(xx):
    temp = {kk:xx[kk] for kk in ['component_id', 'best_score', 'reward', ]}
    metrics0 = xx.get("metrics", {})
    temp['remark'] = metrics0.get("remark_info", "")[:25]
    temp['log'] = metrics0.get("log", "")
    return(temp)


Iternum = 0

suffix = ""
if args.add_cutmix:
    suffix += "_cutmix"
if args.add_mixup:
    suffix += "_mixup"

if args.add_cutout:
    suffix += "_cutout"


savedir0 = os.path.join(os.path.dirname(model_file_name), 'rg_code')
os.makedirs(savedir0, exist_ok=True)
save_path = os.path.join(savedir0, model_file_name.split('/')[-1].replace('.py', f'{suffix}.py'))



if os.path.exists(save_path):
    # exit
    print(f'Modified file already exists at {save_path}. Exiting to avoid overwrite.')
    sys.exit(0)


modified_code = get_modified_code(model_file_name, supple_context = strong_reg_prompt)
dump_py_file(modified_code, save_path, True)

if not modified_code:
    print('No modified code generated.')
    modified_code = get_modified_code(model_file_name, supple_context = strong_reg_prompt)
    savedir0 = os.path.join(os.path.dirname(model_file_name), 'rg_code')
    os.makedirs(savedir0, exist_ok=True)
    dump_py_file(modified_code, save_path, True)

if Iternum > 0:
    print('Iteration number:', Iternum)
    ii = 0
    while ii < Iternum:
        JUDGE_PROMPT = "\nPlease check if the regularization and augmentation has been applied correctly, especially the position of Drop operation. Answer 'Yes' or 'No'. No other explanations are needed."

print('finish')