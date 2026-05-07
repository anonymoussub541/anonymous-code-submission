

def get_prompt_upgrade_code(model_requirement_str):
    PROMPT_UPGRADE_CODE = f"""You are an elite PyTorch architect.

### Non-negotiable requirements
{model_requirement_str}

### Context You May Receive
• One or more previous model versions  
• Their evaluation metrics, training logs, and any instantiation or convergence issues  

### Mission
Create a new model architecture that **decisively out-performs** the best prior version while satisfying every requirement above, aiming to achieve superior model performance.

### Background
There may be a sequence of prior evolving model versions with their associated metrics. Since upgrades are exploratory, the latest version is not necessarily the best. In some cases, recent modifications may fail to improve performance—or even degrade it—indicating limited or negative contribution. It is therefore important to identify the strongest prior versions, evaluate the effectiveness of individual upgraded components and the corresponding original components, and guide model upgrading by building on the most appropriate model versions or selectively combining effective components (conduct effective and promising upgrades based on the best prior versions or carefully chosen component combinations).


### Rules
1. Honour every detail in the requirements—method names, signatures, tensor shapes, dtypes, device handling.
2. Analyse the code and metrics, pinpoint weaknesses, and engineer targeted improvements.
3. Create a high-performance model architecture and think about inventiveness and effectiveness: build modules and `forward` functions you believe could push accuracy, efficiency, or robustness further.
   - **Modules**: Explore effective modules for model performance. If earlier models are available, audit their design and metrics, diagnose model failure/poor performance/performance gaps, and then refine the model to improve performance.
   - **Loss Path**: Ensure task-appropriate forward logic and correct loss computation. Make sure the model is trainable!
   - **Curation**: Curate and refine operators and architectural motifs to enhance structure power. 
   - **Overhaul**: If predecessors faltered, rebuild decisively rather than tweaking timidly.  
4. Explicitly define model depth within the model (e.g., self.num_layers = 5), specifying the number of core layers/blocks and/or the hierarchical structure (e.g., number of stages and blocks per stage) to ensure clarity, reproducibility, and controlled scaling.
5. Depend on Python ≥ 3.1x, PyTorch 2.x, torchvision ≥ 0.22.x, and Transformers ≥ 4.48. Allow the use of Standard Python Libraries like `math` and `numpy`. Optional fields must define a default value (e.g., None).
6. Write clean, idiomatic, production-ready PyTorch (type hints welcome).
7. **Output the upgraded model code only.**
   – No docstrings, markdown, prints, or logs.  
   – Code must run once the requirements are in scope.

#### Some Design/Implementation Tips (optional)
  - For modularity, adaptability, and flexibility, consider using **factory-style helpers** such as `make_layers`, `make_blocks`, `make_stages`, `split_depth`, `split_channels`. If needed, use ratios or expansion/contraction factors to adjust configurations.
  - Maintain modularization: design with clear macro-, micro-, and connection-level structure to enable flexible refinement and extension. Define clear input/output specifications and ensure shape-safe module connections.
  - Conduct code refactoring for flexible and accurate implementation, modularization, or explicit composition if needed.
  - Based on the best knowledge and related context, choose appropriate structural hyperparameters, operators, and modules if they are not specified. Do not randomly guess or arbitrarily select them. Note: Bad hyperparameters or operators can cripple model performance.
  - Be mindful and strategic on selecting/allocating channels num, heads, dimensions, etc. E.g., when previous model exceeds parameter limits, one of solutions is to reducing or restructuring these components. In multi-path modules, distribute the channels or widths across paths rather than allowing them to grow uncontrollably. `torch.split` might be helpful.
  - Could fix divisibility bug in GroupNorm by adaptive group count selection.
  - Tensor alignment: interpolation (mode `bilinear`, `nearest` for upsampling, `area` for downsampling), projection (linear, Conv1x1), adaptive pooling, reshape/view, permutation/transposition (with `contiguous` when needed). 
  - Might need to fix bug (e.g., Dimension mismatches) or solve the issue if some bugs in the latest version. 
  - If group number mismatches or other incompatible shape issues occur in Conv/Norm layers, degrade to regular Conv/Norm layers.

### Required Comment Block in the file
'''  
Summary & Reflections: ≤ 40 words on inspirations and core design choices.  
Input Utilisation: ≤ 40 words on how inputs are encoded/integrated.  
Unchanged: indicate 'from scratch' or ≤ 30 words on what components (modules/blocks/flows/...) remained unchanged compared to the prior best model version.
Upgrade vs. Previous: ≤ 60 words on what changed, and proposed, why it helps, and expectations.  
'''

### Inline Commenting
- Annotate every argument in the **main module’s** `forward` method with its expected type & shape, **and semantic role**, e.g.  
  `past_input_ids: Optional[torch.LongTensor] = None,  # [B, past_seq_len, token_len] – past appointment history sequence`
  `pixel_values: torch.FloatTensor,  # [B, X_dim, Y_dim, Z_dim] – pixels of normalized 3D T1-weightd MRI brain image`
- Track Output Sizes of Key Modules and Key layers, e.g.
  x = self.conv1(x)  # [B, 3, 224, 224] → [B, 64, 112, 112]

"""
    return(PROMPT_UPGRADE_CODE)
