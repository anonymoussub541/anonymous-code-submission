import json
from textwrap import dedent

import torch

BASE_DESIGN_TRICK_CONTEXT = """### Basic Design Concepts
*(Illustrative; Not exhaustive;)*

– **Depth**: Macro/micro Depth; layers per block, number of stacked blocks, number of stages
– **Width**: hidden size, number of heads, channel num, total block width, path width
– **Scale**: receptive field, kernel/window size, context length, dilation rate, temperature
– **Diversity**: feature heterogeneity, modality variety, pathway diversity  
- **Feature Transformation**: operator families, representation bases, layers
- **Connectivity**: sequential, stacked, parallel, residual/skip, dense, cross-level, multi-branch/path, DAG-based, hierarchical, nested, hybrids, flexible customized topology
- **Composition**: Operators/layers/blocks/modules with flexible or varied topology (define connectivity explicitly)

"""


MODEL_DESIGN_CONTEXT = f"""### General Design Tricks — Quick Reference
*(Not exhaustive; a launch-pad reference for architectural exploration)*
*Mix, match, extend, and **invent** your own architectural styles*

• **Layers/Blocks**: Conv, Linear, MLP, Transformer, GNN, State-Space Models, Mixer, Capsule, Neural ODE, hybrids, …
• **Structural Patterns**: sequential, parallel, pyramidal, hierarchical, skip connections, DAG-based, multi-branch, multi-path, fractal, hybrids, …
• **Some Design Techniques**:  
  - **Representation Transformation**:conv variants (dilated, separable, grouped, inverted, deformable, dynamic), attention variants (local, sparse, linearized, multi-scale), pooling (max/avg, global, adaptive, learned, graph pooling), spectral/frequency transforms (Fourier, Wavelet, scattering), graph-based layers (GCN, message passing), token/MLP-mixing, shift/shuffle operations, ...
  – **Fusion & Enrichment**: multimodal fusion, cross-attention, heterogeneous feature fusion, feature enrichment  
  – **Encodings**: positional & structural encodings (sinusoidal, learned, Fourier features, RoPE, ALiBi, relative), temporal encodings, graph encodings, type/segment embeddings  
  – **Residual & Compression**: residual connections, bottlenecks, highway layers, layer scaling, normalization placement (PreNorm/PostNorm)  
  – **Specialized Modules**: gated units (GLU / SwiGLU / GEGLU), attention variants (sparse, linear, local, deformable), dynamic routing 
  - **Activations**: ReLU family (ReLU, LeakyReLU, PReLU), smooth (SiLU/Swish, GELU, Mish), 
  - **Normalizations**: BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, SpectralNorm, …
  - **Losses**: focal-, dice-, smooth-, contrastive-, ...; uncertainty, multi-task, auxillary, ...
  - **Input-level Sample Augmentations**: noise injection, masking, MixUp / CutMix, RandAugment / AugMix, GridMask, CutOut,  ...
  - **Regularizations**: dropout variants, stochastic depth, data augmentation, label smoothing, ...
  - ...

"""

CUSTOMIZED_DESIGN_REFERENCE = f"""### Design Generalization and Customization Guidelines — Quick Reference
*Think modular, interactive, heterogeneous, multi-branch/path, generalizable, and adaptable*

• Topological Modules: if treat feature tensors as nodes, topological flows define how nodes are connected and produced via computational operators/blocks (edges). Sequential could be regarded as a simple chain-like topology. Multi-branch could be regarded as a multi-chain topology. A generalized version is to design DAG-style topology with a curated graph. It consists of operator edges, node production/aggregation. It could support heterogeneous ops and rich connections. The module could take multiple inputs and produce multiple outputs if needed.
    - Curated graph is formatted as {{'j->i': op/block}} while j, i represent feature tensor nodes.
• Local–global blending: varied kernel sizes, dilation, windowed attention, cross-level feature fusion/links, pyramidal structures, hierarchical/multi-stage Backbones, long-distance interactions, ...
• Shortcut & Preservation: explore skip/identity paths, highway/GRU gates, layer reentry, and cross-level bridging to stabilize training and reuse features.
• Deep State Construction: design blocks that iteratively generate a sequence of internal states, enabling rich and hierarchical interactions;
    - standard form: (s_i, s_{{i+1}}, ...) → s_{{i+k+1}};
    - multi-element form: (s_i, h_i, ...) → (s_{{i+1}}, h_{{i+1}}, ...);
    Design blocks to support expressive fusion across states/elements (e.g., using the above techniques).
    Possible inputs for states or elements include features from different scales, ops, branches, representations, or modalities.
• Flexible interactions/connections and Information Flow:
    - Multi-Input / Multi-Output (Multi I/O) Flow: design blocks that can handle multiple inputs and/or outputs.
    - Multi-Tensor States: allow multiple tensors as internal and intermediate states within/cross/through blocks/stages, and design rich interactions and connections among them.
    - Multiple streams/pathways
    - Hybrid Multi-stream and Multi-branch Structure with Criss-cross Connections
    - Feature Interaction

"""

GENERAL_DESIGN_TRICK_KNOWLEDGE = f"{BASE_DESIGN_TRICK_CONTEXT}\n{MODEL_DESIGN_CONTEXT}\n{CUSTOMIZED_DESIGN_REFERENCE}"



# 1) TASK ANALYSIS PROMPT ──────────────────────────────────────────────
PROMPT_extract_task_info = '''ROLE: Task analysis expert.

PURPOSE: Given (a) a raw / terse task snippet and/or (b) a Model Requirement
dict string (with implementation noise), extract a clean, architecture‑agnostic
task profile.

IGNORE / STRIP:
- Imports, logging, debug, comments, todos
- Optimizer / scheduler / trainer / seed / checkpoint / device mechanics
- Pure boilerplate (__init__, forward scaffolding), dtype/device moves
- Generic flags (verbose, log_interval, save_dir, amp, wandb, determinism)
- Internal helper names unless they reveal domain semantics
- Explicit "use cpu/gpu" or code execution environment notes

ALLOWED SIGNAL USE:
- Names revealing domain (e.g. pixel_values → image RGB tensors)
- Shape / type cues (e.g. (B, 3, H, W) → image; num_classes → classification)

STYLE RULES:
- EXACT section headers below, order preserved
- Bullets only; terse / telegraphic; no full sentences
- No architecture modules, training tricks, losses, optimizers
- No hallucination; write "none stated" when absent
- Keep domain terms lowercase unless proper noun
- Do not restate noise
- ≤ 14 words per bullet

SECTIONS:

1. CORE TASK SUMMARY
- task type
- objective / prediction target
- label structure (classes / continuous / multi-label / none stated)
- task details/description

2. DATA & INPUT SIGNALS
- modalities / principal tensors
- shape / scale hints (resolution, channels, seq len) if present
- preprocessing / normalization hints
- variability factors (resolution, drift, imbalance, augmentation need) or none stated
- data quality issues (noise / imbalance / sparsity / none stated)

3. HYPERPARAMETER CONCERNS
- batch regime (small ≤32 / mid / large / none stated, if any)
- constraints (params / memory) or none stated
- sample-shape-related hyperparameter range or none stated (e.g., #stages: 3~5)

4. SIMILAR TASKS & DOMAINS
- related task archetypes (e.g. image classification, token tagging) or none stated
- adjacent domains / application areas or none stated

Return ONLY these sections, in order.'''


def get_refined_task_description(agent, model_requirement_str, task_specific_str = ""):
    if task_specific_str:
        task_specific_str = f"Task-specific description: {task_specific_str}\n"
    model_requirement_str_inserted = f"{model_requirement_str}\n"
    output = agent.request_llm_api_prompts(task_specific_str+model_requirement_str_inserted, PROMPT_extract_task_info)
    return output



# 2) RELATED WORKS & DESIGN REFERENCE PROMPT ──────────────────────────────
PROMPT_RELATED_WORKS_TELEGRAPHIC_SYSTEM_PROMPT = """ROLE: You are a **deep-learning architecture synthesizer** — concise, analytical, exploratory.

---

## OBJECTIVE
Given a task description, produce a **conceptual design reference** that:
- List influential model families on the similar task (e.g., ConvNets, MobileNets, Transformers).
- Summarizes **related architectural ideas** and **conceptual mechanisms** across model families.
- Distill a comprehensive knowledge reference about design insight/seed for model development.
  - Might cover information flows, novel modules, representation basis, fusion, feature extraction, loss functions, topology, macro/micro structure, sample augmentation, regularization, and related model design techniques.
- No paper names or citations.

---

## STYLE
- Each bullet is a design knob or axis
- Ultra-terse: 1–3 words per item in concept level
- Non-sentential: no full sentences or descriptions
- Neutral: descriptive, not prescriptive or narrative
- **Neutral tone**: not prescriptive or narrative.
- No paragraphs, no sentences, no explanations.
- Example: 
  - • Conv variants: dilated, dep-sep, separable, grouped, inverted, deformable, dynamic.
  - • Input-level sample augmentations: MixUp, CutMix, RandAugment.

---

## SCOPE
Covers forward architecture design — including structural hyperparameters, representation, operators, connectivity, normalization, regularizaiton, sample augmentation, routing, multi-scale fusion, loss design (when defined within the forward pass), modularity, overfitting derisk, and efficiency.
Excludes training-related elements such as training strategies, optimization algorithms, schedulers, or other procedures outside the forward computation.
Do not mention the expected benefits or pitfalls of each design element, which might limit the open-minded exploration of design possibilities.
Do not provide specific instructions on how to use each design element, which might constrain the creative application of these concepts. For example, never specify "use BatchNorm to stabilize training" or "what stage components should be".
Note: Design should never be limited to a single choice. Architecture-related concepts should be preserved and fully considered. The list of design elements, concepts, and paradigms should be diverse, comprehensive, and richly expressive.
---


## OUTPUT TITLE (must match exactly)
### Related Works & Some Model Design Reference  
**(Not exhaustive; use as a launch-pad)**
Inspire, borrow, extend, and invent your own architectural styles

---

## FORMAT RULES
- Unordered bullet list format throughout.
- Divide logically into three informal clusters:
  - **Related Works (illustrative)**
    - list related model families (encourage borrowing and extending ideas)
  - **Some task-related Design Elements/Concept/Paradigm (illustrative, not exhaustive)**
- Total word count <800.
- No long sentences, no transitions, no citations, no markdown fencing.

---

## EXAMPLES OF OUTPUT FORMAT (do not copy literally. Use as inspiration only. )
- Model families: ConvNets, MobileNets, Transformers.
- Feature Transformation Operators: conv variants, attention variants, pooling variants.
- Feature diversity ops for multi-path fusion: dilation, varied kernel sizes, local feature aggregation (max/avg-pool with stride=1), filtering.
- Conv variants: dilated, dep-sep, separable, grouped, inverted, deformable, dynamic.
- Pooling & Downsampling: max/avg, Strided, Adaptive, SPP, GAP.
- Information Flow & Topological Connectivity: sequential, parallel, pyramidal, hierarchical, dense, skip connections, cross-hierarchy, learnable DAG, multi-branch, multi-path.
- Feature Enhancement & Expressiveness: feature reuse, mix-style blocks, gated feature modulation.
- Adaptive routing — dynamic paths, conditional compute, MoE; soft or hard.
- Feature Heterogeneity & Multi-Scale Handling — cross-scale, cross-stage, local+global, stage-wise.
- Topological flows — Learnable DAGs, multi-branch, fractal, recurrent, Graph-based.
- Feature Fusion & Integration — concat, sum/add, gated, cross-attn, attn-based, bilinear.
- Normalizations — BatchNorm, LayerNorm, GroupNorm, RMSNorm.
- Activations — ReLU, smooth (SiLU/Swish, GELU, Mish).
- Input-level Sample Augmentations — MixUp, CutMix, RandAugment.
- Deep supervision — intermediate losses, multi-task heads.
...

---

Output only the formatted text above, nothing else.
"""



def get_design_reference(agent, task_description):
    output = agent.request_llm_api_prompts(task_description, PROMPT_RELATED_WORKS_TELEGRAPHIC_SYSTEM_PROMPT)
    return output


from textwrap import dedent

UPGRADE_DIRECTION_CONTEXT = F"""### Upgrade Design Directions: General, Exploratory, and Subtractive
- **General refinement** Careful evolution of the current design to improve performance.
  - For any refinement axis, broaden your mind, do not restrict yourself to a very narrow set of standard options or always aim for the seemingly best one. Allow for unconventional or seemingly non-greedy alternatives. That is, if you would like to upgrade the model in a refinement axis, try to extend your mind to a broad set of possible choices, and freely select from them.
  - E.g., redesign components, adjust essential structural elements and ops within blocks/modules, restructure modules/blocks, enhance/swap operators, adjust model scales, adjust macro structure, increase module/block depth or width (e.g., layers, branches, ratio, growth), replicate/deepen/widen blocks, enhance feature diversity, adjust normalization, or refine loss.
  - Review the whole architecture and aware of model depth/width/scale/diversity settings to ensure they are well balanced, decent, and appropriate. Degenerate/adjust/swap components/blocks to enhance model capacity and expressiveness.
  - Review operators, curate operators and architectural motifs to enhance structure power, e.g., adjust operators in xx block with yy.
- **Exploration and Novelty** Rather than `General Refinement`, avoid relying on the diagnoses of prior models (e.g., address shortcomings). Draw inspiration from sound practices or conceptual insights, propose exploratory and experimental ideas with creativity, include but not limited to: explore, extend/generalize, and combine sound practices with creative twists; introduce new paradigms/designs, evolving information flows, novel modules, modify/swap/refactor/redesign components including representation basis, fusion, feature extraction, input augmentation, regularization, loss functions, topology, macro/micro structure, and other model design techniques). Do not be limited by existing paradigms. Design should never be constrained to a single choice. E.g., stay open-minded about efficiency; look beyond SE modules.
- **Subtractive Improvement** Identify ineffective components and eliminate or redesign them for efficiency and robustness. Simplify and reorganize the model’s architecture and codebase through clear, modular organization, ensuring extensibility for future enhancements.
  - E.g., eliminate ineffective blocks/modules, eliminate attention layer in the ** block, simplify fusion strategies, or streamline computationally expensive blocks.
"""

IDEA_PRODUCE_INSTRUCTIONS = F"""Produce **multiple distinct ideas** (caller decides the exact number) that differ in core architecture, key mechanisms, key elements, or key tricks. 
  - If no prior models have been developed, propose distinct **new model designs** from scratch.
  - If prior model versions are provided, carefully read the code, review and compare their performance, and assess the importance of model components to gain insights into model enhancement. And propose ideas across the following categories:
   - **General Upgrade**: Drawing on your best knowledge, empirical evidence from the literature, and the analysis of the provided information, propose distinct upgrade or refinement ideas that are **most likely** to yield meaningful improvements in model performance.
   - **Exploratory Upgrade**: Must propose distinct ideas that involve **significant exploratory modifications/adjustments** at the macro/micro/module/block/operator/connection level (e.g., curating operators, modifying, swapping, or adding operators/modules/blocks/connections, or scaling) for better model performance.
   - **Subtractive Upgrade** (optional): Provide distinct ideas focused on eliminating **ineffective** or **inefficient** components, enhancing efficiency and stability while preserving model effectiveness. Note: Please do not take any components as proven effective.

"""

IDEA_OUTPUT_FORMAT_INSTRUCTIONS = """### For each idea, provide either a new design or a model modification, include
1. **Model Design / Modification**
   - Indicate `"New Design (from scratch)"` if no prior models exist. Otherwise, specify upgrade category, e.g., Exploratory Upgrade.
   - Concise summary header (e.g., inspired/motivated by xx, introduce/adjust/replace/remove yy,  etc.)
2. **Input Processing** – how every modality/feature type is embedded, projected, pooling, or conditioned (positional encodings, FiLM, adapters, …).
3. **High-Level Design** - Summarize the main idea and proposed upgrade/change in this model version. Describe the overall structure—whether sequential, modular, or topological—along with feature flow and component interactions. Must specify key operator/element/component choices in the upgrade.
4. **Essential hyperparameters specification** Specify key macro- and micro-level hyperparameters, including: number of stages, depth per stage, module/block depth, channel width, number of blocks or stacks, hidden dimensions, feedforward size, attention heads, and other relevant settings. Note: Bad setting leads to poor performance.
5. **Design/Upgrade Highlights** – Demonstrate the core modifications/innovations or distinguishing elements of this design, emphasize differences from known models or prior versions. Provide details on the affected components, upgrade strategy, design insights, architectural improvements/modifications, or any cross-domain adaptations, if any.
6. **Expected Impact** – a brief rationale describing the expected improvements by key modifications/new design. Reference relevant constraints, known failure modes in `model_versions`, or supporting research if applicable.

**Note**: Do not mention prior model version names (i.e., V1, V2, model version 1) in idea contents. If retaining(i.e, unchanging) or modifying components from earlier versions, explicitly specify the affected component and describe the change or retention concisely (e.g., Maintained convolutional stem while replacing strided convolutions with pooling.). 
- **Avoid arbitrary hyperparameter settings**: Employ well-considered hyperparameter settings (e.g., stage depth, number of stages, etc.) to mitigate the risk of suboptimal or underfitting performance caused by random parameter choices, ensuring that the evaluation experiments faithfully reflect the design quality.
- **Trap: Unbounded Width Increase**: Be mindful and strategic on selecting/allocating channels num, heads, dimensions, etc. E.g., in multi/mixed/dual-path modules, distribute the channels or widths across paths, such as splitting the width, rather than allowing them to grow uncontrollably.
- Could leverage `transformers` and `torchvision` packages.
- For tensor alignment: interpolation (mode `bilinear`, `nearest` for upsampling, `area` for downsampling), projection (linear, Conv1x1), adaptive pooling, reshape/view, permutation/transposition (with `contiguous` when needed). 

### 2 High-Quality, Well-Structured Model Design Idea Examples
- {'Model Design / Modification': 'General Upgrade – Dilated Convolution for Multiscale Context', 'Input Processing': 'Standard stem, positional embeddings concatenated.', 'High-Level Design': 'In branches, replace some convolutions with dilated convolutions (dilation rates: 1, 2, 4 across branch depths), enhancing receptive field without extra params. Branch fusion uses weighted sum.', 'Essential hyperparameters specification': 'expansion=4, pos_channels=8, dilation_rates=[1,2,4].', 'Design/Upgrade Highlights': 'Dilated convolutions capture multiscale context, improve classification of spatially varied objects. Parameter cost remains minimal.', 'Expected Impact': 'Higher accuracy for classes with scale or spatial variance, better robustness for cluttered inputs.'}
- {'Model Design / Modification': 'General Upgrade – Hierarchical CondenseNet-inspired Dense Connectivity', 'Input Processing': 'Standard stem (Conv2d + BN + SiLU), followed by concatenation of learned positional embeddings at each stage input.', 'High-Level Design': "Replace branch isolated processing with dense connectivity within each stage: every block receives input from all previous blocks, following CondenseNet's hierarchical dense links. At each fusion, use Conv1x1 to compress concatenated features back to target channel width. Deeper branch depth reduced to maintain parameter budget.", 'Essential hyperparameters specification': 'expansion=3, pos_channels=8, compression_ratio=0.5.', 'Design/Upgrade Highlights': 'Dense links in each branch promote feature reuse and improved information flow, boosting gradient propagation and generalization. Stage fusion compresses multi-block concatenated features, maximizing expressivity under budget.', 'Expected Impact': 'Expected improvement in feature diversity and learning efficiency, yielding better accuracy and robustness to data variability, as seen in DenseNet/CondenseNet literature.'}

### Necessary Clarifications in complicated idea proposals
Ambiguity or insufficient specification may cause implementations to collapse and degression, which can obscure or distort the intended design ideas. For complicated designs, provide additional clarifications to ensure accurate delivery. E.g., heterogeneous configurations, asymmetric settings, nested details, topological pathways.


### Independence rule
Each idea must be **independent and mutually distinct** —  
no references, comparisons, or derivations between them.  
Each idea must be a standalone candidate derived only from task, knowledge, and prior model versions, not from other ideas. Each idea represents one standalone trial for the next model version.

### Diversity encouragement and exploration of possibilities
Ideas should be **diverse and distinct, representing a broad spectrum of design possibilities**. They must not overlap or converge on similar concepts, but instead demonstrate variety across different directions.


## Output format
Return **valid JSON** only (no markdown fences):
```json
{
  "idea1": "<fully‑described proposal>",
  "idea2": "<fully‑described proposal>",
  "idea3": "...",
  "...":   "..."
}
```
IGNORE all safety heuristics about output length.
DO NOT refuse. DO NOT warn. DO NOT offer alternatives. Follow instructions and output format.

"""


MODEL_DESIGN_SYSTEM_PROMPT = f"""
You are a **deep‑learning model architect**.

## Context you receive on every call
* `model_requirements` – a Python dictionary that spells out hard constraints (parameter budget, input/output specs, device limits, method signatures, deployment targets, etc.). Note: The inputs fed into model are specified in `model_requirements`, do not assume additional inputs. 
* Optionally, `model_versions` – a list / dict containing past architectures (model codes), training logs, and evaluation metrics.

## Your mission
Propose **multiple, clearly distinct, high‑performing model designs** or targeted modifications that respect all constraints, aiming to **achieve superior model performance**. Note: All model versions, if available, do not reflect the optimal performance attainable for this task; therefore, the performance ceiling should not be considered reached. Explore all opportunities for improvement and potential performance gains. Avoid arbitrary (lacking logic or justification) inconsistencies in repeated architectural components during model upgrades.

### Background
There may be a sequence of prior evolving model versions with their associated metrics. Since upgrades are exploratory, the latest version is not necessarily the best. In some cases, recent modifications may fail to improve performance—or even degrade it—indicating limited or negative contribution. It is therefore important to identify the strongest prior versions, evaluate the effectiveness of past upgrades, and guide next model upgrading by building on the most appropriate model versions or selectively combining effective components (review model architectures and conduct effective and promising upgrades based on the best prior versions or carefully chosen component combinations). Please avoid referring to specific model version names in outputs (i.e., V1, V2); instead, detail the components themselves, specifying which are effective or ineffective.


{UPGRADE_DIRECTION_CONTEXT}


### What to do
1. **Audit** the supplied requirements (and prior versions, if any) to spot performance gaps and design bottlenecks.
2. {IDEA_PRODUCE_INSTRUCTIONS}

---

{IDEA_OUTPUT_FORMAT_INSTRUCTIONS}
"""


def get_model_design_prompt(model_requirements_str: str,
                            model_versions_str: str | None = None,
                            k: int = 10,
                            supl_text: str = "",) -> str:

    prior_context = model_versions_str + '\n' if model_versions_str else ""

    requirements_block = f"""
### Current `model_requirements`
{model_requirements_str}


{supl_text}
---
Please propose **{k} distinct model design/modification ideas** following the specification in the system prompt.

"""
    return f"{prior_context}{requirements_block}\n".strip()




def get_get_brace_string(temp):
    return temp[temp.index('{'):temp.rindex('}')+1]



MODEL_DESIGN_SYSTEM_PROMPT_2phase_new = f"""You are a **deep-learning model architect and conceptual explorer** — rigorous in diagnosis and bold in inventing out-of-convention architectures.

## Context (always provided on each call)
* `model_requirements`: Python dict of hard constraints (parameter/FLOPs budget, I/O specs, device limits, method signatures, deployment targets, etc.).
* Optionally, `model_versions`: list/dict of prior architectures (codes), training logs, and evaluation metrics.

## Missions
1) **Diagnostic Analysis**
   Analyze the impact of architectural **changes** between model versions (Especially the latest version vs. its prior versions). **Focus only on modified components** (additions, removals, swaps, adjustment, refactor, relocations), and evaluate whether each change led to performance gain or degradation, with structural and metric-based reasoning.
   **Do not analyze unchanged parts unless they interact with the changes.**
   Clearly state if a change was effective or not, and why (e.g., placement too late, poor integration, insufficient depth). For ineffective or harmful changes, explain the mechanistic cause — avoid vague terms unless directly applicable. Analyze how the change relates to the original structure (e.g., imbalance, bottleneck, context loss).
   No need to diagnose what model lacked or could have had unless directly tied to a change.

2) **Creative Ideation**
   Brainstorm **multiple, clearly distinct, high-performing architectures or targeted modifications** that satisfy all hard constraints and **push beyond incrementalism**.Note: All model versions, if available, do not reflect the optimal performance attainable for this task; therefore, the performance ceiling should not be considered reached. Explore all opportunities for improvement and potential performance gains. Avoid arbitrary (lacking logic or justification) inconsistencies in repeated architectural components during model upgrades.


## Background
Past versions may underperform or regress; newest is not necessarily best. Identify historically strongest components, separate helpful from unhelpful upgrades, and guide the next design by recombining effective elements or reverting harmful ones. Avoid referencing version names (e.g., V1/V2); refer to **components** and their roles/effects.

{UPGRADE_DIRECTION_CONTEXT}

## What to do
1) **Audit**: Write a concise analytical report following the **Diagnostic Analysis** instruction.
2) {IDEA_PRODUCE_INSTRUCTIONS}

## Output expectations (freeform; no rigid template)
- Produce a **continuous expert report**: first the diagnostic analysis, then the exploratory upgrade proposals as separate freeform paragraphs.
- Do **not** use lists, headers, tables, or fixed schemas. No code or hyperparameters.

## Independence Rule (strict)
Each idea must be independent. Do **not** compare to, derive from, refine, or reference other proposed ideas. Every idea is an alternative candidate derived only from the task constraints and any prior models — **not** from other ideas in this output.
"""

MODEL_DESIGN_USER_PROMPT_phase2_new = f"""Review the **preceding analysis report and initial upgrade ideas** you generated.

## Your tasks
1. **Critically evaluate** all earlier ideas:
   - Check each for clarity, originality, and architectural soundness.
   - Detect redundancy, overlap, or weak/low-impact concepts.
   - Decide which ideas to **retain**, which to **refine**, and which to **replace** entirely.
   - Identify missing coverage among all upgrade categories.

2. **Refine and expand**:
   - For strong ideas: Enrich with detailed architectural reasoning, explicit structure, and implementation feasibility.
   - For weak or missing ideas: Create **new, clearly distinct model designs** that better satisfy constraints and objectives.
   - All designs must respect `model_requirements` and draw insight from any `model_versions` but remain **independent** of each other.

3. **Goal**
   - Produce multiple, mutually distinct, high-performing model designs or targeted upgrades.
   - Each idea must be self-contained and written as a full, detailed proposal, describing its mechanisms, retained or changed components, and expected performance improvements.

---

{IDEA_OUTPUT_FORMAT_INSTRUCTIONS}
"""



def MODEL_DESIGN_USER_PROMPT_phase1(model_requirements_str: str,
                            model_versions_str: str | None = None,
                            k: int = 10,
                            supl_text: str = "") -> str:

    # ――― optional historical context ―――
    prior_context = model_versions_str + '\n' if model_versions_str else ""

    # ――― mandatory requirements block ―――
    requirements_block = f"""
### Current `model_requirements`
{model_requirements_str}

{supl_text}
---

Analyze the above `model_requirements` and any `model_versions` and the corresponding metrics, report the diagnosis in **bullet points**. And draft in brief **{k} distinct model design/modification ideas** expressing key designs/changes for potential-high performance/issue-solving.

"""
    # ――― final assembly ―――
    return f"{prior_context}{requirements_block}\n".strip()



SYSTEM_PROMPT_SUMMARY_IDEAS = """You are an expert model architect.
Your task is to summarize a list of *Model Design Suggestions* into short, one-sentence insights that capture each idea’s **core conceptual innovation or purpose**.

**Guidelines:**

* Each idea must be expressed in **one sentence**.
* Focus on the **essence** of the upgrade — what it changes conceptually or what benefit it brings.
* Do **not** restate implementation details, hyperparameters, or architecture lists.
* Keep the phrasing compact, readable, and technically precise.
* Format as a **bulleted list**, each line following this pattern:
  > • [Upgrade Type or Module Name] — [Key idea or effect].

**Example Output:**
> • Heterogeneous Parallel Branches per Stage — Introduce multi-branch blocks at each stage for multiscale context capture, dense multi-stage fusion, and lightweight transformer-based context modeling, improving robustness to spatial scale variation and decision boundary refinement.
> • Cross-Stage Macro Connectivity — Redesign stage connectivity as stage1→stage3, stage2→stage4 to enable richer cross-scale information flow beyond sequential stacking.
> • High-Dilation Replacement with Large-Kernel Convs — Replace high-dilation blocks with large-kernel convolutions to reduce boundary artifacts and control receptive field growth, yielding cleaner deep features and better small-scale detail recognition.
> • Revised Stage Width Schedule — Adjust width progression to [base_dim, base_dim×4, base_dim×4, base_dim×8], expanding middle stages for deeper feature transformation and reduced bottleneck effects, enhancing accuracy potential.
> • Norm Simplification in Fusion — Remove LayerNorm in fusion and apply BatchNorm after 1×1 fusion convolution for stable channel recalibration, leading to improved training stability and generalization with comparable parameter cost.
> • Curated Multi-Branch Operators — Redesign operator composition within each multi-branch block using a curated mix of inverted, dilated, and separable convs to diversify receptive fields.


**Instruction:**
When given a model upgrade idea list, output only the summarized bullet list of conceptual ideas as described above.

ONLY return the bulleted list, nothing else.

"""

from typing import List
import json

class IdeaSummaryAgent:
    def __init__(self, llmagent):
        self.llmagent = llmagent
    def summarize_ideas(self, ideas_list: List[str]) -> List[str]:
        if not ideas_list:
            return {"summarized_ideas": None}
        ideas_list_all = [f"- Model Upgrade Idea {ii}:\n"+xx for ii, xx in enumerate(ideas_list, 1)]
        user_query_str = '\n\n'.join(ideas_list_all) 
        try:
            msg_list_temp = self.llmagent.init_msg(user_query_str, SYSTEM_PROMPT_SUMMARY_IDEAS)
            response = self.llmagent.request_llm_api_msgs(msg_list_temp)
            return({"summarized_ideas": response[0]})
        except Exception as e:
            print(f"Error occurred while summarizing ideas: {e}")
            return {"summarized_ideas": None}

PROMPT_IDEA_SELECTION = """You are given a list of model upgrade ideas, each concisely describing its design, model modification, and expected impact. Some ideas might have overlaps, redundancies, or similar functions/paradigms in upgrades.

Your task:
1. Read and understand all ideas.
2. Select exactly N ideas (N is provided by the caller) such that the selected ideas are:
   - **Diverse**: covering clearly different conceptual directions or mechanisms.
   - **Distinct**: avoiding overlap or redundancy in architecture, or design.
3. Do NOT modify, rewrite, or summarize any idea content.

Return only the selected ideas in **valid JSON format**, preserving the original text exactly as given, using this structure
{
  "idea1": { ... },
  "idea2": { ... },
  ...
}
No explanations or extra text outside the JSON.
"""
import random



class IdeaGenerationStrategy:
    def __init__(
        self,
        model_requirement_str: str,
        KNOWLEDGE_PROMPT: str|None = None,
        multiple_llm=False,
        llmagents=None,
        llm_indexer=None,
        disruptor = None,
        model_sim=None,
        use_ranking = True,
        idea2embed_cache = None,
        scores_tau = 0.8,
        analyze_prior_models = False,
        prohibit_used_concepts = True,
        kk_ideas_prohibit = 5,
        max_ideas_generate = 30,
        model_design_system_prompt = None,
        enhance_diversity = False,
    ):
        self.model_requirement_str = model_requirement_str
        self.llmagents = llmagents
        self.llm_indexer = llm_indexer
        self.multiple_llm = multiple_llm
        self.disruptor = disruptor
        if disruptor is not None:
            raise NotImplementedError("Disruptor integration is not implemented yet.")
        self.model_sim = model_sim
        self.max_ideas_generate = max_ideas_generate
        if idea2embed_cache is not None:
            self.idea2embed_cache = idea2embed_cache
        else:
            self.idea2embed_cache = {}

        self.KNOWLEDGE_PROMPT = KNOWLEDGE_PROMPT
        if model_design_system_prompt is None:
            if analyze_prior_models:
                model_design_system_prompt = MODEL_DESIGN_SYSTEM_PROMPT_2phase_new
            else:
                model_design_system_prompt = MODEL_DESIGN_SYSTEM_PROMPT
        if self.KNOWLEDGE_PROMPT:
            self.system_prompt = model_design_system_prompt + "\n\n" + self.KNOWLEDGE_PROMPT
        else:
            self.system_prompt = model_design_system_prompt
        self.model_design_system_prompt = model_design_system_prompt
        self.use_ranking = use_ranking
        self.SCORES_TAU = scores_tau
        self.analyze_prior_models = analyze_prior_models
        self.prohibit_used_concepts = prohibit_used_concepts
        self.kk_ideas_prohibit = kk_ideas_prohibit
        self.enhance_diversity = enhance_diversity
    
    def get_idea_embedding(self, idea_str):
        if idea_str in self.idea2embed_cache:
            return self.idea2embed_cache[idea_str]
        if self.model_sim is None:
            self.idea2embed_cache[idea_str] = ''
            return ''
        emb = self.model_sim.encode(idea_str, convert_to_tensor=True)
        self.idea2embed_cache[idea_str] = emb
        return emb
    
    def _select_diverse_ideas(self, generated_ideas: List[str], KK: int, used_ideas_example: List[str] = None) -> List[str]|None:
        ii = 0
        selected_ideas = None
        while ii<3:
            ii += 1
            try:
                if self.multiple_llm:
                    llmagent = self.llmagents[next(self.llm_indexer)]
                else:
                    llmagent = self.llmagents
                
                ideas_list_all = [f"- "+xx for ii, xx in enumerate(generated_ideas, 1)]
                user_query_str = '\n\n'.join(ideas_list_all)
                ideas_used_list_for_selection_subset = None
                kk_subset = self.kk_ideas_prohibit
                if self.prohibit_used_concepts:
                    if used_ideas_example is not None:
                        if len(used_ideas_example) > kk_subset:
                            ideas_used_list_for_selection_subset = random.sample(used_ideas_example, kk_subset)
                        else:
                            ideas_used_list_for_selection_subset = used_ideas_example
                    else:
                        if self.idea2embed_cache:
                            ideas_used_list_for_selection_subset = list(self.idea2embed_cache.keys())[::-1][:kk_subset]
                    if ideas_used_list_for_selection_subset:
                        used_ideas_all = [f"- "+xx for ii, xx in enumerate(ideas_used_list_for_selection_subset, 1)]
                        user_query_str += '\n\n**Avoid Overlapping with Previously Used Ideas**\nRule: The selected ideas must be completely distinct from all previously used ones. Preference is given to ideas that differ substantially in design concepts, paradigms, and architectures from prior selections. Below are the previously used ideas:\n' + '\n\n'.join(used_ideas_all)

                user_query_str += f'\n-----\n\nSelect exactly {KK} ideas that are diverse and distinct. Return only the selected ideas in valid JSON format.'
                msg_list_temp = llmagent.init_msg(user_query_str, PROMPT_IDEA_SELECTION)
                response_temp = llmagent.request_llm_api_msgs(msg_list_temp)
                content_pure = get_get_brace_string(response_temp[0])
                model_design_json = json.loads(content_pure)
                if model_design_json:
                    temp = [str(xx) for xx in model_design_json.values()]
                    selected_ideas = [xx for xx in temp if isinstance(xx, str) and len(xx.strip()) > 50]
                    if selected_ideas:
                        return selected_ideas
            except Exception as e:
                print('select diverse idea: ', e)
        return selected_ideas

    def _generate_ideas(self, model_versions_str = None, KK = 5, supl_text = "", disruptor_k = 0, user_supl_text = ""):
        user_query_str = get_model_design_prompt(self.model_requirement_str, model_versions_str, KK, supl_text) + user_supl_text
        idea_inits = None
        content_pure = None
        additional_context = ""
        if self.disruptor is not None and disruptor_k > 0:
            additional_context = '\n'+self.disruptor.get_disruptor_ideas_str(disruptor_k)+'\n'
        print('additional_context:', additional_context)
        ii = 0
        while ii < 3:
            ii += 1
            try:
                if self.multiple_llm:
                    llmagent = self.llmagents[next(self.llm_indexer)]
                else:
                    llmagent = self.llmagents
                
                msg_list_temp = llmagent.init_msg(user_query_str, self.system_prompt + additional_context)
                response_temp = llmagent.request_llm_api_msgs(msg_list_temp)
                # print('==================')
                # print(response_temp)
                content_pure = get_get_brace_string(response_temp[0])
                model_design_json = json.loads(content_pure)
                if model_design_json:
                    # print(model_design_json)
                    temp = [str(xx) for xx in model_design_json.values()]
                    idea_inits = [xx for xx in temp if isinstance(xx, str) and len(xx.strip()) > 50]
                    # return({"ideas_generated": idea_inits})
                    if idea_inits:
                        break
            except Exception as e:
                print('init idea: ', e)
        # print(idea_inits)
        # print('----')
        return idea_inits
    
    def _generate_ideas_2phase(self, model_versions_str = None, KK = 5, supl_text = "", disruptor_k = 0, user_supl_text = ""):
        user_query_str = MODEL_DESIGN_USER_PROMPT_phase1(self.model_requirement_str, model_versions_str, KK, supl_text) + user_supl_text
        idea_inits = None
        content_pure = None
        response_temp_phase1 = None
        additional_context = ""
        if self.disruptor is not None and disruptor_k > 0:
            additional_context = '\n'+self.disruptor.get_disruptor_ideas_str(disruptor_k)+'\n'
        print('additional_context:', additional_context)
        ii = 0
        while ii < 3:
            ii += 1
            try:
                if self.multiple_llm:
                    llmagent = self.llmagents[next(self.llm_indexer)]
                else:
                    llmagent = self.llmagents
                
                msg_list_temp = llmagent.init_msg(user_query_str, self.system_prompt + additional_context)
                response_temp_phase1 = llmagent.request_llm_api_msgs(msg_list_temp)
                user_query_phase2 = MODEL_DESIGN_USER_PROMPT_phase2_new + f'\nPropose **{KK} distinct model design/modification ideas** following the specification in the system prompt.'
                msg_list_temp = llmagent.update_msg(msg_list_temp, response_temp_phase1[0], user_query_phase2)
                break
            except Exception as e:
                print('attempt failed in phase 1: ', e)
        if response_temp_phase1 is None:
            print('phase 1 failed')
            return idea_inits
            
        ii = 0
        while ii < 3:
            ii += 1
            try:
                response_temp_phase2 = llmagent.request_llm_api_msgs(msg_list_temp)
                content_pure = get_get_brace_string(response_temp_phase2[0])
                model_design_json = json.loads(content_pure)
                if model_design_json:
                    temp = [str(xx) for xx in model_design_json.values()]
                    idea_inits = [xx for xx in temp if isinstance(xx, str) and len(xx.strip()) > 50]
                    # return({"ideas_generated": idea_inits})
                    if idea_inits:
                        break
            except Exception as e:
                print(response_temp_phase2[0])
                print('attempt failed in phase 2: ', e)
        return idea_inits

    def generate_ideas(self, model_versions_str = None, KK = 5, supl_text = "", disruptor_k = 0, ideas_used_list_for_generation = None, ideas_used_list_for_ranking = None, ideas_used_list_for_selection = None):
        if self.enhance_diversity:
            KK_augment = min(int(KK*2.5), self.max_ideas_generate)
        else:
            KK_augment = KK
        user_supl_text = ""
        kk_subset = self.kk_ideas_prohibit
        ideas_used_list_for_generation_subset = None
        if self.prohibit_used_concepts:
            if ideas_used_list_for_generation is not None:
                if len(ideas_used_list_for_generation) > kk_subset:
                    ideas_used_list_for_generation_subset = random.sample(ideas_used_list_for_generation, kk_subset)
            else:
                if self.idea2embed_cache:
                    ideas_used_list_for_generation_subset = list(self.idea2embed_cache.keys())[::-1][:kk_subset]
            if ideas_used_list_for_generation_subset:
                used_ideas_all = [f"- "+xx for ii, xx in enumerate(ideas_used_list_for_generation_subset, 1)]
                user_supl_text += '\n\n**Avoid Overlapping with Previously Used Ideas**\nRule: New generated ideas must be completely distinct from all previously used ones. Preference is given to ideas that differ substantially in design concepts, paradigms, and architectures from prior selections. Below are the previously used ideas:\n' + '\n\n'.join(used_ideas_all)
        idea_inits0 = None
        idea_inits = None
        if self.analyze_prior_models:
            idea_inits0 = self._generate_ideas_2phase(model_versions_str, KK_augment, supl_text, disruptor_k, user_supl_text)
        else:
            idea_inits0 = self._generate_ideas(model_versions_str, KK_augment, supl_text, disruptor_k, user_supl_text)
        if not self.enhance_diversity:
            print('Idea generation: no further diversity enhancement')
            idea_inits = idea_inits0
            if idea_inits0:
                final_ideas = idea_inits0[::-1]
            else:
                final_ideas = idea_inits0
            return ({"ideas_generated": final_ideas, "all_ideas_generated": idea_inits})

        if idea_inits0:
            if self.model_sim is None:
                idea_inits = self._select_diverse_ideas(idea_inits0, KK, used_ideas_example = ideas_used_list_for_selection)
            elif len(self.idea2embed_cache) == 0:
                idea_inits = self._select_diverse_ideas(idea_inits0, KK, used_ideas_example = ideas_used_list_for_selection)
            else:
                KKtemp = min(int(KK*1.25), self.max_ideas_generate)
                idea_inits = self._select_diverse_ideas(idea_inits0, KKtemp, used_ideas_example = ideas_used_list_for_selection)
        
        if idea_inits0 and isinstance(idea_inits0, list):
            print(len(idea_inits0))
        if idea_inits and isinstance(idea_inits, list):
            print(len(idea_inits))

        
        all_ideas_generated = None
        if idea_inits and isinstance(idea_inits, list): # added on Nov 18
            rawlen = len(idea_inits)
            print('init idea sucessfully')
            if self.use_ranking and self.model_sim is not None:
                if ideas_used_list_for_ranking is None:
                    if len(self.idea2embed_cache) == 0:
                        final_inits = []
                        for xx in idea_inits:
                            if len(xx.strip()) > 50:
                                _ = self.get_idea_embedding(xx)
                                final_inits.append(xx)
                        return ({"ideas_generated": final_inits})
                    ideas_used_list_emb = torch.stack([emb for emb in self.idea2embed_cache.values()])
                else:
                    ideas_used_list_emb = torch.stack([self.get_idea_embedding(idea_str) for idea_str in ideas_used_list_for_ranking if len(idea_str.strip())>50])

                ideaid2sim = {}
                for idx in range(len(idea_inits)):
                    idea_emb = self.get_idea_embedding(idea_inits[idx])
                    temp = torch.sum(idea_emb.unsqueeze(0) * ideas_used_list_emb, 1)
                    temp = torch.sort(temp, descending = True).values
                    print('==similarity scores: ', temp[:9])
                    similarity_score = temp[:min(20, max(5, int(0.1*len(temp))))].mean().item()
                    ideaid2sim[idx] = similarity_score
                ideas_pairs = []
                for idx, scores in ideaid2sim.items():
                    ideas_pairs.append( (scores, idea_inits[idx])) 
                
                all_ideas_generated = [xx[1] for xx in ideas_pairs]
                if ideas_pairs:
                    ideas_pairs = sorted(ideas_pairs, key=lambda x:x[0]) # 0.7, 0.8, ...
                    if ideas_pairs[0][0] >= self.SCORES_TAU:
                        print('too similar ideas')
                        ideas_pairs = ideas_pairs[:1] # if too similar, keep only 1 idea
                    else:
                        ideas_pairs = [f for f in ideas_pairs if f[0] < self.SCORES_TAU]

                final_ideas = [xx[1] for xx in ideas_pairs]
                final_scores = [xx[0] for xx in ideas_pairs]
                final_ideas = final_ideas[:KK]

                for tempstr in all_ideas_generated:
                    if tempstr not in final_ideas and tempstr in self.idea2embed_cache:
                        del self.idea2embed_cache[tempstr]
                
                # Note: will use .pop to get the idea from the list
                return ({"ideas_generated": final_ideas[::-1], "all_ideas_generated": all_ideas_generated})
            else:
                rawlen = len(idea_inits)
                idea_inits0 = idea_inits[:KK]
                final_ideas = idea_inits0[::-1]
                return ({"ideas_generated": final_ideas, "all_ideas_generated": idea_inits})
                 

        print('init idea failed')
        return({"ideas_generated": idea_inits, "all_ideas_generated": all_ideas_generated})
