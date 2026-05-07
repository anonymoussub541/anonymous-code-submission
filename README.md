# EvoTreeNAD Reproducibility Repository

This repository contains the implementation of EvoTreeNAD, including discovery
code, task configurations, full-fidelity evaluation scripts, and the neural
architectures discovered in the reported experiments.

The repository supports two reproducibility workflows:

1. Evaluate the provided discovered architectures under the full-fidelity
   recipes used in the manuscript.
2. Run EvoTreeNAD discovery from the task interface contracts and reference
   LLM configurations.

The first workflow does not require LLM access. The second workflow requires
GPU resources and either OpenAI, Azure OpenAI, or an OpenAI-compatible local
model endpoint.


## Repository Structure

- `EvoTreeNAD/` contains the core EvoTreeNAD implementation.
- `exps_nad/` contains task interfaces, discovery configs, and reference LLM
  configs for CIFAR and MedMNIST-v2 tasks.
- `exps_eval_discovered_models/` contains full-fidelity training scripts for
  evaluating selected discovered architectures.
- `DiscoveredModelsEvoTreeNAD/` contains selected architectures discovered by
  independent EvoTreeNAD runs, with two discovered models provided per task.


## Environment Setup

From the repository root, install the required Python packages:

```bash
pip install -r requirements.txt
```

PyTorch and torchvision should match the CUDA version on the local machine. If
needed, install the appropriate PyTorch wheel first, then run the command above.

For commands that import modules from this repository, expose the repository
root:

```bash
cd /path-to-project
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

Optional LLM credentials for discovery runs are configured through environment
variables:

```bash
export OPENAI_API_KEY=your_key_here
export AZURE_OPENAI_API_KEY=your_key_here
export AZURE_ENDPOINT=your_azure_endpoint_here
export OSS20_LOCAL_ENDPOINT=http://your_local_endpoint/v1/chat/completions
```

`OSS20_LOCAL_ENDPOINT` should point to an OpenAI-compatible endpoint for a
locally hosted OSS20B model. The reference configs can be adapted to use any
available model pool with the same chat-completion interface.

An optional template is provided at `EvoTreeNAD/.env.example`. Copy it to
`EvoTreeNAD/.env` or export the same variables in the shell.

If `USE_MODEL_SIMILARITY_FOR_IDEAGEN` is enabled in a custom discovery config,
install `sentence-transformers` in addition to the default requirements.


## Task Mapping

Generic task folder names are used to avoid explicit dataset-name leakage in
our experiments.

- `fds1`: CIFAR-10
- `fds2`: CIFAR-100
- `mds1`: PathMNIST
- `mds2`: OctMNIST
- `mds3`: TissueMNIST
- `mds4`: VesselMNIST3D
- `mds5`: SynapseMNIST3D
- `mds6`: OrganMNIST3D


## Evaluate Provided Discovered Architectures

The provided architectures are in `DiscoveredModelsEvoTreeNAD/`. The examples
below evaluate one discovered model per task. To evaluate the second independent
run, replace `evotreenad_run1_best` with `evotreenad_run2_best` in both
`--modulepath` and `--modelname`.

Run evaluation commands from `exps_eval_discovered_models/`:

```bash
cd /path-to-project/exps_eval_discovered_models
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
```

### CIFAR-10

```bash
python train_cifar.py \
  --data /path/to/data \
  --data_flag=cifar10 \
  --modelclass=ImageClfModel \
  --modulepath=DiscoveredModelsEvoTreeNAD.cifar10.evotreenad_run1_best \
  --modelname=evotreenad_run1_best \
  --gpu=0 \
  --batch_size=128 \
  --epochs=500 \
  --learning_rate=0.002 \
  --eval_log_ep=2 \
  --model_depth=15 \
  --model_dim=32
```

### CIFAR-100

```bash
python train_cifar.py \
  --data /path/to/data \
  --data_flag=cifar100 \
  --modelclass=ImageClfModel \
  --modulepath=DiscoveredModelsEvoTreeNAD.cifar100.evotreenad_run1_best \
  --modelname=evotreenad_run1_best \
  --gpu=0 \
  --batch_size=128 \
  --epochs=500 \
  --learning_rate=0.001 \
  --eval_log_ep=2 \
  --model_depth=15 \
  --model_dim=32
```

### MedMNIST-v2 2D Example

```bash
python train_medmnist.py \
  --data /path/to/data \
  --data_flag=octmnist \
  --modelclass=ImageClfModel \
  --modulepath=DiscoveredModelsEvoTreeNAD.medmnist.octmnist.evotreenad_run1_best \
  --modelname=evotreenad_run1_best \
  --gpu=0 \
  --batch_size=160 \
  --epochs=60 \
  --learning_rate=0.001 \
  --eval_log_ep=1 \
  --model_depth=12 \
  --model_dim=32
```

### MedMNIST-v2 3D Example

```bash
python train_medmnist.py \
  --data /path/to/data \
  --data_flag=synapsemnist3d \
  --modelclass=Image3DClfModel \
  --modulepath=DiscoveredModelsEvoTreeNAD.medmnist.synapsemnist3d.evotreenad_run1_best \
  --modelname=evotreenad_run1_best \
  --gpu=0 \
  --batch_size=64 \
  --epochs=200 \
  --learning_rate=0.0005 \
  --eval_log_ep=5 \
  --model_depth=12 \
  --model_dim=32
```

Evaluation scripts create timestamped output folders containing logs,
checkpoints, and final metrics.


## Run EvoTreeNAD Discovery

Each task directory in `exps_nad/` contains:

- `customize_code.py`, which defines task data loaders, smoke-test loaders,
  short-budget training, and evaluation.
- `model_requirements.py`, which defines the executable model interface and
  task-specific architectural constraints.
- `config.yaml`, which stores local project paths.
- `run_config_osscode_gpt41idea.json`, a reference LLM configuration.

Before running discovery on a new machine, update the local paths in the task
`config.yaml`:

```yaml
project:
  SAVEDIR: /path-to-project/exps_nad/
  CODEDIR: /path-to-project/EvoTreeNAD/
```

Dataset locations are configured through the task `customize_code.py` files.
For CIFAR and MedMNIST experiments, update the `/home/DATA` placeholders to a
local data root if needed.

Example discovery command for CIFAR-10:

```bash
cd /path-to-project/exps_nad/fds1

python3 ../../EvoTreeNAD/main_run.py \
  --cuda 0 \
  --run_name run1 \
  --config_dir ./ \
  --customized_run_config_path ./run_config_osscode_gpt41idea.json \
  --batch_iter_num 5
```

Key arguments:

- `--cuda` selects the GPU device.
- `--run_name` names the discovery run and output folder.
- `--config_dir` points to the task directory.
- `--customized_run_config_path` points to the LLM and discovery configuration.
- `--batch_iter_num` controls the number of evolutionary batches.

The portable entry point is the direct Python command above. The shell wrapper
`exps_nad/evotreenad_run.sh` can also be used from any task directory or with
explicit path arguments.


## Discovery Outputs and Principal Lineage

Each discovery run writes generated models, evaluation outputs, node summaries,
and tree information into:

```text
exps_nad/[task]/[run_name]/
```

After each evolutionary batch, EvoTreeNAD saves the current principal lineage
as:

```text
exps_nad/[task]/[run_name]/r[batch_id]_principle_lineage.csv
```

The filename uses `principle_lineage` for compatibility with the released code.
In the manuscript, this object is referred to as the principal lineage.


## Fixed Full-Fidelity Recipe Hooks

For selected candidate files, `EvoTreeNAD/apply_reg.py` applies the fixed
task-level recipe hooks used before full-fidelity retraining.

```bash
cd /path-to-project/EvoTreeNAD
python apply_reg.py --file=../exps_nad/fds1/run1/model_100.py --add_cutmix --add_mixup
```

Task-level hook choices used in the reported experiments:

- CIFAR-10/100: `--add_cutmix --add_mixup`
- MedMNIST-v2 2D tasks: `--add_mixup`
- MedMNIST-v2 3D tasks: no augmentation flag

Processed files are written to an `rg_code/` subdirectory next to the selected
model file.


## Optional Base Architecture

The reported discovery runs start from the task interface rather than a
predefined architecture. For additional experiments, EvoTreeNAD can initialize
from a user-provided root architecture by placing `model0.py` in the task
directory before launching a discovery run.


## Reproducibility Notes

The manuscript and appendix provide the full experimental protocol, including
run counts, discovery budgets, full-fidelity recipes, generated-candidate
counts, validation and early-stopping statistics, GPU usage, token usage, and
API costs. This repository provides the code and artifacts needed to inspect,
rerun, and extend those experiments after local paths, data roots, GPU devices,
and model endpoints are configured.
