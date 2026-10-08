# DISS-SleepNet

Model implementation and training scripts for DISS-SleepNet: Dual-Input Self-Supervised Learning for Label-Efficient Sleep Staging.

## Contents

- `sleepssl/`: model, dataset loaders, losses, and utilities.
- `scripts/pretrain.py`: self-supervised pretraining.
- `scripts/proxy_probe.py`: linear-probe checkpoint selection.
- `scripts/finetune.py`: downstream sleep staging.
- `experiments/run_main.py`: main experiment runner.

Baseline comparison and ablation scripts are not included.

## Setup

Use a Python environment with PyTorch. Install the dependencies:

```bash
pip install -r requirements.txt
```

The Mamba configuration used in the paper additionally requires compatible `mamba-ssm` and `causal-conv1d` packages and a supported CUDA environment. The fallback sequence implementation is not the paper configuration.

## Usage

Run commands from the repository root:

```bash
python -m scripts.pretrain --help
python -m scripts.proxy_probe --help
python -m scripts.finetune --help
python -m experiments.run_main --help
```

Before running the experiment runner, replace `/path/to/python`, `/path/to/project`, and `/path/to/output/main_experiments` in `experiments/run_main.py` with your local paths. Update its dataset, label-subset, and existing-checkpoint paths for your files. The runner retains the original experiment workflow, including reuse of existing Sleep-EDF-20 checkpoints.

Raw datasets, preprocessing scripts, fold and label-subset files, and trained checkpoints are not included. The training scripts require prepared HDF5 metadata and aligned raw/rhythm NumPy memmap files. Obtain Sleep-EDF Expanded and ISRUC-Sleep from their official providers and prepare the input files before training.

This repository provides the model and training implementation, rather than an end-to-end raw-data reproduction package.
