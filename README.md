# pyMLFCI

Machine learning tools for modeling quantum geometry in fractional Chern insulators from single-particle form factors.

This repository accompanies the paper:

**Modeling Quantum Geometry for Fractional Chern Insulators with unsupervised learning**  
Ang-Kun Wu, Louis Primeau, Jingtao Zhang, Kai Sun, Yang Zhang, Shi-Zeng Lin  
arXiv:2510.03018  
https://arxiv.org/abs/2510.03018

This repository provides the model code, training scripts, and analysis scripts associated with the study. The datasets generated and/or analyzed in the paper are not included in this repository because of their large size and are available from the corresponding author on reasonable request. Pretrained model checkpoints used in the study are also not distributed in this repository.

## Overview

Fractional Chern insulators (FCIs) in moire and lattice systems provide a route to strongly correlated topological phases beyond idealized Landau-level physics. This project studies whether machine learning can learn physically meaningful structure directly from form factors, which encode the quantum geometry of topological flat bands.

In this codebase, we use unsupervised learning, primarily variational autoencoders (VAEs), to:

- learn low-dimensional latent representations of form factors
- distinguish FCI and non-FCI regimes from the learned structure
- generate new form factors from the latent space
- interpolate between known form factors to explore nearby quantum geometries
- compare learned representations with more conventional approaches such as PCA and simple supervised baselines

The repository also includes exploratory diffusion-model code and auxiliary analysis scripts.

## Associated Paper

If you use this repository in your work, please cite:

> A.-K. Wu, L. Primeau, J. Zhang, K. Sun, Y. Zhang, and S.-Z. Lin,  
> "Modeling Quantum Geometry for Fractional Chern Insulators with unsupervised learning,"  
> arXiv:2510.03018, 2025.

## Code and Data Availability

This repository contains:

- model implementations
- training scripts
- evaluation and analysis scripts

The full datasets used in the associated study are not included here. They were omitted because of their large size and are available from the corresponding author on reasonable request.

Pretrained model checkpoints used in the paper are also not included in this repository. Some scripts support loading checkpoints from a local `checkpoints/` directory, but users should generally expect to train models themselves or provide local checkpoint files.

## Repository Structure

```text
pyMLFCI/
├── src/                # training, evaluation, and analysis scripts
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
```

Important scripts in `src/`:

- `utils.py`: data loading, dataset wrappers, and preprocessing helpers
- `VAEclass.py`: VAE model definitions and training utilities
- `VAEformFactorTrain.py`: train a VAE on form-factor data
- `VAEformFactorTest.py`: latent-space analysis, reconstruction, sampling, and interpolation
- `TrainsVAEFF.py`: alternate VAE training script for larger parameter sweeps
- `PCA.py`, `PCAclean.py`, `VAEPCAformfactor.py`: PCA-based analysis
- `LinearModelClassifier.py`, `SVMclassifier.py`, `NNclassifier.py`: baseline supervised classifiers
- `DDPMclass.py`, `DDPMtest.py`, `SimpleDiffusion.py`: diffusion-model experiments

## Methods Implemented

The repository contains several generative model variants for form-factor learning, including:

- fully connected VAEs
- convolutional VAE variants
- transformer-based VAE variants
- hybrid convolution-transformer VAE variants

It also includes PCA-based analyses, baseline supervised classifiers, and exploratory diffusion-model experiments.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/angkunwu/pyMLFCI
cd pyMLFCI
```

### 2. Create a Python environment

For example, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Requirements

The training and analysis scripts expect form-factor data stored in CSV files. These datasets are not bundled with the repository due to the large size.

The main data loader constructs datasets from parameter sweeps over quantities such as:

- `alpha`
- `c0` variations at selected reference points

It also reads labels from corresponding exact-diagonalization output files containing an `IsFCI` column.

### Path Note

Some scripts assume external dataset locations. In the current implementation, the default loader uses a local path of the form:

```python
folder="~/QuarticCrossing/data/"
```

and then appends system-size-specific subfolders such as `35data/` or `46data/`.

If data loading fails, the first thing to check is whether your local dataset layout matches the expected directory structure. Reproducing the full experiments requires access to the external CSV datasets used in the study.

## Quick Start

### Train a VAE on form-factor data

```bash
python -m src.VAEformFactorTrain
```

This script:

- loads form-factor data
- optionally filters to FCI samples
- builds a PyTorch dataset
- trains a VAE model
- reports reconstruction and KL losses

### Analyze a trained model

```bash
python -m src.VAEformFactorTest
```

This script can be used to:

- load and analyze a trained model checkpoint
- inspect latent encodings
- visualize reconstructed form factors
- sample new form factors from latent space
- interpolate between latent points

Before running it, you may need to update the checkpoint path in the script or provide your own trained model locally.

### Run a larger training configuration

```bash
python -m src.TrainsVAEFF
```

### Run PCA or baseline classifiers

Examples:

```bash
python -m src.PCAclean
python -m src.LinearModelClassifier
python -m src.SVMclassifier
python -m src.NNclassifier
```

## Typical Workflow

A typical use pattern is:

1. obtain access to the external CSV form-factor dataset
2. choose a lattice size such as `(Nx, Ny) = (3, 5)` or `(4, 6)`
3. train a VAE using one of the training scripts
4. inspect the latent space and reconstructions
5. generate or interpolate new form factors
6. compare with PCA or simple supervised baselines

## Notes and Current Status

This repository is organized as a research codebase rather than a polished software package. In particular:

- many scripts are standalone experiments instead of a unified command-line interface
- several file paths and parameters are hard-coded for specific datasets
- some scripts assume locally available trained checkpoints
- some scripts are exploratory and may require small local adjustments before reuse

This is typical for a paper companion repository, but it is worth stating explicitly so users know what to expect.

## Citation

If this repository contributes to your research, please cite:

```bibtex
@article{wu2025modeling,
  title={Modeling Quantum Geometry for Fractional Chern Insulators with unsupervised learning},
  author={Wu, Ang-Kun and Primeau, Louis and Zhang, Jingtao and Sun, Kai and Zhang, Yang and Lin, Shi-Zeng},
  journal={arXiv preprint arXiv:2510.03018},
  year={2025}
}
```

## Contact

For questions about the code or requests for dataset access, please open an issue or contact the authors through the paper information on arXiv.
