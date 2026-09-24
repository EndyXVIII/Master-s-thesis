# Aligning 3D and Text Latent Spaces

**An Empirical Study of Alignment Methods**

Master's Thesis — MSc in Computer Science, University of Padova
Author: **Endi Hysa** · Supervisor: **Dr. Marco Fiorucci**
Department of Mathematics, University of Padova

[![Thesis PDF](https://img.shields.io/badge/thesis-PDF-9B0013.svg)](./thesis.pdf)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## Overview

This thesis investigates whether 3D geometric representations and natural language representations — learned by **independently trained**, uni-modal encoders — share enough intrinsic structure to be aligned *after the fact*, purely through mathematical post-processing, with no joint retraining of either encoder.

Two families of alignment methods are evaluated end-to-end on a shared dataset and protocol:

- **Supervised methods** (linear and non-linear), which use thousands of known 3D–text correspondences.
- **Unsupervised Optimal Transport**, specifically **Gromov-Wasserstein (GW)**, which uses no cross-modal correspondence at all — only the internal relational structure of each space independently.

The central finding is that unsupervised relational transport **fails systematically** to recover a meaningful alignment in this setting, and the thesis provides a rigorous, multi-angle explanation for *why*: independently trained 3D and text latent spaces do not satisfy the structural isometry that methods like Gromov-Wasserstein require in order to succeed without supervision.

## Key Results

| Method | Supervision | Top-5 Accuracy (PointNet × CLIP) |
|---|---|---|
| CCA + Affine (linear baseline) | Supervised (30k anchors) | **35.7%** |
| Riemannian Metric Learning | Supervised (30k anchors) | 27.2% |
| Kernel CCA | Supervised (30k anchors) | 15.8% |
| Fused Gromov-Wasserstein | Semi-supervised (1k anchors) | 4.4% |
| Sliced Gromov-Wasserstein | Unsupervised | 3.2% |
| Low-Rank Gromov-Wasserstein | Unsupervised | 1.2% |
| Random baseline | — | ≈1.6% |

The best unsupervised method reaches roughly **8× below** the supervised baseline — a gap explained, and independently verified, through:

- A **thirteen-test diagnostic ablation study**, systematically ruling out implementation bugs, hyperparameter misconfiguration, dataset artefacts, and library-specific issues as explanations for the failure.
- **Structural evidence** (Centered Kernel Alignment scores, a diagnostic Gromov-Wasserstein cost, and transport-plan concentration analysis) confirming the absence of genuine structural isometry between independently trained 3D and text encoders.
- An **external, independent verification** using an exact Quadratic Assignment Problem solver (Schnaus et al., CVPR 2025, *"It's a (Blind) Match!"*) on category-aggregated data, confirming the same qualitative failure with a completely different optimisation method.

Full details, all thirteen ablation tests, and the complete set of results across all six 3D–text encoder combinations are reported in the thesis document.

## Repository Structure

```
.
├── thesis.pdf                  # Full thesis document
├── src/
│   ├── features/                # Feature extraction pipeline (PointNet, SparseConv, CLIP, RoBERTa, BERT)
│   ├── baseline/                 # CCA + Affine supervised baseline
│   ├── nonlinear/                # Kernel CCA and Riemannian Metric Learning
│   ├── gromov_wasserstein/       # SGW, RISGW, SS-RISGW, FGW, LR-GW implementations
│   ├── ablation/                 # Thirteen diagnostic ablation tests
│   ├── diagnostics/               # CKA and GW-cost-as-compatibility-metric analysis
│   └── blind_match/              # External QAP-based verification
├── results/
│   └── *.json                    # Raw numerical results underlying every table and figure in the thesis
├── figures/
│   └── *.pdf                     # Figures reproduced in the thesis
├── environment/
│   ├── ott_env.yml               # Conda environment for JAX / OTT-JAX / POT experiments
│   └── itsamatch_env.yml         # Conda environment for the external blind-matching verification
└── README.md
```

*(Adjust folder names above to match the actual layout of this repository before publishing.)*

## Dataset

Experiments use **46,675** paired 3D–text objects, built from [Objaverse](https://objaverse.allenai.org/) with captions from the [Cap3D](https://github.com/crockwell/Cap3D) framework. Of these, 30,000 pairs serve as anchors for supervised training, and 500 are held out exclusively for evaluation. Raw extracted features are not redistributed in this repository due to their size; instructions for regenerating them from the public Objaverse and Cap3D sources are provided in `src/features/`.

## Encoders

| Modality | Encoder | Output Dimension | Notes |
|---|---|---|---|
| 3D | PointNet | 1024 | Pre-trained |
| 3D | SparseConv | 512 | Random, untrained weights (deliberate lower-bound control) |
| Text | OpenCLIP (ViT-bigG-14) | 1280 | Text tower only |
| Text | RoBERTa-base | 768 | — |
| Text | BERT-base-uncased | 768 | — |

## Reproducing the Experiments

```bash
# Clone the repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# Set up the environment (JAX / OTT-JAX / POT experiments)
conda env create -f environment/ott_env.yml
conda activate ott_env

# Run feature extraction (requires Objaverse + Cap3D access)
python src/features/extract_all.py

# Reproduce the supervised baseline
python src/baseline/run_cca_affine.py

# Reproduce the Gromov-Wasserstein experiments
python src/gromov_wasserstein/run_all_variants.py

# Run the full thirteen-test ablation study
python src/ablation/run_all_tests.py
```

*(Update commands above to match the actual entry-point scripts in this repository.)*

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{hysa2026aligning,
  author  = {Hysa, Endi},
  title   = {Aligning 3D and Text Latent Spaces: An Empirical Study of Alignment Methods},
  school  = {University of Padova},
  year    = {2026},
  type    = {Master's Thesis},
  note    = {Supervisor: Marco Fiorucci}
}
```

## Acknowledgements

This work builds directly on the supervised alignment framework proposed by Hadgi et al. in *["Escaping Plato's Cave"](https://arxiv.org/abs/2503.05283)* (CVPR 2025), and evaluates it alongside the Gromov-Wasserstein theory of Vayer et al. and the diagnostic methodology of Li et al. The external verification uses the exact QAP solver introduced by Schnaus, Araslanov, and Cremers in *["It's a (Blind) Match!"](https://arxiv.org/abs/2503.24129)* (CVPR 2025).

## License

This repository is released under the [MIT License](./LICENSE). See the `LICENSE` file for details.

## Contact

Endi Hysa — [endi.hysa@studenti.unipd.it](mailto:endi.hysa@studenti.unipd.it)
