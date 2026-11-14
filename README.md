# EncMonitor: Encrypted Cross-Silo Drift Monitoring beyond Pooled Statistics

Code for the experiments of the paper of the same title (Mahwish Naz, Hojong Choi, Hyunyoung Oh).
License: MIT (see `LICENSE`); third-party notices in `THIRD_PARTY_NOTICES`; citation in `CITATION.cff`.

Datasets, model checkpoints, cached representations and results are **not** included. The public datasets are
downloaded by the scripts below, and the models are trained by the scripts of this repository. The MNIST (LeNet)
and CIFAR-10 (ResNet-18) models are trained from random initialization (`scripts/03_train_model.py`). The
Camelyon17 model (`scripts/31_camelyon_train.py`) is fine-tuned from the torchvision ResNet-18 ImageNet weights
(IMAGENET1K_V1), which torchvision downloads at run time; a fine-tuned checkpoint, if released separately, starts
from these ImageNet weights.

## Environment
Python 3.11 with the packages in `requirements.txt` (exact versions of the original runs: `pip_freeze.txt`).
OpenFHE 1.5.1 and its Python bindings (openfhe-python) are built from source: https://github.com/openfheorg.
Install the package from this folder with `pip install -e .`. GPU jobs use one GPU (`CUDA_VISIBLE_DEVICES=0`).

## Data
`python scripts/download_data.py` downloads MNIST and CIFAR-10 (torchvision, into `data/`), Camelyon17-WILDS 1.0
(WILDS, into `data/wilds`) and the ACS census files (Folktables, into `experiments/census/data`).
`python scripts/download_data.py --check` only reports which datasets are present.

## Layout
- `encmonitor/`: library (RFF map, statistics, thresholds, CKKS circuit, network protocol).
- `configs/`: `main.yaml` (MNIST, CIFAR-10), `camelyon.yaml`, `census.yaml` (partition seeds of the census
  experiment), `ablation.yaml`, `ckks.yaml`. The partition seeds 2031-2045 extend the original five.
- `scripts/`: numbered steps of the main pipeline.
- `experiments/`: further experiments and analyses of the saved records (shared code in `experiments/common/`).

## Order of the runs (from this folder)
1. Models and shared artifacts (MNIST, CIFAR-10): `scripts/02`-`06`.
2. Main experiment: `scripts/14_build_screening_cache.py`, `15_run_screening.py`, `16_analyze_screening.py`,
   then `25`-`27`. Settings: `configs/main.yaml`. Outputs: `outputs/main/`.
3. Ablations: `scripts/17`-`19` (`configs/ablation.yaml`). Outputs: `outputs/ablation/`.
4. CKKS replay, cost grid and network prototype: `scripts/20`-`24` (`configs/ckks.yaml`). Outputs:
   `outputs/ckks/`, `outputs/network/`.
5. Camelyon17-WILDS: `scripts/30`-`35` and `camelyon_common.py` (`configs/camelyon.yaml`); outputs
   `outputs/camelyon/`. Slide-level split: `experiments/camelyon_slide_split/`. Patient-level split:
   `experiments/camelyon_patient_split/cache.py`, then
   `scripts/33_camelyon_windows.py --cache-dir outputs/camelyon/patient_split/camelyon17`.
6. Census data: `experiments/census/` (`download.py`, `model.py`, `windows.py`, `analyze.py`).
7. Further analyses of the saved records: the other folders in `experiments/`. Each script states its usage in
   its docstring.
8. Tables and analyses of the reported numbers (all partition seeds): `scripts/run_seeds.sh` runs the
   experiments for every seed; `analysis/` holds the analyses (weighted variants, clipping, patient split,
   falsified residual, streams, worked example, cost projection; shared code in `analysis/shared.py`) and
   `analysis/tables/make_all_tables.py` regenerates every table into `outputs/tables/`. Results of the
   analyses go to `outputs/analysis/`.
9. Reproduction check of partition seed 2026: `scripts/reproduce_seed2026.sh`, then
   `scripts/compare_seed2026.py` (identical window indices, statistics within a relative difference of 1e-6,
   identical alert decisions).

Each script reads the outputs of the earlier steps from `outputs/` (or from its own folder) and writes its own.
