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

## Names
Some internal names come from the numbering of the experiments while the code was written. They are kept so that
the code matches the saved records. What they stand for:

| Name | Where | Meaning |
|---|---|---|
| `rx_common` | `experiments/common/rx_common.py` | shared helpers that load and score the saved per-window records (thresholds: 95th percentile of the 500 clean calibration windows per data set and partition seed, as in scripts 16, 25 and 26) |
| `rx_windows` | `experiments/common/rx_windows.py` | window generator on the cached representations; copies the pool construction and window sampling of scripts 15 and 33 and returns the Gram matrix of the site residuals per window |
| `RX` | several scripts | the `experiments/` folder |
| `e3_part`, `E3_FAMILIES`, `E3_GEOMETRIES` | `experiments/normalized_weights/run.py` | the weak shifts: vector (0.1, 0.25, 0.5) and label prior (0.05, 0.1, 0.2) at one site, two sites or all five sites, 50 windows per condition; files `e3_windows.csv`, `e3_windows_per_seed.csv` |
| `E8`, `e8` | `experiments/private_localany/run.py`, `analysis/` | the module `experiments/normalized_weights/run.py` (normalized weights and the weak shifts); check keys "vs E8" compare with its output |
| `e7` | `experiments/private_localany/run.py` | the module `experiments/arl/run.py` (ARL0 of the 2-of-3 rule, `arl_2of3`) |
| `e10`, `E10` | `experiments/private_localany/ckks.py`, `keysizes.py` | the module `experiments/ckks_keyswitch/run.py` (CKKS context with BV key switching, `build_context_bv`) |
| `e1` | `experiments/private_localany/run.py` | table read from `experiments/equal_fpr/condition_full.csv` |
| `e6` | `experiments/l1_leakage/run.py` | table read from `experiments/leakage_dp/a_site_identity_per_seed.csv` |
| `e9` | `experiments/private_localany/run.py` | table read from `experiments/l1_leakage/l1_leakage_per_seed.csv` |
| `P1`, `P1C` | `analysis/`, `experiments/disclosure_counts/` | the modules `experiments/private_localany/run.py` (scoring of all methods) and `experiments/private_localany/ckks.py` (CKKS cost) |
| `e0` | `analysis/` | mean clean site energy over the calibration windows |

## Which tag produced which numbers
| Numbers | Tag |
|---|---|
| Detection rates, false-positive rates and all other per-window results of the experiments | `revision1-code` |
| Analysis tables (per-seed tables, paired and matched comparisons, weighted variants, clipping, patient split, falsified residual, streams, worked example) | `revision1-code-analysis` |
| CKKS cost of encrypted Local-any (alert-count parameters) and the cost projection to S = 5, 10, 20, 50 | `revision1-code-final` |

The tag `revision1-code-final` changes only the parameters of the encrypted Local-any alert count
(`experiments/private_localany/ckks.py`: first modulus 35 bits instead of 30); no detection result depends on them.
