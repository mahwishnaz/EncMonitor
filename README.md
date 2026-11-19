# EncMonitor: Encrypted Cross-Silo Drift Monitoring beyond Pooled Statistics

Code for the experiments of the paper of the same title (Mahwish Naz, Hojong Choi, Hyunyoung Oh).
License: MIT (see `LICENSE`); third-party notices in `THIRD_PARTY_NOTICES`; citation in `CITATION.cff`.

Datasets, model checkpoints, cached representations and results are **not** included (see Data archive below). The public datasets are
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

## Data archive (saved partitions, calibration windows, per-window outputs)

The saved partitions, the calibration windows and the per-window outputs of the 20-seed experiments behind
Tables 4 to 7 of the article are archived with the code at https://doi.org/10.5281/zenodo.24358726. They are not part of the Git history.
The archive consists of four files, one per experiment family; unpacking all four gives one folder
`encmonitor_data/`.

| Archive | Contents | Files | Unpacked |
|---|---|---|---|
| `encmonitor_data_mnist_cifar10.tar.gz` | MNIST and CIFAR-10, seeds 2026-2045 | 190 | 219.2 MB |
| `encmonitor_data_camelyon17.tar.gz` | Camelyon17-WILDS patch split and patient split (seeds 2026-2045), slide split (seeds 2026-2030) | 190 | 270.6 MB |
| `encmonitor_data_census.tar.gz` | Census (ACSIncome) at n = 250 and n = 1,000, seeds 2026-2045, and the census pools | 131 | 81.3 MB |
| `encmonitor_data_tables_streams.tar.gz` | Summary tables, per-seed rates, chronological streams | 14 | 4.6 MB |

The archive contains only indices and computed statistics: no image, no dataset record, no representation of an
individual input and no model weight. The datasets must be obtained from their sources with the download scripts of
this repository (`scripts/download_data.py`, `experiments/census/download.py`).

### Code version

The files belong to the code tag `revision1-code-final`
(commit `6f2d6bd962fa1285dcc5910b3e00bbab3f59ac53`). The header line of each CSV file and the `RUN_INFO.json` files
name the tag `revision1-code` (commit `d51e25a71ac4bf588e56b8e2596a34487502fc0c`), with which the runs of seeds
2031-2045 and the summary tables were produced; seeds 2026-2030 come from the original runs (no `RUN_INFO.json`).
Between the two tags the experiment code changed only in comments and messages and in the encrypted Local-any
parameters, which do not enter these files; with `revision1-code-final`, MNIST seed 2026 is reproduced with
bit-identical window statistics.

Two parts of the archive were exported after the runs, with the same code and seeds, by scripts added in commit
`44449d288b2490b93e6c252e47aeaf912803dd24`; the same commit adds the archive builder and the script of the interval comparison
(Supplementary Table S14):

| Script | Writes |
|---|---|
| `tools/export_census_pools.py` | `census/pools/` (census pools and their verification) |
| `analysis/tables/export_weak_all_windows.py` | `weak_shift_records.jsonl` in `mnist/`, `cifar10/`, `camelyon17/patch_split/` |
| `tools/build_release.py` | the four archives, `CHECKSUMS.txt`, `FILELIST.tsv` |
| `analysis/tables/intervals.py` | `tables/interval_comparison.csv` (from `per_seed/per_seed_rates.csv` and the bootstrap intervals in `experiments/private_localany/records_summary.csv`) |

- **Census pools.** The census experiment draws its pools from the partition seed at run time and did not save
  them. They were exported after the runs with the pool-drawing code of `experiments/census/windows.py` and the same
  seeds. For all 20 seeds the exported pools reproduce the stored pool sizes (30 per seed) and positive rates of
  `conditions.json` exactly, and the 700 clean windows per seed (500 calibration and 200 held-out windows) recomputed
  from them reproduce `d_mix`, `d_site` and `heterogeneity` of `records.jsonl` exactly at n = 250 and n = 1,000
  (14,000 windows; maximum relative difference 0). See `census/pools/verification.json`.
- **Weak shifts at one, two or all five sites.** `analysis/tables/compute_rates.py` generates these windows in
  memory from the saved representations. They were regenerated from the saved representations with the same code
  and seeds (`analysis/tables/export_weak_all_windows.py` calls the same generator and only writes one record per
  window) and reproduce the reported per-seed rates: for all 20 seeds, the three data sets, the three site
  geometries, the five methods of `per_seed/per_seed_rates.csv` and both threshold rules, 1,620 rates were
  compared (540 of them for the weak shift at all five sites), all with identical alert counts; maximum absolute
  difference 0.

### Layout and file types

Per seed, in `encmonitor_data/<experiment>/seed_<seed>/`:

- `manifest.npz` (saved partition). Integer row indices of each site's pools: `baseline_site_<s>` (reference pool)
  and `monitor_site_<s>` (monitoring pool). MNIST and CIFAR-10 also contain `train_site_<s>` and `test_site_<s>`
  (site allocation of the training and test sets; indices into the torchvision training and test sets),
  `class_to_site` (10 x 5 class-to-site Dirichlet probabilities that define the partition), and
  `train_class_proportions`, `test_class_proportions` (5 x 10 class shares per site). These three arrays are
  site-level shares computed from the public datasets; they contain no label of an individual image and document
  how heterogeneous the partitions are. For Camelyon17 the indices are row positions in the WILDS Camelyon17 v1.0
  dataset.
- `records.jsonl` (one line per window). `condition`, `family`, `severity`, `geometry`, `window`, `window_seed`,
  `partition_seed`; the released statistics `d_mix`, `d_site`, `heterogeneity`; per-site lists of length 5:
  `contributions` (A_s), `site_energy`, `directional` (not for census), `shifted_mask`, `acc` (per-site accuracy of
  the window; Camelyon17 and census); `exact_mmd` (exact-kernel pooled MMD; empty for census); Camelyon17 also
  `d_mix_global`, `d_site_global`, `heterogeneity_global` (global-reference comparison); attribution scores `top1`,
  `topk_recall` and, except census, `ndcg`, `spearman` (shifted windows only). Rows with condition
  `clean_calibration` (500 per seed) are the **calibration windows**; `clean_heldout` (200 per seed) are the
  held-out clean windows for the false-positive rate; all other rows (50 windows per condition) are shifted windows.
- `weak_shift_records.jsonl` (MNIST, CIFAR-10, Camelyon17 patch split; 900 lines). The weak shifts of Table 4 (last
  detection row) and Supplementary Tables S5 and S6: vector shifts (0.1, 0.25, 0.5) and label-prior shifts (0.05,
  0.1, 0.2) at one site (`single`), two sites (`coherent`) or all five sites (`all`), 18 conditions x 50 windows,
  window seed = partition seed + 2000. Same fields as `records.jsonl`; `exact_mmd` and `acc` are not computed for
  these windows (null or absent), and `spearman` is null when all five sites are shifted equally. They are scored
  against the `clean_calibration` and `clean_heldout` rows of `records.jsonl` of the same seed.
- `conditions.json`: the list of shift conditions (family, severity, geometry, site directions) and run metadata
  (MNIST, CIFAR-10, Camelyon17: number of windows, run time, target class of the label-prior shift, device of the
  exact-kernel baseline, and for Camelyon17 the clean accuracy per site; census: pool sizes, positive rates and
  clean accuracy per site, kernel bandwidth).
- `RUN_INFO.json` (seeds 2031-2045 and the patient split): commit, tag, date and command of the run.

Census pools: `encmonitor_data/census/pools/seed_<seed>.npz`: `ref_site_<s>` (reference pool, 2014) and
`mon_site_<s>_<year>` (monitoring pool for 2014, the clean year, and for 2015-2018), as row positions in
`experiments/census/cache/acs_<year>.npz` written by `experiments/census/model.py`. Both window sizes use the same
pools.

### Which file belongs to which table

| Article | Summary file (`encmonitor_data/tables/`, `streams/`) | Per-window source |
|---|---|---|
| Table 4, detection rows; Figure 2 | `table1.csv` | `mnist/`, `cifar10/`, `camelyon17/patch_split/`, `camelyon17/patient_split/`, `census/n250/`, `census/n1000/` |
| Table 4, weak shift at all five sites; Figure 2(b) | `table1.csv`; per seed: `per_seed/per_seed_rates.csv` (aggregate `weak_all`) | `weak_shift_records.jsonl` of `mnist/`, `cifar10/`, `camelyon17/patch_split/` (geometry `all`), with the clean rows of `records.jsonl` |
| Table 5 (FPR, ARL0) | `table3.csv` | `clean_heldout` rows of the same folders |
| Table 6 (hospitals) | `table4.csv` | `camelyon17/` |
| Table 7 (census) | `table5.csv` | `census/n250/`, `census/n1000/`, `census/pools/` |
| Tables S1, S2 (per seed) | `table1_per_seed.csv` | as Table 4 |
| Table S5 (matched FPR) | `matched_fpr.csv` | as Table 4 |
| Table S6 (paired comparison) | `paired.csv`; per-seed differences: `paired_per_seed.csv` | as Table 4 |
| Table S14 (five seeds, t-interval and cluster bootstrap) | `interval_comparison.csv` | seeds 2026-2030 of the same folders |
| Chronological streams (Section 7.3.1, Supplementary S1.1) | `streams/summary.csv`, `streams/per_stream.csv`, `streams/autocorr.csv` | census pools; `camelyon17/patient_split/` partitions |
| All of the above | `per_seed/per_seed_rates.csv` (per seed, method and aggregate, default and matched thresholds); `per_seed/record_inventory.csv` | |

The slide split (`camelyon17/slide_split_5seeds/`, seeds 2026-2030) is the five-seed row of Tables S1, S5, S6 and
S14. The tables are recomputed from the records by the scripts in `analysis/tables/` of this repository
(`make_all_tables.py`); `interval_comparison.csv` is written by `analysis/tables/intervals.py`, which compares the
five-seed t-intervals of `per_seed_rates.csv` with the seed-cluster bootstrap intervals that
`experiments/private_localany/run.py` writes to `records_summary.csv` from the saved records. The header line of each
CSV file names the internal table number used during the analysis (for example, "Table 1" in `table1.csv`); the table
above gives the numbers of the article.

### Not included

- Ablation grid (per-window records of the single-factor ablations, about 1.3 GB, and its 120 partitions).
- Representations and derived arrays: `pieces.npz`, `baseline_clean.npz`, `monitor_clean.npz`, and the per-seed
  preprocessing, random-feature and reference-sketch files (`preprocessor.npz`, `rff_map.npz`, `baselines.npz`,
  `global_baseline.npy`).
- Model checkpoints and pretrained weights; datasets and the census cache (`acs_<year>.npz`).
- Per-window values of the chronological streams (only per-stream results were saved: `streams/per_stream.csv`).
- Camelyon17 `patient_table.csv` (patient identifiers per pool, used for Table S7).
- Cost and timing measurements (Table 4 cost rows); the five-seed analyses (CKKS replay, leakage attack, noise
  experiment, falsified residuals, clipping, harmful windows, Table S15); the TCP prototype (seeds 2026 and 2027).

### Checksums and license

`CHECKSUMS.txt` (next to the archives in the Zenodo record) lists the SHA-256 checksum of each archive and of every
file inside it (525 files); `FILELIST.tsv` lists archive, path and size of every file. To check, place the
archives and `CHECKSUMS.txt` in one folder, unpack the archives there and run `sha256sum -c CHECKSUMS.txt`.

The MIT license of this repository (`LICENSE`) also covers these files.

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
4. CKKS replay, cost grid and TCP prototype: `scripts/20`-`24` (`configs/ckks.yaml`). Outputs:
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
