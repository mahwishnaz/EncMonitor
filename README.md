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
(commit `0055720a4ba6a3b8c8bf9113c0aaea3589a7376f`). The per-window runs of seeds 2031-2045 come from the tag
`revision1-code` (commit `6c2843930fd44e9135e0aae437857e8e3e8c8542`), which their `RUN_INFO.json` files name; seeds
2026-2030 come from the original runs (no `RUN_INFO.json`). The summary tables were computed on the `revision1-code`
checkout (`6c28439`; the header line of each CSV file names it by its former hash `d51e25a`, see below) with table code committed later: in the tag
`revision1-code-analysis` (commit `8426b3bbde99b1649a1936e6d87b4152db504643`) and, for the archive scripts, in commit `907e5b4`.
Between `revision1-code` and `revision1-code-final` the experiment code changed only in comments and messages and in the encrypted Local-any
parameters, which do not enter these files; with `revision1-code-final`, MNIST seed 2026 is reproduced with
bit-identical window statistics.

Two parts of the archive were exported after the runs, with the same code and seeds, by scripts added in commit
`907e5b404fe9a0b260460bf580fbe257b9bc0560`; the same commit adds the archive builder and the script of the interval comparison
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
   falsified residual, streams, cost projection; shared code in `analysis/shared.py`) and
   `analysis/tables/make_all_tables.py` regenerates every table into `outputs/tables/`. Results of the
   analyses go to `outputs/analysis/`.
   Worked example of Figure 1: `analysis/worked_example.py` reads the saved per-window records
   (`seed_<seed>/records.jsonl`) and needs nothing else. With the unpacked data archive:
   `python analysis/worked_example.py encmonitor_data/mnist` (or `outputs/main/mnist`). For each seed it calibrates
   τ_mix, τ_site and c on the 500 calibration windows (95th percentile, "higher"), takes the first window of
   MNIST seed 2026, condition `vector2__opposing` (S = 5, m = 64), in which the EncMonitor alert fires and the
   pooled alert is silent, and prints D_mix/τ_mix, D_site/τ_site, score/c, A_s, the residual norms and the
   alert rates of EncMonitor and the pooled score for seed 2026 and their mean over the 20 seeds.
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
| `gate` (configuration key), `gate()`, `axis_gate()` | `configs/*.yaml`; scripts 16, 19, 21 | pass criterion fixed before the results (cancellation criterion in `main.yaml` and `camelyon.yaml`, robustness criterion of the ablations, accuracy criterion of the CKKS replay); the functions evaluate it |
| `family_gate.csv`, `axis_gate.csv` | outputs of scripts 16 and 19 | result of the pass criterion per shift family and per ablation axis |
| `g4`, `g5` | `scripts/23_net_session.py` | `g4`: in-process CKKS results read from `--ckks-jsonl`, compared with the network session; `g5`: default `--run-id` of a session |
| `A1_natural_opposing_mixed` | `experiments/common/rx_common.py` and the scripts that score records; columns and aggregate values of their CSV files | all shift families except the vector shift, at the opposing or mixed geometry (32 conditions for MNIST and CIFAR-10, 36 for Camelyon17); for census the injected label-prior and age-mix shifts at opposing/mixed (16). `A1` in the check keys of `experiments/private_localany/run.py` and in the columns `tpr_*_A1` of `experiments/leakage_dp/` is the same aggregate |
| `A2_all_opposing_mixed`, `A3_all_shifted`, `A4_single`, `A5_coherent` | `experiments/common/rx_common.py` | opposing/mixed including the vector shift; every shifted condition; one shifted site; two sites shifted in the same direction (census: label-prior and age-mix shifts only) |
| `vector_all`, `vector_opposing_mixed` | `analysis/clipping_check.py`, `experiments/census/analyze.py` | vector shift at all geometries; vector shift at opposing/mixed |
| `best_case_rows`, `best_case_rows_of_paper` | `experiments/arl/`, `experiments/normalized_weights/`, `experiments/camelyon_slide_split/analyze.py` | per shift family, the opposing/mixed condition with the highest hidden-drift rate (selection of script 16, `experiments/equal_fpr/best_rows.csv`); not a table of the article |
| `"patch_level (paper)"`, `"slide_level (E5)"` | `experiments/camelyon_slide_split/analyze.py`, column `split` of its CSV files | Camelyon17 patch split of the main experiment; slide-level split |
| `paper`, `paper_profiles` | `experiments/ckks_params/run.py` | the CKKS contexts of the EncMonitor profile (`build_context`) for m = 32, 64, 128, 256 |
| `ckks_d0`, `bgv_t65537_d0` | `experiments/private_localany/ckks.py`, `analysis/cost_projection.py` | encrypted Local-any count of alert bits with CKKS at depth 0 (first modulus 35 bits, scaling 25 bits), and with BGV at depth 0 (plaintext modulus 65537) |
| `table1`, `table3`, `table4`, `table5` (scripts, `.csv`, `.md` and the title in their header line) | `analysis/tables/`, `encmonitor_data/tables/` | Table 4 (detection rows), Table 5, Table 6 and Table 7 of the article; `table1_per_seed` = Tables S1, S2 |
| `t9`, `table9_extra.csv` | `scripts/25_extra_analyses.py` | best-case conditions (per data set and family, the opposing/mixed condition with the highest hidden-drift rate) with detection, joint calibration, 2-of-3 delay and ranking; not a table of the article |
| `F3_detection_<dataset>.png`, `F4_hidden_drift_<dataset>.png`, `F5_attribution_<dataset>.png` | `scripts/16_analyze_screening.py` | figures of the detection rate and of the hidden-drift rate against severity, and of site attribution; not figures of the article |
| `F6_rff_tradeoff.png` | `scripts/19_analyze_ablation.py` | figure of the RFF-dimension trade-off of the ablations; not a figure of the article |
| `private_localany`, `fig_rff_r3.py` | `experiments/` | encrypted Local-any (encrypted alert bits), its scoring and cost; `fig_rff_r3.py` draws the RFF-dimension trade-off figure `fig_rff_tradeoff_r3.pdf` |
| check keys "vs E1", "E3 ... vs E8", "vs E9"; keys `E3/...` | `experiments/private_localany/run.py` (`checks.json`), `experiments/normalized_weights/run.py` (`decision_differences.csv`) | E1: `experiments/equal_fpr/` (`condition_full.csv`); E3: the weak shifts (as `e3_part`); E8: `experiments/normalized_weights/run.py`; E9: `experiments/l1_leakage/` |

## Which tag produced which numbers
| Numbers | Tag |
|---|---|
| Detection rates, false-positive rates and all other per-window results of the experiments | `revision1-code` |
| Analysis tables (per-seed tables, paired and matched comparisons, weighted variants, clipping, patient split, falsified residual, streams) | computed on the `revision1-code` checkout (`6c28439`; former hash `d51e25a` in the CSV headers) with table code committed later in `revision1-code-analysis` (and in commit `907e5b4` for the archive scripts) |
| CKKS cost of encrypted Local-any (alert-count parameters) and the cost projection to S = 5, 10, 20, 50 | `revision1-code-final` |
| Worked example of Figure 1 (`analysis/worked_example.py`) | `v0.1.1` |

The tag `revision1-code-final` changes only the parameters of the encrypted Local-any alert count
(`experiments/private_localany/ckks.py`: first modulus 35 bits instead of 30); no detection result depends on them.

The worked example (`analysis/worked_example.py`) only reads the saved records. The MNIST records of seed 2026 come
from the original runs and have no `RUN_INFO.json`, so no tag is named for them; the `RUN_INFO.json` files of seeds
2031-2045 name `revision1-code`.

The data archives written for the article name the former hash `d51e25a` of `revision1-code` (header line of each
CSV file and `RUN_INFO.json`); no other former hash occurs in them. The commit history was later rewritten, so the same
commits now have the hashes below; their file contents are identical, and the
tags are unchanged and point to the same code.

| Tag | Former hash | Hash |
|---|---|---|
| `revision1-code` | `d51e25a71ac4bf588e56b8e2596a34487502fc0c` | `6c2843930fd44e9135e0aae437857e8e3e8c8542` |
| `revision1-code-analysis` | `5a54f2fca71c66c7d5582663abfde79ba6fff07c` | `8426b3bbde99b1649a1936e6d87b4152db504643` |
| `revision1-code-final` | `6f2d6bd962fa1285dcc5910b3e00bbab3f59ac53` | `0055720a4ba6a3b8c8bf9113c0aaea3589a7376f` |
| (archive scripts) | `44449d288b2490b93e6c252e47aeaf912803dd24` | `907e5b404fe9a0b260460bf580fbe257b9bc0560` |
| `v0.1.0` | `75d0e2238f39c9c6c64640ff01c3468c60da53f0` | `ad0e712d18eed03d66d490c5c5709b4229010ca4` |
| (comments and messages) | `e7b6e1d42338f7f7f0658556de2045c740f8dcb2` | `3418fd1de89c3d88c478c66c892ea972c0e21a02` |
| (version 0.1.1 wording) | `6d1c6a851fb61dad91753b8cdace2fd0991bf5d3` | `c0612f07bc3c53c77bb443d9d14f06030ff141d2` |

## Version v0.1.1
v0.1.1 differs from v0.1.0 only in comments, docstrings, help texts and printed messages, figure titles, the
worked-example script, requirements.txt and this README, in the version in CITATION.cff and pyproject.toml,
which is 0.1.1, and in the commit hashes of this README and of scripts/run_seeds.sh (see above); no identifier, key, file name or result changed.
