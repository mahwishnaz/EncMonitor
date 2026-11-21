#!/bin/bash
# Runs the experiments for all partition seeds, in this order:
#   A. main experiment, new seeds 2031-2045: MNIST, CIFAR-10, Camelyon17 patch split (Exact-MMD on), census n=250/1,000
#   B. Camelyon17 patient-level split, seeds 2026-2045 (cache + windows, Exact-MMD on)
#   C. ablation grid (scripts 17-18) for 20 seeds, then the same-reference exact analysis (exact_site_mmd/run.py)
# Every step is skipped if its output exists, so the script can be restarted at any time:
#   tmux new-session -d -s runs bash scripts/run_seeds.sh   (inside the Python environment of requirements.txt)
# One log per run in outputs/logs/runs/; start/end/uptime per run in outputs/logs/runs/times.tsv.
# The script stops unless the checkout it runs in (ENCMONITOR_CODE, default: this repository) is at CODE_REF
# (default: commit 6c28439, the code of the reported runs). To repeat the reported runs exactly:
#   git worktree add ../encmonitor-6c28439 6c28439 && ENCMONITOR_CODE=../encmonitor-6c28439 bash scripts/run_seeds.sh
# To run with the current checkout instead: CODE_REF=HEAD bash scripts/run_seeds.sh
set -u
C=${ENCMONITOR_CODE:-$(cd "$(dirname "$0")/.." && pwd)}
CODE_REF=${CODE_REF:-6c2843930fd44e9135e0aae437857e8e3e8c8542}
O=$C/outputs
L=$O/logs/runs
mkdir -p $L
cd $C
TAG_COMMIT=$(git rev-parse $CODE_REF^{commit})
HEAD_COMMIT=$(git rev-parse HEAD)
if [ "$TAG_COMMIT" != "$HEAD_COMMIT" ]; then echo "clone HEAD $HEAD_COMMIT is not $CODE_REF ($TAG_COMMIT); stop" | tee -a $L/driver.log; exit 1; fi
NEW_SEEDS="2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045"
ALL_SEEDS="2026 2027 2028 2029 2030 $NEW_SEEDS"
ALL_CSV=$(echo $ALL_SEEDS | tr ' ' ',')

ld() { uptime | sed 's/.*load average: //'; }
run() {   # run <name> <gpu:0|1> <command...>; logs to $L/<name>.log, timing to times.tsv; GPU jobs retried
  local name=$1 gpu=$2; shift 2
  local t0=$(date +%s) s0="$(date '+%F %T')" l0="$(ld)" ok=1
  echo "[$s0] start $name (commit $HEAD_COMMIT)" >> $L/driver.log
  if [ "$gpu" = 1 ]; then
    for i in 1 2 3 4 5 6 7 8; do
      if CUDA_VISIBLE_DEVICES=0 "$@" >> $L/$name.log 2>&1; then ok=0; break; fi
      echo "retry $i after failure" >> $L/$name.log; sleep 15
    done
  else
    "$@" >> $L/$name.log 2>&1 && ok=0
  fi
  local t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$s0" "$(date '+%F %T')" "$((t1 - t0))" "$ok" "$l0" "$(ld)" >> $L/times.tsv
  echo "[$(date '+%F %T')] end $name status=$ok seconds=$((t1 - t0))" >> $L/driver.log
  return $ok
}
[ -f $L/times.tsv ] || printf "run\tstart\tend\tseconds\tstatus(0=ok)\tload_start\tload_end\n" > $L/times.tsv
info() {  # RUN_INFO.json next to the results of one run directory
  mkdir -p "$1"; printf '{"commit": "%s", "code_ref": "%s", "written": "%s", "command": "%s"}\n' "$HEAD_COMMIT" "$CODE_REF" "$(date '+%F %T')" "$2" > "$1/RUN_INFO.json"
}

# ---------------- A. main experiment, new seeds
for s in $NEW_SEEDS; do
  d=$O/main/mnist/seed_$s
  [ -f $d/cache_done.json ] || run mnist_cache_$s 1 python scripts/14_build_screening_cache.py --dataset mnist --model lenet --checkpoint outputs/shared/models/mnist_lenet.pt --out-dir outputs/main/mnist --seeds $s
  [ -f $d/records.jsonl ] || { run mnist_windows_$s 1 python scripts/15_run_screening.py --cache-dir outputs/main/mnist --seeds $s && info $d "scripts 14+15 mnist seed $s"; }
  d=$O/main/cifar10/seed_$s
  [ -f $d/cache_done.json ] || run cifar10_cache_$s 1 python scripts/14_build_screening_cache.py --dataset cifar10 --model resnet18_cifar --checkpoint outputs/shared/models/cifar10_resnet18.pt --out-dir outputs/main/cifar10 --seeds $s
  [ -f $d/records.jsonl ] || { run cifar10_windows_$s 1 python scripts/15_run_screening.py --cache-dir outputs/main/cifar10 --seeds $s && info $d "scripts 14+15 cifar10 seed $s"; }
  d=$O/camelyon/main/camelyon17/seed_$s
  [ -f $d/cache_done.json ] || run patch_cache_$s 1 python scripts/32_camelyon_cache.py --root data/wilds --seeds $s
  [ -f $d/records.jsonl ] || { run patch_windows_$s 1 python scripts/33_camelyon_windows.py --seeds $s && info $d "scripts 32+33 camelyon17 patch seed $s"; }
  d=$O/census/records_n250/seed_$s
  [ -f $d/records.jsonl ] || { run census_n250_$s 0 python experiments/census/windows.py --seeds $s --out $O/census/records_n250 && info $d "census windows n=250 seed $s"; }
  d=$O/census/records_n1000/seed_$s
  [ -f $d/records.jsonl ] || { run census_n1000_$s 0 python experiments/census/windows.py --seeds $s --window 1000 --out $O/census/records_n1000 --only-real && info $d "census windows n=1000 only-real seed $s"; }
done
echo "[$(date '+%F %T')] A done" >> $L/driver.log

# ---------------- B. patient-level split, all 20 seeds
for s in $ALL_SEEDS; do
  d=$O/camelyon/patient_split/camelyon17/seed_$s
  [ -f $d/cache_done.json ] || run patient_cache_$s 1 python experiments/camelyon_patient_split/cache.py --root data/wilds --seeds $s
  [ -f $d/records.jsonl ] || { run patient_windows_$s 1 python scripts/33_camelyon_windows.py --cache-dir outputs/camelyon/patient_split/camelyon17 --seeds $s && info $d "patient split cache + script 33 seed $s"; }
done
echo "[$(date '+%F %T')] B done" >> $L/driver.log

# ---------------- C. ablation grid (20 seeds) and same-reference exact analysis (20 seeds)
for ds in mnist cifar10; do
  if [ $ds = mnist ]; then M=lenet; CK=outputs/shared/models/mnist_lenet.pt; else M=resnet18_cifar; CK=outputs/shared/models/cifar10_resnet18.pt; fi
  for s in $ALL_SEEDS; do   # caches per seed (S = 10, early layer, logits); skipped inside the script if present
    [ -f $O/ablation/cache/$ds/S5_logits/seed_$s/cache_done.json ] || run ablation_cache_${ds}_$s 1 python scripts/17_build_ablation_cache.py --dataset $ds --model $M --checkpoint $CK --seeds $s
  done
  [ -f $O/ablation/runs/$ds/run_meta.json ] || { run ablation_runs_$ds 1 python scripts/18_run_ablation.py --dataset $ds --seeds $ALL_CSV && info $O/ablation/runs/$ds "script 18 $ds seeds $ALL_CSV"; }
done
[ -f $O/ablation/analysis/q095/ablation_summary.csv ] || run ablation_analysis 0 python scripts/19_analyze_ablation.py --out-dir outputs/ablation/analysis/q095 --quantile 0.95
run exact_site_mmd 1 python experiments/exact_site_mmd/run.py
echo "[$(date '+%F %T')] C done; driver finished" >> $L/driver.log
