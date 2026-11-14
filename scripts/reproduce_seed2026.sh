#!/bin/bash
# Re-run partition seed 2026 of every data set (MNIST, CIFAR-10, Camelyon17 patch and slide split, census
# n = 250 and n = 1,000) with the code of this checkout into outputs/seed2026_check/; logs in
# outputs/seed2026_check/logs/. Then compare with the saved records: python scripts/compare_seed2026.py
# Usage (inside the Python environment of requirements.txt, GPU 0): bash scripts/reproduce_seed2026.sh
C="$(cd "$(dirname "$0")/.." && pwd)"
O=$C/outputs/seed2026_check
L=$O/logs
mkdir -p $O $L
cd $C
stamp() { echo "$(date '+%F %T') $* | $(uptime | sed 's/.*load/load/')" >> $L/stages.txt; }
gpu() {  # retry loop for the GPU start problem; $1 = log name, rest = command
  local log=$1; shift
  for i in 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=0 "$@" >> $L/$log.log 2>&1 && return 0; echo "retry $i" >> $L/$log.log; sleep 15; done
  return 1
}
echo "commit $(git rev-parse HEAD)" > $L/commit.txt
stamp start
gpu mnist_cache python scripts/14_build_screening_cache.py --dataset mnist --model lenet --checkpoint outputs/shared/models/mnist_lenet.pt --out-dir $O/main/mnist --seeds 2026 --verify-prefix mnist && stamp mnist_cache
gpu mnist_windows python scripts/15_run_screening.py --cache-dir $O/main/mnist --seeds 2026 && stamp mnist_windows
gpu cifar_cache python scripts/14_build_screening_cache.py --dataset cifar10 --model resnet18_cifar --checkpoint outputs/shared/models/cifar10_resnet18.pt --out-dir $O/main/cifar10 --seeds 2026 --verify-prefix cifar10 && stamp cifar_cache
gpu cifar_windows python scripts/15_run_screening.py --cache-dir $O/main/cifar10 --seeds 2026 && stamp cifar_windows
gpu patch_cache python scripts/32_camelyon_cache.py --root data/wilds --out-dir $O/camelyon_patch --seeds 2026 && stamp patch_cache
gpu patch_windows python scripts/33_camelyon_windows.py --cache-dir $O/camelyon_patch --seeds 2026 && stamp patch_windows
gpu slide_cache python experiments/camelyon_slide_split/cache.py --root data/wilds --out-dir $O/camelyon_slide --seeds 2026 && stamp slide_cache
gpu slide_windows python scripts/33_camelyon_windows.py --cache-dir $O/camelyon_slide --seeds 2026 && stamp slide_windows
python experiments/census/windows.py --seeds 2026 --out $O/census_n250 > $L/census_n250.log 2>&1 && stamp census_n250
python experiments/census/windows.py --seeds 2026 --window 1000 --out $O/census_n1000 --only-real > $L/census_n1000.log 2>&1 && stamp census_n1000
stamp done
