#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output=${1:-"$root/local/quantaspindle-cuda"}
mkdir -p "$(dirname -- "$output")"
"${NVCC:-nvcc}" -std=c++17 -O3 --fmad=false -arch="${CUDA_ARCH:-sm_86}" -cudart=static "$root/accelerators/cuda/runner.cu" -o "$output"
printf '%s\n' "$output"
