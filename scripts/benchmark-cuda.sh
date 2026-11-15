#!/bin/sh
# Run in the CUDA/Go development image documented in docs/cuda-performance.md.
# PHASE=baseline collects CPU and release one-shot results while current source
# is being developed. PHASE=measure (default) rebuilds and measures all modes.
set -eu
out=${OUT:-benchmarks/generated/cuda-resident}
mkdir -p "$out" local/linux
phase=${PHASE:-measure}
trials=${TRIALS:-5}
paths=${PATHS_LIST:-256,4096,16384,262144,524288}
{
  date -u
  go version
  go env GOOS GOARCH CGO_ENABLED
  nvcc --version
  c++ --version
  uname -a
  nproc
  cat /sys/fs/cgroup/memory.max
  cat /sys/fs/cgroup/cpu.max
  cat /proc/meminfo
  cat /proc/cpuinfo
  nvidia-smi
  git -c safe.directory=/work rev-parse HEAD HEAD^{tree} v0.1.0^{commit} v0.1.0^{tree}
  git -c safe.directory=/work diff --stat
  sha256sum engine.go cuda/*.go accelerators/cuda/runner.cu cmd/qsbench/main.go scripts/benchmark-cuda.sh
} > "$out/environment-$phase.txt"
# Build the actual release kernel, not a reimplementation of the baseline.
git -c safe.directory=/work show v0.1.0:accelerators/cuda/runner.cu > "$out/runner-v0.1.0.cu"
nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static "$out/runner-v0.1.0.cu" -o local/linux/quantaspindle-cuda-v0.1.0
go build -o local/linux/qsbench ./cmd/qsbench
backends=cpu,cuda-batch
if [ "$phase" != baseline ]; then
  nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static accelerators/cuda/runner.cu -o local/linux/quantaspindle-cuda
  QS_CUDA_RUNNER=/work/local/linux/quantaspindle-cuda go test -count=1 ./cuda -v > "$out/device-tests.txt" 2>&1
  backends=cpu,cuda-batch,cuda
fi
local/linux/qsbench -backend "$backends" -cuda-runner /work/local/linux/quantaspindle-cuda \
  -cuda-batch-runner /work/local/linux/quantaspindle-cuda-v0.1.0 \
  -paths "$paths" -batches 16384 -cpu-batches 128,1024 -workers 1,4,8,16 -trials "$trials" > "$out/$phase.jsonl"
sha256sum local/linux/qsbench local/linux/quantaspindle-cuda* > "$out/binaries-$phase.sha256"
{
  date -u
  nvidia-smi
} > "$out/device-after-$phase.txt"
