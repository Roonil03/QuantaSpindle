#!/bin/sh
# Run only when no other CPU/GPU timing job is active. See docs/cuda-startup.md.
set -eu
out=${OUT:-benchmarks/generated/cuda-startup}
trials=${TRIALS:-5}
repeats=${REPEATS:-5}
paths=${PATHS_LIST:-256,4096,16384,524288}
mkdir -p "$out" local/linux
out=$(cd "$out" && pwd)
baseline="$out/v0.1.1"
mkdir -p "$baseline"
git -c safe.directory=/work archive v0.1.1 | tar -x -C "$baseline"
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
  cat /proc/cpuinfo
  nvidia-smi
  git -c safe.directory=/work rev-parse HEAD HEAD^{tree} v0.1.1^{commit} v0.1.1^{tree}
  git -c safe.directory=/work diff --stat
  find . -maxdepth 3 -type f \( -name '*.go' -o -name '*.cu' -o -name 'benchmark-startup.sh' \) -not -path './local/*' -not -path './benchmarks/*' -print0 | sort -z | xargs -0 sha256sum
} > "$out/environment.txt"
find . -maxdepth 3 -type f \( -name '*.go' -o -name '*.cu' -o -name '*startup*.py' -o -name 'benchmark-startup.sh' \) -not -path './local/*' -not -path './benchmarks/*' -print0 | sort -z | xargs -0 sha256sum > "$out/sources.sha256"
nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static "$baseline/accelerators/cuda/runner.cu" -o "$out/runner-v0.1.1"
(cd "$baseline" && go build -o "$out/qsbench-v0.1.1" ./cmd/qsbench)
if [ "${PHASE:-measure}" = baseline-build ]; then exit 0; fi
nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static accelerators/cuda/runner.cu -o local/linux/quantaspindle-cuda
go build -o local/linux/qsstartup ./cmd/qsstartup
go build -o local/linux/qsbench ./cmd/qsbench
QS_CUDA_RUNNER=/work/local/linux/quantaspindle-cuda go test -count=1 ./cuda ./hybrid -v > "$out/device-tests.txt" 2>&1
sha256sum "$out/qsbench-v0.1.1" "$out/runner-v0.1.1" local/linux/qsstartup local/linux/qsbench local/linux/quantaspindle-cuda > "$out/binaries.sha256"
: > "$out/baseline.jsonl"
: > "$out/measure.jsonl"
: > "$out/alternative.jsonl"
# Pair CUDA-only release/current blocks before the CPU sweep. Alternate which
# build runs first; both blocks have only CUDA workloads and their own discarded
# warm-up. A new process/session remains cold for every one-off sample.
trial=1
while [ "$trial" -le "$trials" ]; do
  block=1
  for side in $(if [ $((trial % 2)) -eq 1 ]; then echo baseline current; else echo current baseline; fi); do
    if [ "$side" = baseline ]; then
      "$out/qsbench-v0.1.1" -backend cuda -cuda-runner "$out/runner-v0.1.1" -paths "$paths" -batches 16384 -trials 1 > "$out/baseline-trial.jsonl"
      python3 -c 'import json,sys; t,b=map(int,sys.argv[1:]); [print(json.dumps(dict(json.loads(l), trial=t, mode="cuda-v0.1.1", source="v0.1.1", phase="cold-pair", block_order=b))) for l in sys.stdin]' "$trial" "$block" < "$out/baseline-trial.jsonl" >> "$out/baseline.jsonl"
    else
      local/linux/qsstartup -cuda-runner /work/local/linux/quantaspindle-cuda -modes cuda-cold -paths "$paths" -trials 1 -label "cold-pair trial $trial" > "$out/current-cold-trial.jsonl"
      python3 -c 'import json,sys; t,b=map(int,sys.argv[1:]); [print(json.dumps(dict(json.loads(l), trial=t, phase="cold-pair", block_order=b))) for l in sys.stdin]' "$trial" "$block" < "$out/current-cold-trial.jsonl" >> "$out/measure.jsonl"
    fi
    block=$((block+1))
  done
  # Actually test the current adapter with unchanged tagged native source.
  local/linux/qsstartup -cuda-runner "$out/runner-v0.1.1" -modes cuda-cold -paths "$paths" -trials 1 -label current-adapter-tagged-native > "$out/alternative-trial.jsonl"
  python3 -c 'import json,sys; t=int(sys.argv[1]); [print(json.dumps(dict(json.loads(l), trial=t, phase="tagged-native-alternative", block_order=3))) for l in sys.stdin]' "$trial" < "$out/alternative-trial.jsonl" >> "$out/alternative.jsonl"
  local/linux/qsstartup -cuda-runner /work/local/linux/quantaspindle-cuda -modes cpu-default,cpu,cuda-reuse,auto-cold,auto-reuse -paths "$paths" -trials 1 -repeats "$repeats" -label "main trial $trial" > "$out/current-trial.jsonl"
  python3 -c 'import json,sys; t=int(sys.argv[1]); [print(json.dumps(dict(json.loads(l), trial=t, phase="main", block_order=4))) for l in sys.stdin]' "$trial" < "$out/current-trial.jsonl" >> "$out/measure.jsonl"
  trial=$((trial+1))
done
{
  date -u
  nvidia-smi
} > "$out/device-after.txt"
sha256sum "$out/baseline.jsonl" "$out/measure.jsonl" "$out/alternative.jsonl" > "$out/records.sha256"
python3 scripts/check-startup-records.py "$out/baseline.jsonl" "$out/measure.jsonl" "$out/alternative.jsonl" "$trials" "$paths" > "$out/record-checks.txt"
python3 scripts/summarize-startup.py "$out/baseline.jsonl" "$out/measure.jsonl" "$out/alternative.jsonl" > "$out/summary.jsonl"
sha256sum -c "$out/sources.sha256" > "$out/source-checks.txt"
