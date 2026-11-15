# Reproducing measurements

## Select the source and workload

- For automatic routing and repeated small jobs in v0.1.2, follow the [CUDA startup experiment](cuda-startup.md). Its cold CUDA reference builds both the Go adapter and the native runner from the v0.1.1 tag. For each automatic result, record which backend produced the scenarios.
- A service can pay for CUDA initialization before it receives a request. Include preparation and shutdown when comparing the whole service lifetime. Report later requests separately when measuring latency on an already prepared GPU.
- Record the exact source commit/tag before building either runner or Go executable. Check out v0.1.0 in a separate local directory to reproduce historical one-shot measurements; never move its published tag.
- For v0.1.1 lifecycle/optimization comparisons, use the matched experiment and commands in [CUDA performance](cuda-performance.md). Keep those results separate from the historical tables in [validation](validation.md).
- Run both a single-batch case and a multibatch case. Per-run process reuse can amortize setup only when a run contains multiple batches; a warm-up run does not keep its process alive for the next run.
- Include complete `Run` startup and close in full timings. Kernel event time, runner request time, and Go allocation counts cover different boundaries. Go allocations exclude native and device allocations.

## Historical v0.1.0 environment and commands

Run the historical commands below from the recorded original source (the matched CPU/CUDA experiment used delivery commit `3f41ec8`). Running them from v0.1.1 selects its session-enabled `cuda` mode and produces a new experiment, not the original one-process-per-batch baseline. The current `scripts/benchmark-cuda.sh` instead builds the tagged v0.1.0 runner and selects `cuda-batch` explicitly alongside the current `cuda` mode.

Use identical models, seeds, step counts, scenario counts, batch sizes, output modes, and trial counts for comparisons. In this project a scenario contains one path per asset. Scenarios/s and asset paths/s are equal only for a single asset. Throughput, per-batch latency, and numerical validity are separate quantities.

The pinned CPU environment is `golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66`. It was pulled on 2026-11-15. The module has no external dependencies. Use this local-only Dockerfile at the exact ignored path `Dockerfile.benchmark.local`:

```dockerfile
FROM golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66
WORKDIR /work
CMD ["sh", "scripts/benchmark-cpu.sh"]
```

From PowerShell, build and run it with the repository mounted:

```powershell
docker build -f Dockerfile.benchmark.local -t quantaspindle-benchmark:local .
docker run --rm --mount "type=bind,source=$((Get-Location).Path),target=/work" quantaspindle-benchmark:local
```

The Dockerfile is excluded from Git; create it locally using the content above. Container output goes to ignored directories. Preserve raw results with your experiment rather than committing generated logs. For the validation run, see docs/validation.md. Record `git rev-parse HEAD`, `docker image inspect`, the engine version, CPU model and assigned cores, assigned memory, OS/kernel, Go version, model settings, seed, workers, and batch size with any published number. Container resource assignments can differ from physical host resources.

Run one warm-up followed by five measured trials. Report the median and minimum/maximum of elapsed time, paths/s, scenarios/s, B/op, and allocs/op. The benchmarks include workers and batch-size cases. Profiling is a separate run; profiling overhead must not enter the published throughput sample. The saved CPU profile is the starting point for tuning, not evidence of a speedup.

For profile-guided optimization, compare the representative profile with standard compilation:

```sh
go test -pgo=off -run '^$' -bench 'BenchmarkRun/workers=4/batch=128$' -benchtime 500ms -benchmem -count 5 .
go test -pgo=profiles/cpu.prof -run '^$' -bench 'BenchmarkRun/workers=4/batch=128$' -benchtime 500ms -benchmem -count 5 .
```

The recorded comparison is in docs/validation.md. Generated profiles are ignored. No custom compiler, runtime, unsafe code, assembly, or unmeasured optimizer changes are included.

CUDA measurements require a working NVIDIA GPU, driver, toolkit, and optional supported container runtime. See docs/cuda.md. Report kernel event time separately from full Go invocation time, including process startup, device setup, transfer, allocation, and synchronization. A native Windows run is not a GPU container run. Mark unavailable container checks explicitly. CPU/CUDA tests compare distributions because their random generators differ; no bitwise cross-backend equality is promised.

The historical v0.1.0 GPU measurements used an existing local CUDA 12.4.131 toolchain image plus the pinned Go 1.27.1 toolchain, not the incomplete CUDA 13.2 image download. Reproducing the exact local image requires access to the local image identities listed in docs/validation.md. They are not claimed as public images. The ignored `local/Dockerfile.cuda` used the pinned Go image as a first stage, the recorded local CUDA base as a second stage, copied `/usr/local/go` between them, set PATH to include `/usr/local/go/bin` and `/usr/local/cuda/bin`, and set WORKDIR to `/work`. Build it locally, then run:

```powershell
docker build -f local/Dockerfile.cuda --build-arg CUDA_BASE=fin-dl-benchmark@sha256:98e28ee2f2323e68a2fdc5a571e4c41e93e41305da77a1f8baad15cfb0802e64 -t quantaspindle-cuda-benchmark:local .
docker run --rm --user 0 --gpus all --entrypoint sh --mount "type=bind,source=$((Get-Location).Path),target=/work" -w /work quantaspindle-cuda-benchmark:local -c 'set -eu; mkdir -p local/linux benchmarks/generated; nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static accelerators/cuda/runner.cu -o local/linux/quantaspindle-cuda; QS_CUDA_RUNNER=/work/local/linux/quantaspindle-cuda go test -count=1 ./cuda -v'
docker run --rm --user 0 --gpus all --entrypoint sh --mount "type=bind,source=$((Get-Location).Path),target=/work" -w /work quantaspindle-cuda-benchmark:local -c 'set -eu; go run ./cmd/qsbench -backend cpu > benchmarks/generated/batches-cpu.jsonl; go run ./cmd/qsbench -backend cuda -cuda-runner /work/local/linux/quantaspindle-cuda > benchmarks/generated/batches-cuda.jsonl'
```

At the historical measurement revision, `qsbench` ran one warm-up and five measured trials for batch sizes 256, 4096, and 16384. Each JSON record includes the exact config, backend, Go version, elapsed time, Go allocations, and optional CUDA timing/provenance. `runtime.GC()` runs before timing each trial. It does not measure peak memory or include JSON encoding in the elapsed time. Record `docker image inspect`, `nvcc --version`, host compiler, OS, assigned resources, and `nvidia-smi` as well. Rebuild under another base only as a new experiment and record that identity; do not label it the recorded image.

FPGA performance model, RTL simulation, vendor emulation, and physical board results are all unavailable. Keep them separate from CPU and CUDA measurements when adapters are added.
