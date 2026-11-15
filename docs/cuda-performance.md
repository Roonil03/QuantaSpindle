# CUDA performance experiment

Measurements dated 2026-11-15 compare complete multi-batch `Run` calls with the released one-process-per-batch CUDA runner, the persistent CUDA runner, and the portable CPU backend. The optimization keeps a native process and its CUDA resources alive for one `Run`; each new measured `Run` starts and closes its own process. Direct `Backend.Batch` continues to use one process per call.

The persistent runner's sample median was **1.34× faster than the fastest tested CPU configuration at 262,144 scenarios and 1.94× faster at 524,288 scenarios**, including startup and teardown. It was 7.72× and 11.86× faster than the released CUDA runner. At the three smaller tested workloads, parallel CPU remained faster. The 262,144-scenario CPU and CUDA ranges overlap; these sample-median ratios do not establish statistical significance or a universal crossover.

## Measured results

All times below are milliseconds per complete Run, reported as median [minimum to maximum] of five trials. Throughput is scenarios/s; it also equals asset paths/s for this single-asset model. "Fastest CPU" means the lowest median among the eight tested worker/batch combinations, with no claim of exhaustive CPU tuning.

- **262,144 scenarios, 16 CUDA batches:** fastest CPU (workers 16, batch 1,024) 509.158 [493.447 to 545.500], 514,858 scenarios/s; released CUDA 2,942.160 [2,646.472 to 3,498.135], 89,099 scenarios/s; persistent CUDA 381.296 [339.633 to 693.529], 687,507 scenarios/s.
- **524,288 scenarios, 32 CUDA batches:** fastest CPU (workers 16, batch 1,024) 980.901 [892.519 to 1,243.223], 534,496 scenarios/s; released CUDA 5,997.168 [5,217.480 to 8,088.080], 87,423 scenarios/s; persistent CUDA 505.462 [493.779 to 606.071], 1,037,245 scenarios/s.
- **256 scenarios:** fastest CPU (configured workers 8, batch 128; only two batches) 1.459 [1.362 to 1.736]; released CUDA 145.106 [140.556 to 178.390]; persistent CUDA 159.634 [143.704 to 196.605].
- **4,096 scenarios:** fastest CPU (workers 16, batch 128) 7.840 [7.429 to 11.681]; released CUDA 157.988 [144.792 to 215.264]; persistent CUDA 155.663 [142.791 to 199.807].
- **16,384 scenarios:** fastest CPU (workers 16, batch 1,024) 32.252 [28.227 to 39.524]; released CUDA 208.473 [162.602 to 649.741]; persistent CUDA 220.070 [187.416 to 470.422].

The large-case CPU sweep confirms the parallel comparison. In worker order 1/4/8/16, the 262,144-scenario CPU medians were 3,093.994/1,007.003/696.072/606.667 with batch 128 and 3,227.473/926.495/628.001/509.158 with batch 1,024. At 524,288 scenarios they were 6,388.472/2,050.296/1,370.650/1,146.876 and 6,096.950/1,941.992/1,254.756/980.901 respectively.

Persistent CUDA summed kernel medians were 116.593 [114.607 to 117.349] and 200.407 [160.565 to 222.053] at the two large sizes. Its summed native host medians were 256.356 and 317.148. Those are narrower internal clocks, not the end-to-end speedup denominator. Median Go allocated bytes/allocations were 61,023,208/1,049,069 and 122,000,168/2,098,055 for persistent CUDA, versus 52,490,856/1,049,433 and 104,981,560/2,098,860 for the fastest tested CPU cases.

For review without relying on ignored logs, the five full timing samples in trial order are:

```text
262144 CPU 16/1024: 545.500, 520.532, 493.447, 496.477, 509.158 ms
262144 CUDA run:    381.296, 383.643, 339.633, 693.529, 348.886 ms
262144 CUDA v0.1.0: 2942.160, 2869.453, 2646.472, 3498.135, 3206.155 ms
524288 CPU 16/1024: 1137.945, 980.901, 931.575, 1243.223, 892.519 ms
524288 CUDA run:    505.462, 498.794, 525.919, 606.071, 493.779 ms
524288 CUDA v0.1.0: 5217.480, 5997.168, 5549.704, 8088.080, 6260.752 ms
```

The complete 250-record [raw JSONL release asset](https://github.com/Roonil03/QuantaSpindle/releases/download/v0.1.1/measure.jsonl) has SHA-256 `7ae00f3370f79c0d908d76e5f2d05202e7f9466dad86fa39de80943a4e42cc27`. It retains all eight CPU configurations, full config, trial order, timing, allocations, summaries and device metadata. The final script exited successfully. Real-device `go test -count=1 ./cuda -v` passed, including deterministic stream replay against direct one-shot Batch and overlapping Runs with different batch partitions. CPU/GPU moment and zero-volatility checks also passed. The summarizer's configuration/file separation and duplicate-trial checks passed with `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest scripts/test_summarize_cuda.py` after timed work.

## Workload and timing

Every case uses one asset, price 100, drift 0.05, volatility 0.2, seed 1, 252 steps and DT 1/252 year, no regimes, correlation or stored paths, and a nil sink. CUDA uses one worker and batches of at most 16,384 scenarios. CPU worker counts 1, 4, 8 and 16 are measured with batch sizes 128 and 1,024 so that large cases expose enough batches to use every worker. Both paths validate scenarios and compute the same summary fields. The random algorithms differ; distributional agreement and deterministic replay within each backend are tested separately from throughput.

One warm-up per case precedes five measured trials. The case order reverses on alternate trials. Explicit pre-trial GC and memory-statistics reads, compilation and output JSON serialization are outside the clock. The clock encloses `qs.Run`, including native process startup, CUDA initialization, allocations, transfers, synchronization, result decoding, Go validation/aggregation and process shutdown. No initialized session crosses a trial boundary. Go program startup is excluded. CUDA event time is the sum of kernels across a Run and is reported separately from elapsed time. Native host timings and event timings must not be substituted for full Run time or subtracted as if separate medians were a measured decomposition.

Results compare the fastest measured CPU worker configuration for each workload, rather than only a serial CPU. Go allocation totals exclude native process and device allocations; no peak memory or energy measurement was collected. These are shared laptop measurements with uncontrolled background work and thermal/power limits, not a dedicated-device benchmark or a general crossover guarantee.

## Source and hardware identity

The final experiment compiled implementation commit `b5f1a31c356ae301ef1f5702284e6a4c9e4bd36c`, Git tree `3ad48595be7693eedcb63fc50431019e2eb2ad78`. Documentation edits were present outside that implementation commit. The released native baseline is `v0.1.0`, peeled commit `3ffad625663d2ec24a54de6aabc41252491c9210`, tree `585d3cb04b13e6a99c705874c36d5b5d146a6f8f`. The current native source SHA-256 is `081803744a34d364e88e5378461df92f5f8c3d98aa721e1e4c454cf92b25d07f`; the exact source and binary hashes are retained in the ignored experiment directory. Earlier development runs are separate and are not pooled into these results.

The CPU is an Intel Core i9-11900H, eight physical cores and 16 logical CPUs. Docker Engine 29.9.3 exposes 16 CPUs and 8,182,960,128 bytes of WSL2 memory with 2 GiB swap; the container cgroup has no additional CPU/memory quota. GOMAXPROCS is 16. The kernel is `6.18.58.1-microsoft-standard-WSL2`, Linux amd64, and Go `CGO_ENABLED=1`. The GPU is an NVIDIA GeForce RTX 3060 Laptop GPU with 6,144 MiB reported device memory. The Linux NVIDIA-SMI shim is 620.73.02 and Windows kernel-mode driver is 621.24. Actual runner metadata reports CUDA runtime API 12040 and driver API 13050. The pre-run GPU snapshot showed 292 MiB in use and 3% utilization, with no processes listed by NVIDIA-SMI; this does not establish exclusive access to the host GPU.

## Reproduction

The measured local image is `quantaspindle-cuda-benchmark:local`, image ID `sha256:bccbfdea8c8dcfaf6d47a46633f5e0964b10151778960d8ee635a0751660d05d`. Its base is the previously installed local CUDA image identified in [the validation record](validation.md). It supplies CUDA 12.4.131, GCC 11.4.0 and Ubuntu 22.04.4; Go is 1.27.1. The image is a local measurement environment, not a public download.

From PowerShell in the repository:

```powershell
docker image inspect quantaspindle-cuda-benchmark:local
docker version
docker run --rm --user 0 --gpus all --entrypoint sh --mount "type=bind,source=$((Get-Location).Path),target=/work" -w /work -e OUT=benchmarks/generated/cuda-resident-final quantaspindle-cuda-benchmark:local scripts/benchmark-cuda.sh
docker run --rm --user 0 --entrypoint python3 --mount "type=bind,source=$((Get-Location).Path),target=/work" -w /work quantaspindle-cuda-benchmark:local scripts/summarize-cuda.py benchmarks/generated/cuda-resident-final/measure.jsonl
```

The script builds the actual `v0.1.0` native source through `git show`, compiles the current native runner with identical `nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static` flags, runs real-device tests, then runs matched cases. `cuda-batch` exposes only the public `Name` and `Batch` interface and uses the release executable, so the current engine's optional session hook cannot change that baseline. The current runner uses the ordinary `cuda` backend with the hook enabled. `PATHS_LIST`, `TRIALS`, `OUT`, and `PHASE=baseline` optionally narrow or isolate a run. The default experiment uses 256, 4,096, 16,384, 262,144 and 524,288 scenarios, five trials, CPU workers 1/4/8/16 with batches 128/1,024, and GPU workers 1. Raw JSONL, source and binary SHA-256 hashes, environment and device-test logs remain in ignored `benchmarks/generated/cuda-resident-final/`; they are not committed generated results.

The summarizer groups records by input filename, full configuration and mode, and rejects duplicate trial numbers within a group. Separate input files remain separate experiments rather than being silently pooled. Run its checks with `python3 scripts/test_summarize_cuda.py`.

For a publicly obtainable alternative environment, this local Dockerfile pins the CUDA development and Go images:

```dockerfile
FROM golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66 AS go
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5
RUN apt-get update && apt-get install -y --no-install-recommends git python3 && rm -rf /var/lib/apt/lists/*
COPY --from=go /usr/local/go /usr/local/go
ENV PATH=/usr/local/go/bin:/usr/local/cuda/bin:${PATH}
WORKDIR /work
```

Build that Dockerfile under a separate local image tag and substitute it into the commands. This is a reproducible alternative recipe; it was not used for the measurements here and its performance has not been measured. The CUDA image digest was verified with `docker buildx imagetools inspect`. Driver and NVIDIA container-runtime setup are external prerequisites. See [CUDA setup](cuda.md) and [benchmark methodology](benchmarks.md).
