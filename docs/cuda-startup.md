# CUDA startup and automatic routing (v0.1.2)

On an RTX 3060 Laptop GPU, keeping a CUDA process ready reduced median latency to **2.67 ms for 4,096 scenarios and 10.44 ms for 16,384**. The fastest CPU medians were 6.54 and 26.75 ms. For a single cold call at either size, automatic routing chose CPU. Even across six calls, setting up and shutting down a new GPU service cost more per request than CPU. At 256 scenarios, both cold and prepared automatic routing chose CPU.

## Measured results

Measurements were collected on 2026-11-15. Each cell gives median [minimum to maximum] milliseconds across five trials. Each ready trial contributes the median of five requests made after the first real request. The last column divides the full service lifetime by six user requests, including preparation, device probe and shutdown. [All 380 timing samples](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836)

| Paths | Tagged v0.1.1 cold CUDA | Measured v0.1.2 candidate, forced cold CUDA | Fastest tested CPU | Cold auto (actual route) | Ready auto (actual route) | Six-call auto including setup, ms/call |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 139.960 [128.170 to 166.834] | 138.895 [128.792 to 147.577] | 0.783 [0.761 to 0.890] | 0.832 [0.753 to 1.334] CPU | 0.761 [0.754 to 0.831] CPU | 22.932 [22.677 to 24.361] |
| 4,096 | 141.904 [131.247 to 164.131] | 138.803 [127.798 to 142.447] | 6.536 [5.738 to 7.756] | 6.210 [5.512 to 9.347] CPU | 2.671 [2.637 to 2.986] CUDA | 25.614 [24.547 to 34.363] |
| 16,384 | 147.179 [137.404 to 172.334] | 139.216 [133.646 to 150.788] | 26.751 [24.422 to 52.373] | 29.236 [24.839 to 35.513] CPU | 10.444 [10.179 to 12.527] CUDA | 34.662 [34.092 to 35.466] |
| 524,288 | 407.643 [391.624 to 481.063] | 395.407 [390.898 to 460.724] | 810.008 [751.037 to 843.400] | 417.568 [396.958 to 439.244] CUDA | 282.183 [272.530 to 316.494] CUDA | 311.097 [293.104 to 334.698] |

The fresh CUDA timings overlap v0.1.1 at every measured size; no cold CUDA improvement over that release is established. The small-request gains come from skipping GPU startup on CPU routes and reusing an initialized process on GPU routes. At 524,288 scenarios, the cold automatic call took 417.57 ms on GPU versus 810.01 ms for the fastest CPU configuration.

The fastest CPU worker count and batch size, in increasing workload order, were 8/64, 16/64, 16/1,024 and 16/1,024. The default CPU configuration (16/64) measured 0.859, 6.536, 28.889 and 1,019.994 ms. These comparisons include the default and every combination of 1, 4, 8 or 16 workers with batch sizes 64, 128 or 1,024.

### Explicit CUDA and setup costs

| Scenarios | Prepare (ms) | First real request (ms) | Ready CUDA (ms) | Close (ms) | Six requests including setup (ms/request) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 108.369 | 2.137 | 1.963 | 28.575 | 24.908 |
| 4,096 | 107.151 | 2.802 | 2.741 | 28.808 | 25.590 |
| 16,384 | 103.800 | 11.771 | 11.338 | 28.813 | 33.464 |
| 524,288 | 132.291 | 332.925 | 283.331 | 29.444 | 326.838 |

Each timing column has its own median. The last column comes from timing each whole sequence. The first real request can grow device storage beyond the one scenario used for preparation.

Automatic selection considers cold CUDA at 262,144 scenarios or more, and a prepared GPU at 4,096 or more. Both require at least 252 steps and a supported single-asset model. The experiment used 252 steps. The 262,144 threshold also draws on the earlier measurements. Other hardware, intermediate sizes, longer horizons and concurrent service demand can change which route is faster. [Earlier crossover measurements](cuda-performance.md)

## Workload and controls

All cases used one asset with initial price 100, annual drift 0.05, volatility 0.2, seed 1, 252 steps and a time interval of 1/252 year. There were no regimes, supplied correlations or stored paths. Every run validated scenarios, computed the same summary fields and used a nil sink. CUDA used one worker and batches of up to 16,384. The automatic cases explicitly selected those CUDA execution settings. The library otherwise preserves the supplied settings.

- **Tagged cold CUDA.** Both the native executable and the Go engine, adapter and benchmark were built from [v0.1.1](https://github.com/Roonil03/QuantaSpindle/tree/v0.1.1) (`ed503e730f6740315d32add48c087d60414479d1`). The clock covers the complete `qs.Run`, including native process startup, device initialization, transfers, decoding, aggregation and shutdown.
- **Measured candidate cold CUDA.** One fresh process per run with the v0.1.2 implementation at [source `2db27040b08b6094766196bf0d93b99e2114093b`](https://github.com/Roonil03/QuantaSpindle/commit/2db27040b08b6094766196bf0d93b99e2114093b), tree `e169d445bdb725bba9600a1b8f1624d57dd98203`. Its clock also includes backend construction and timestamp/slice bookkeeping. The OS and driver caches can already be warm for either executable. Release-documentation changes do not change this experiment's recorded source identity.
- **Prepared CUDA.** Each trial starts its own service. `Prepare` performs a one-scenario, one-step device probe. The harness times six real requests and closes the service. No service survives a trial boundary.
- **Automatic selection.** Cold timings include selection, execution and cleanup. If CUDA is chosen, the first computed batch is saved for delivery. Ready automatic timings include a new selection and lease close for each request. The six-call experiment charges preparation even when all six calls select CPU.

Each case had a discarded warm-up. Each of the five trial blocks compared tagged and measured-candidate cold CUDA consecutively before the CPU sweep, alternating which build ran first. As another check, the measured Go adapter ran with the tagged native executable. Its cold medians were 141.172, 140.180, 168.506 and 430.921 ms. That combination always ran after the cold pair. Paths and configurations retained their order within each block. The laptop was shared, and temperature, power and background load were uncontrolled. The ranges show observed variability; they do not establish statistical significance or a general crossover.

Compilation, Go program startup, explicit pre-trial GC, memory-statistics reads and JSON serialization were outside the clocks. Go allocation counters include harness bookkeeping and exclude native and device memory. End-to-end comparisons use wall time.

## Hardware, validation and source

The Intel Core i9-11900H has eight cores and 16 logical CPUs. Go used `GOMAXPROCS=16`. The RTX 3060 Laptop GPU has 6,144 MiB of reported memory. The Linux amd64 container ran Ubuntu 22.04.4, Go 1.27.1 with cgo, CUDA 12.4.131 and GCC 11.4.0 under WSL2 kernel `6.18.58.1-microsoft-standard-WSL2`. There was no additional container CPU or memory quota. The native process reported CUDA runtime API 12040 and driver API 13050.

The local image was `quantaspindle-cuda-benchmark:local`, ID `sha256:bccbfdea8c8dcfaf6d47a46633f5e0964b10151778960d8ee635a0751660d05d`. The measurement ran from 08:04:11 to 08:11:12 UTC, including builds and pre-timing device tests. The checker verified 340 measured-implementation records, 20 tagged records and 20 records with the alternative native executable. All CUDA summaries matched the tagged implementation exactly for the tested seeds. All CPU-routed summaries matched explicit CPU. Real-device tests passed before timing and `go test -race -count=1 ./...` passed separately afterward.

Full source and binary hashes, JSONL records, environment details and test logs remain in the local output directory `benchmarks/generated/cuda-startup-release/`. The public CSV preserves the latency measurements. [Public samples](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836)

```text
native source     a8a32287684b5072ffe446152b076805adb3c9e0937eec173bc68b6779be6d37
native binary     9457573d59f7643e2d0fddaaacd2b1355a0fcb4d86fcbe12efc40e1043f3288d
qsstartup binary  abb0ca1c7530909c8d37c3f2a44a5480119b24cd5eb0ed5838c1c626ec77032a
baseline.jsonl    3ff1761dddb2839e455b544b8e912e24d5bd12cc2bd8b0d9bafad3efb35bd3c8
measure.jsonl     bb9c505f11240a7ba493e89f6614eb7ec09fa5a6a03e0b998780222004bed9ea
alternative.jsonl cf27311f439a3ba45b4bcef36f0d0408f137e459635fef6f52ccc51d71604910
```

### Earlier native output experiment

We also tried buffering native output in 32 KiB blocks on commit `8e9d250481191e0cf0990e8456c0ca0f8e82df14`. Cold CUDA took 137.662, 136.701, 158.347 and 473.587 ms across the four sizes, versus 133.319, 134.030, 143.464 and 408.872 ms for v0.1.1. The candidate cold run followed a long CPU sweep. Even its unchanged large-case kernel showed different clocks (212.102 ms versus 159.795 ms tagged), so the difference cannot be attributed to buffering. We restored the v0.1.1 output path and paired the cold measurements in the final experiment. [Candidate's 380 timing samples](https://github.com/Roonil03/QuantaSpindle/pull/9#issuecomment-6891385926)

## Reproduce the comparison

Use Go 1.27.1 and a full Git checkout so the script can extract the original v0.1.1 tag. The following PowerShell commands select the measured source in a new directory:

```powershell
git clone https://github.com/Roonil03/QuantaSpindle.git QuantaSpindle-startup
cd QuantaSpindle-startup
git switch --detach 2db27040b08b6094766196bf0d93b99e2114093b
```

On a machine with the original local image and NVIDIA container support, run:

```powershell
docker image inspect quantaspindle-cuda-benchmark:local
docker run --rm --user 0 --gpus all --entrypoint sh --mount "type=bind,source=$((Get-Location).Path),target=/work" -w /work -e OUT=benchmarks/generated/cuda-startup-release quantaspindle-cuda-benchmark:local scripts/benchmark-startup.sh
```

For another machine, save the public two-stage Dockerfile from the [earlier experiment](cuda-performance.md#reproduction) as `Dockerfile.cuda-public`. Build it with `docker build -f Dockerfile.cuda-public -t quantaspindle-cuda-public .`. Substitute `quantaspindle-cuda-public` for `quantaspindle-cuda-benchmark:local` in the commands above. This alternative image has not been benchmarked here. Record the resulting image ID and device details alongside the new measurements.

The script compiles both native executables with `nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static`. The optional environment variables `TRIALS`, `REPEATS` and `PATHS_LIST` narrow the experiment. By default the script runs the device tests and full matrix, then calls `scripts/check-startup-records.py` and `scripts/summarize-startup.py`. They check completeness, routing, replay and timing boundaries and summarize each experiment separately.

The current benchmark script also accepts `CUDA_ARCH`, defaulting to `sm_86`. Pass, for example, `-e CUDA_ARCH=sm_80` to `docker run` when that target is supported by your toolkit and GPU. Both tagged and current native builds use the selected target, which is recorded as `CUDA_ARCH=<target>` in `environment.txt`, including `PHASE=baseline-build` runs. The historical source selected above retains its original flags; use the current script for this option and record the new source and hardware as a separate experiment. Run the bounded command/metadata check with `python3 -m unittest scripts/test_benchmark_arch.py` in a Linux environment; it uses compiler stubs and produces no GPU timings.
