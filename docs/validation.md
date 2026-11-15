# Validation record

## Version scope

- The [startup and dispatch experiment](cuda-startup.md) compares the implementation shipped in v0.1.2 with the complete v0.1.1 CUDA adapter and native runner. It covers one-off calls, reusable services and the automatic backend choice.
- **v0.1.0 historical record:** the environments, checks, CPU baselines, original one-process-per-batch CUDA measurements, PGO experiment, and publication verification below describe the original release work. They are not measurements of the optimized implementation.
- **v0.1.1 follow-up:** per-run CUDA process/device reuse and its matched measurements are recorded in [CUDA performance](cuda-performance.md). Follow-up CI and rebuilt-device checks are recorded below, separately from the original release checks.
- **Publication:** v0.1.0 remains at its original tag. A later documentation edit or GitHub release-note follow-up does not change its source, checksum, or behavior.

## Hybrid routing checks on 2026-11-15

During v0.1.2 release preparation, the Windows Go 1.27.1 test suite and vet passed. The rebuilt native runner passed the CUDA and hybrid package tests on the RTX 3060 Laptop GPU. The full Linux suite also passed with race detection enabled and the GPU available. An independent check of the three original benchmark files verified their SHA-256 hashes, all 380 records and the reported latency medians and ranges.

On Windows amd64, the rebuilt CUDA runner and `go test -count=1 ./...` passed on commit `8e9d250` with `QS_CUDA_RUNNER` pointing to the RTX 3060 Laptop runner. The device suite checks replay across independently submitted requests and batch partitions. The portable subprocess tests also exercise canceling a queued request, recovery following a failed exchange, a stale or truncated greeting, and closing a service with an outstanding run.

GitHub Actions passed formatting, vet, CPU tests, race detection and benchmark smoke checks on the same revision. The implementation and review fixes merged in [PR 9](https://github.com/Roonil03/QuantaSpindle/pull/9). [CI run](https://github.com/Roonil03/QuantaSpindle/actions/runs/43691757152)

The comparison harness and native output path used in the final experiment were reviewed at `2db2704`. The rebuilt Windows runner passed `go test -count=1 ./cuda ./hybrid`, and the four CLI checks below passed again on that native source. A Linux smoke comparison validated 19 records and the alternating cold-run block metadata before the full matrix. [PR 10](https://github.com/Roonil03/QuantaSpindle/pull/10) · [CI run](https://github.com/Roonil03/QuantaSpindle/actions/runs/43692699962)

The complete five-trial Linux run produced 380 records. Its checks verified all expected modes, trial IDs, CPU configurations, cold comparison order, per-request execution choices, source hashes and fixed-seed summaries. After the benchmark, `go test -race -count=1 ./...` passed with the rebuilt CUDA runner on the real GPU. [Timing records](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836)

The ordinary CLI completed four checks with its default automatic policy. A 256-scenario run selected CPU. At 262,144 scenarios it selected CUDA with the working runner, then CPU in two separate checks with a missing runner and with the GPU hidden from the working runner using `CUDA_VISIBLE_DEVICES=-1`. In each case, the metadata and final summary identified the same backend and the summary contained every requested scenario.

## Historical release checks

Date: 2026-11-15. This records completed checks and measured results, not financial calibration. The source implementation reviewed for the corrected CPU baseline was CPU commit `6d03dc7` and combined CUDA branch commit `6f451d7`, subsequently merged into main at `ccdc873`. The independent pre-implementation CPU baseline is preserved locally as `benchmarks/generated/cpu-initial-98b5e91.txt`. It was gathered before any CUDA optimization; no CUDA tuning or speedup claim followed it.

## Environment

- CPU: Intel Core i9-11900H at reported 2.50 GHz, 8 cores and 16 logical CPUs.
- CPU container: Docker Engine 29.9.3, Linux amd64, WSL2 kernel `6.18.58.1-microsoft-standard-WSL2`, 16 visible logical CPUs, 8,182,960,128 bytes assigned memory, 2 GiB swap.
- Image: `golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66`. Go 1.27.1, `CGO_ENABLED=1`, standard compiler flags, no external Go modules. GOMAXPROCS was 16.
- Native host: Windows amd64, Go 1.27.1, `CGO_ENABLED=0`. No GCC toolchain was available for native Go race tests. The Linux container ran those checks instead.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB reported device memory, Windows driver 621.24. Native nvcc 13.2.51, Visual Studio 2022 MSVC 14.42.34433, runner architecture `sm_86`, FP64, `-O3 --fmad=false`, no fast-math. Actual runner metadata reports CUDA runtime API 13020 and driver API 13050.
- GPU container access probe: `docker run --rm --gpus all <pinned-Go-image> nvidia-smi` succeeded. The Linux driver shim reported 620.73.02, Windows kernel-mode driver 621.24. Completed container execution results are recorded separately below.

Memory assigned to WSL2 is not a measurement of physical host RAM. Background host work and laptop thermal/power limits affect these measurements. Do not compare them with another system without controlling those conditions.

## Completed checks

Native Windows `go test ./...` and `go vet ./...` passed during implementation. A short CLI run exposed zero elapsed time at the platform clock's resolution; the CLI now omits throughput when elapsed time is zero rather than emitting nonfinite JSON. The CLI test passes after that correction.

The corrected source passed `go test ./...` and `go test -race ./...` in the pinned Go Linux container. CPU tests cover analytical zero-volatility paths, 30,000-path GBM mean/variance checks using predeclared six-standard-error tolerances, 16,000-path empirical innovation correlations, singular PSD inputs, regime timing, replay across worker and batch choices, parameter and storage limits, cancellation, sink ownership and partial delivery, malformed backend paths/states, and hand-computed risk/impact results. Race detection covers Go code, not native GPU memory.

The rebuilt native Windows runner passed `QS_CUDA_RUNNER=<absolute-runner-path> go test ./cuda -v`. This includes actual device execution, deterministic replay and scenario-ID partitioning, analytical and CPU/GPU moment comparisons at stated tolerances, deterministic zero-volatility drawdown, single-step behavior, overflow, malformed native requests including total-horizon overflow, process deadlines, and bounded subprocess stdout/stderr. CUDA tests have no CPU fallback. Portable CUDA adapter tests and vet also passed for Go 386 during implementation. Wider OS and GPU architecture support remains untested.

The reviewed design and source changes were split into Foundations, Scenarios, Kernels, and Delivery pull requests. Astra Medium posted findings and re-reviewed Terra's corrections before Terra merged them.

## CPU throughput baseline

Workload: one asset, initial price 100, annual drift 0.05, volatility 0.2, 252 steps, DT 1/252 year, seed 1, 1,024 scenarios per operation, no stored paths, no regimes or supplied correlation. The sink discards results while the library computes summary means. One scenario equals one asset path here. Setup, worker scheduling, simulation, and aggregation are included. JSON serialization, process startup, and compilation are excluded from these Go benchmarks.

One warm-up (`-benchtime 1x`) preceded five measured trials with `-benchtime 500ms -count 5 -benchmem`. Each trial is Go's elapsed-time average over its calibrated operation count. The table reports median and minimum/maximum of those five averages. Allocations and bytes are median values per complete 1,024-scenario operation, not peak memory.

| Workers | Batch | Median ms/op | Min to max ms/op | Median scenarios/s and paths/s | B/op | Allocs/op |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 12.930 | 12.756 to 15.987 | 79,195 | 251,650 | 4,614 |
| 1 | 128 | 12.728 | 11.893 to 13.345 | 80,453 | 209,816 | 4,166 |
| 2 | 16 | 10.755 | 9.413 to 11.994 | 95,214 | 248,333 | 4,454 |
| 2 | 128 | 8.804 | 8.227 to 10.919 | 116,317 | 209,404 | 4,146 |
| 4 | 16 | 8.630 | 8.292 to 9.625 | 118,654 | 246,380 | 4,374 |
| 4 | 128 | 5.488 | 5.090 to 6.067 | 186,598 | 209,147 | 4,136 |

The separate full-path case used 128 scenarios, workers 2 and batch 16, with other model settings unchanged. It measured median 2.328 ms/op, range 1.898 to 5.289 ms/op, median 54,986 scenarios/s and asset paths/s, 558,559 B/op and 946 allocs/op. Its smaller scenario count and extra output make it a separate workload.

Commands and profile files are produced by `sh scripts/benchmark-cpu.sh` inside the pinned image. The final raw CPU output is local at `benchmarks/generated/cpu.txt`. The corrected CPU profile attributed 32.15% of samples to `math.archExp`, 19.39% to `math.archLog`, and 17.02% directly to `simulate`; sampling does not prove that replacing those functions preserves numerical behavior. No unsafe code, assembly, custom compiler, or speculative micro-optimization was added.

## Matched CPU and CUDA container measurements

The official CUDA 13.2 development image download did not finish in the available session and was canceled. An already installed CUDA 12.4 image supplied a separately identified container environment. Its actual native compiler was nvcc 12.4.131. The Go 1.27.1 installation was copied through a local multistage Docker build from the pinned Go image above. The native GPU correctness suite passed in that container, reporting runtime API 12040 and driver API 13050 on the RTX 3060 Laptop GPU.

The local CUDA base was `fin-dl-benchmark:cuda12.4@sha256:98e28ee2f2323e68a2fdc5a571e4c41e93e41305da77a1f8baad15cfb0802e64`. The resulting local image was `quantaspindle-cuda-benchmark@sha256:bccbfdea8c8dcfaf6d47a46633f5e0964b10151778960d8ee635a0751660d05d`. Its OS was Ubuntu 22.04.4 LTS, with GCC 11.4.0, the same WSL2 kernel, 16 visible logical CPUs, and 8,182,960,128 bytes assigned memory. These are local image identities, not a promise that either image is publicly downloadable. Only their CUDA/host compiler toolchain was used; no existing application source, model, or data from that image was used. Reproducing the exact environment requires the recorded local image. Another supported CUDA 12.4 development image can run the same source and commands but constitutes a new measured environment.

Source: delivery commit `3f41ec8`, containing the reviewed CPU/CUDA engine. The runner was compiled with `nvcc -std=c++17 -O3 --fmad=false -arch=sm_86 -cudart=static accelerators/cuda/runner.cu -o local/linux/quantaspindle-cuda`. Both backends used Go 1.27.1, one worker, seed 1, price 100, drift 0.05, volatility 0.2, 252 steps, DT 1/252, no paths or regimes, and one batch per invocation. GOMAXPROCS was 16. One unreported warm-up preceded five measured trials per backend and batch size. The clock covers Run, including CUDA process startup, device initialization, transfers, allocation, synchronization, decoding, and aggregation. Compilation, explicit pre-trial GC, memory statistics reads, and output JSON encoding are outside the timed region.

| Backend | Scenarios and batch | Full median ms | Full min to max ms | Median scenarios/s and asset paths/s | Kernel median ms (min to max) | Runner host median ms | Go allocated bytes | Go allocations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | 256 | 2.841 | 2.691 to 2.929 | 90,093 | Not applicable | Not applicable | 58,496 | 1,040 |
| CUDA | 256 | 170.781 | 150.714 to 239.890 | 1,499 | 5.208 (4.861 to 6.250) | 123.221 | 149,152 | 1,128 |
| CPU | 4,096 | 45.810 | 45.039 to 46.957 | 89,414 | Not applicable | Not applicable | 819,776 | 16,399 |
| CUDA | 4,096 | 234.936 | 176.510 to 269.905 | 17,435 | 5.507 (5.322 to 7.266) | 173.613 | 1,390,528 | 16,484 |
| CPU | 16,384 | 197.988 | 189.792 to 201.047 | 82,752 | Not applicable | Not applicable | 3,277,360 | 65,550 |
| CUDA | 16,384 | 222.015 | 195.858 to 240.858 | 73,797 | 10.333 (10.207 to 11.035) | 159.920 | 5,421,472 | 65,639 |

Go allocations are per invocation and exclude the native child and device. GPU event timings measure kernels only. Runner host timing includes its setup and transfer work. Neither is a substitute for full Go time. Different columns report separate medians and should not be subtracted to derive unmeasured transfer or startup components. The initial container attempt passed device tests, then failed to open the output directory under its non-root default user. Measurements were subsequently run with `--user 0` and completed successfully.

An unrelated Python GPU process was visible during the container setup; no user process was stopped. These shared-host, sequential trials do not establish dedicated-GPU performance. The CUDA median full execution time was higher than the CPU median at all three tested sizes. No general crossover size or GPU speedup is claimed. CPU/CUDA random streams differ; analytical and moment comparisons test their distributions rather than identical paths. See docs/benchmarks.md for exact reproduction commands. Raw records remain local at `benchmarks/generated/batches-cpu.jsonl` and `benchmarks/generated/batches-cuda.jsonl`.

Peak process memory, GPU memory peaks, and CUDA Nsight traces have not been collected. Go allocation measurements exclude native/device allocations. FPGA performance modeling, RTL simulation, vendor emulation, and physical board tests are not implemented or run. No board support or FPGA performance is claimed.

## Standard Go PGO experiment

The same CPU image ran `go test -pgo=off -run '^$' -bench 'BenchmarkRun/workers=4/batch=128$' -benchtime 500ms -benchmem -count 5 .`, followed by the same command with `-pgo=profiles/cpu.prof`. The profile was collected from the same 252-step GBM model. PGO means profile-guided optimization using the standard Go compiler. No compiler modifications were made.

PGO off measured median 4.336 ms/op, range 4.154 to 4.620. PGO on measured median 3.939 ms/op, range 3.784 to 5.729. These sequential samples overlap and had different host conditions from the earlier baseline, so they do not establish a stable improvement. PGO is not enabled by default. Reproduce both commands with a newly collected representative profile rather than assuming these values transfer. Raw outputs and profile remain local at `benchmarks/generated/pgo-off.txt`, `benchmarks/generated/pgo-on.txt`, and `profiles/cpu.prof`; `scripts/benchmark-cpu.sh` regenerates the profile.

## Initial private release preparation

At this preparation stage, the v0.1.0 private tag and release were planned after release review and CI, and the repository was private under the owner's instruction. This paragraph is historical; the public publication follow-up below supersedes its status. An authenticated clean private consumer successfully ran `go mod download` for commit `19779c16763ee0ad705ba7c2697e01cb9c97f1da`. GitHub Actions CPU checks passed on that delivery commit (run 43687073691). At that stage, public clean-consumer fetch, Go proxy/index discovery, pkg.go.dev documentation/badges, and a Go Report Card check awaited visibility and subsequent validation. The completed publication follow-up below records their later outcomes. Authoritative registry/name searches and trademark clearance remain outside those checks. The authenticated private-vulnerability-reporting endpoint returned 404 on 2026-11-15, so no private reporting route was advertised at that stage. The local checks did not themselves authorize a public indexing request or visibility change.

## Public publication follow-up — 2026-11-15

The owner subsequently made [Roonil03/QuantaSpindle](https://github.com/Roonil03/QuantaSpindle) public. GitHub reports public visibility and detects the existing license as GPL-3.0. This supersedes the visibility and reporting-route status in the historical preparation record above. The v0.1.0 tag remains at commit `3ffad625663d2ec24a54de6aabc41252491c9210`; it was not moved for that publication-verification follow-up. That follow-up did not cut a new version; the separately scoped v0.1.1 optimization is later work.

A fresh consumer downloaded `github.com/Roonil03/QuantaSpindle@v0.1.0` without authentication, using `GOPROXY=https://proxy.golang.org`, an isolated `GOMODCACHE=local/public-consumer-cache`, and empty `GOPRIVATE` and `GONOSUMDB`. The download succeeded and reported origin commit `3ffad625663d2ec24a54de6aabc41252491c9210` and module checksum `h1:AKKiG8Vv3fphJDgM/YU+3PM+Q1TIKzqcbLzzv4TQUII=`. This verifies public proxy retrieval of the tagged source; it does not add model calibration or production validation.

An indexing POST to `https://pkg.go.dev/fetch/github.com/Roonil03/QuantaSpindle@v0.1.0` returned HTTP 200. A subsequent GET of the [versioned package page](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.0) returned HTTP 200 and displayed the package documentation, version v0.1.0 and license GPL-3.0. The [reference badge](https://pkg.go.dev/badge/github.com/Roonil03/QuantaSpindle.svg) also returned HTTP 200 with valid SVG. The README linked to that verified version at that checkpoint; its current versioned link follows the latest documented release.

GitHub private vulnerability reporting was enabled through the repository API; a subsequent GET confirmed it is enabled. [SECURITY.md](../SECURITY.md) now links to the [private advisory reporting route](https://github.com/Roonil03/QuantaSpindle/security/advisories/new). No response-time or supported-version policy was added.

The Go Report Card project URL returned HTTP 200 with the title `Go Report Card | Thank you` and a service-retirement message; the [official service homepage](https://goreportcard.com/) independently confirmed retirement. It is not an active project assessment, so no grade or badge is claimed. Authoritative registry/name checks and trademark clearance remain outside these publication checks.

## v0.1.1 checks on 2026-11-15

- [GitHub Actions run 43689434736](https://github.com/Roonil03/QuantaSpindle/actions/runs/43689434736) passed for implementation commit `b5f1a31`: formatting, vet, CPU tests, Go race detection, and benchmark smoke checks. The implementation was merged through [PR #7](https://github.com/Roonil03/QuantaSpindle/pull/7); the later documentation/measurement follow-up is separate.
- The benchmark script rebuilt the CUDA runner from that implementation and passed `QS_CUDA_RUNNER=/work/local/linux/quantaspindle-cuda go test -count=1 ./cuda -v` in the recorded CUDA 12.4.131/Go 1.27.1 Linux container. The local complete log is `benchmarks/generated/cuda-resident-final/device-tests.txt`; the environment and source identity are in [CUDA performance](cuda-performance.md).
- The native Windows runner was rebuilt and passed `go test -count=1 ./cuda -v` on the RTX 3060 Laptop GPU, reporting CUDA runtime API 13020 and driver API 13050. The ordinary CLI also completed 257 scenarios in five batches with three workers. These are correctness/lifecycle checks, not Windows performance measurements.
- Real-device checks passed analytical and CPU/GPU moment comparisons, one-shot replay, malformed native inputs, process deadline, and `TestCUDAStreamingIntegration`. The stream test checks exact output against one-shot results, replay under shifted batch partitions, and overlapping independent runs using the same backend.
- Portable subprocess tests passed single-child reuse, queued terminal-failure propagation, truncated/malformed/extra output, exit/shutdown failures, bounded diagnostics, stale-runner startup, sink-error cleanup, and active/queued cancellation with process reaping. Those fake-child cancellation tests establish adapter behavior; they are distinct from GPU device execution.
- `TestSessionLifetimeExcludesIdleFromBatchWall` passed: idle/sink time belongs to the completed streaming session lifetime, not the successful batch exchange sum. Full benchmark latency wraps `Run` through session closure.
- v0.1.1 public module retrieval and pkg.go.dev publication verification are recorded in the new GitHub release notes after the tag is published. The v0.1.0 public checks above do not establish v0.1.1 availability.
