# Changelog

## Unreleased

- Include Go build information and per-mode resolved CUDA runner SHA-256 identity in direct `qsbench` trial records, collected before timing.
- Add `CPU.Prepare` and `PreparedCPU.Batch` to reuse a validated correlation factor across arbitrary batches, with owned configuration snapshots and concurrent calls.
- Preserve cancellation residuals in run summary means, finite means when scenario totals overflow, and successfully delivered partial-run summaries.
- Retain representable CPU price increments from tiny log returns using compensated exponential updates.
- Preserve finite impact prices and costs when spread plus temporary impact exceeds float64, and retain representable permanent midpoint moves.
- Allow overflowing portfolio loss products to cancel before rounding the final loss to float64.
- Preserve sample standard deviations for tiny and subnormal losses in risk summaries.
- Compute finite risk means and expected shortfall across signed loss extremes, preserving cancellation residuals and subnormal mean rounding.
- Deduplicate effective mode/configuration pairs in `qsbench` sweeps so each case receives one warm-up and one sample per trial.

## v0.1.2 - 2026-11-15

### Automatic selection and prepared CUDA

- The CLI now defaults to `-backend auto`. The new `hybrid` Go package selects one backend before delivering output and reports its decision. CPU handles multiple assets, regimes, correlations and stored paths. Pin `-backend cpu` or `-backend cuda` for replay: the backends use different random generators.
- `cuda.Backend.Prepare` initializes a reusable device process for successive requests. Applications own its lifetime and can pass it to `hybrid.Run` or `quantaspindle.Run`. The ordinary CUDA backend continues to close its process at the end of each run.
- Automatic selection considers cold CUDA from 262,144 scenarios and prepared CUDA from 4,096, with at least 252 steps and a supported single-asset model. These are measured-workload heuristics, not a general crossover guarantee. Missing runners and classified device initialization failures can select CPU before output. Cancellation, protocol and numerical errors still propagate; a later failure never mixes CPU and CUDA output in one run.

### Measured latency and scope

- Five trials on an NVIDIA RTX 3060 Laptop GPU (6,144 MiB) and Intel Core i9-11900H (8 cores/16 logical CPUs), using Go 1.27.1 and CUDA 12.4.131 in a Linux container, compared one-asset FP64 workloads with 252 steps, seed 1 and no stored paths. Complete request clocks include routing, execution and owned-process cleanup; CLI startup and JSON encoding are excluded. Ready calls use an already prepared process. The six-request figures include preparation, all requests and shutdown, divided by six.
- At 4,096 and 16,384 scenarios, ready automatic CUDA medians were **2.671 ms and 10.444 ms**, versus the fastest tested CPU medians of **6.536 ms and 26.751 ms** (2.45x and 2.56x ratios). Cold automatic calls selected CPU and took 6.210 ms and 29.236 ms. Including setup and shutdown over six requests raised automatic per-request medians to **25.614 ms and 34.662 ms**, both slower than CPU. At 256 scenarios both policies selected CPU.
- At 524,288 scenarios, cold automatic CUDA took **417.568 ms** versus **810.008 ms** on the fastest tested CPU; ready automatic CUDA took 282.183 ms, and six requests including setup averaged 311.097 ms per request. Fresh forced CUDA and tagged v0.1.1 had overlapping trial ranges at every measured size. The measurements do not establish a cold CUDA improvement over v0.1.1, statistical significance, or performance on other hardware.
- The [startup report](https://github.com/Roonil03/QuantaSpindle/blob/v0.1.2/docs/cuda-startup.md) records all trial ranges, timing boundaries, configurations, hardware and reproduction commands. Measurements used [source `2db2704`](https://github.com/Roonil03/QuantaSpindle/commit/2db27040b08b6094766196bf0d93b99e2114093b), compared with the unchanged [v0.1.1 source](https://github.com/Roonil03/QuantaSpindle/tree/v0.1.1). [All 380 public latency samples](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836) retain those source identities. Earlier v0.1.1 and v0.1.0 measurements below remain scoped to their original experiments.

### Validation

- The measured implementation passed rebuilt real-device tests before timing and `go test -race -count=1 ./...` afterward. The record checker verified 340 current, 20 tagged and 20 alternate-runner records, including routing and replay. Tested CUDA summaries matched v0.1.1 exactly; CPU-routed summaries matched explicit CPU. See the [source-specific validation record](https://github.com/Roonil03/QuantaSpindle/blob/v0.1.2/docs/cuda-startup.md#hardware-validation-and-source) and [release checks](https://github.com/Roonil03/QuantaSpindle/blob/v0.1.2/docs/validation.md) for their respective scope and outcomes.

## v0.1.1 - 2026-11-15

### Changed

- Added optional per-run backend sessions. `Run` owns session startup and closure; existing batch-only backends retain their interface.
- CUDA serves batches through one native `--stream` process per run and reuses device resources. Direct `Backend.Batch` keeps the v1 one-shot interface. The CLI needs no new flags.
- Preserved the FP64 model and Philox seed/scenario-ID mapping; unsupported CUDA models still fail explicitly. Rebuild the native runner with the Go adapter for stream capability.
- Reorganized the README around usage, outputs, limits, and a point-wise documentation index. Historical baselines and optimized measurements are separately scoped.

### Measurements and validation

- Passed [implementation CI](https://github.com/Roonil03/QuantaSpindle/actions/runs/43689434736) on `b5f1a31` and the rebuilt real-GPU suite in the recorded CUDA 12.4.131 container. Streaming output exactly matched one-shot replay across partitions and overlapping independent runs; portable subprocess tests covered cancellation, protocol failures, and cleanup.
- On the recorded Linux RTX 3060 Laptop/i9-11900H setup, full-run CUDA medians at 262,144 and 524,288 scenarios were 381.30 ms and 505.46 ms, versus the fastest measured CPU configurations at 509.16 ms and 980.90 ms (1.34x and 1.94x). At 524,288 scenarios, the released one-shot runner measured 5,997.17 ms (11.86x the v0.1.1 latency). These are sample-median ratios; the 262,144-scenario CPU/CUDA trial ranges overlapped. CPU won all three smaller cases; these shared-host results do not establish statistical significance or a general crossover.
- The five-trial experiment, exact settings, source identities, timing boundaries, min/max and limits are recorded in [CUDA performance](docs/cuda-performance.md), with raw `measure.jsonl` trials in the [v0.1.1 release assets](https://github.com/Roonil03/QuantaSpindle/releases/tag/v0.1.1). Historical v0.1.0 figures below remain measurements of the original source.
- The original v0.1.0 tag is unchanged. Optimization behavior and follow-up measurements belong to v0.1.1; editing the original GitHub release notes does not alter that tag.

### Documentation and publication verification

- Updated installation instructions for the public repository and verified an unauthenticated v0.1.0 download through the public Go proxy on 2026-11-15.
- Verified the versioned pkg.go.dev API documentation and added its working reference badge.
- Enabled and verified GitHub private vulnerability reporting, and documented its private reporting URL.
- Recorded that Go Report Card has retired; no grade or badge is claimed.

## v0.1.0 - 2026-11-15

### Added

- An independent design review and source comparison for the Go and CUDA scenario engine.
- Contribution, conduct, security, attribution, and release documentation under GNU GPL version 3.
- A Go API for multiasset GBM, configurable regimes, validated semidefinite correlations, deterministic scenario streams, bounded ordered streaming, and cancellation.
- Risk summaries, a separate square-root execution-impact estimate, a streaming JSON CLI, and a Go usage example.
- Optional real FP64 CUDA kernels, a bounded binary process adapter, device metadata and timing, build guides, and conditional hardware tests.
- Synthetic correctness tests, repeated CPU benchmarks, profiling commands, a standard Go PGO comparison, and CI configuration.
- An FPGA extension guide and configuration template. No FPGA execution is implemented.

### Validation

- Passed CPU unit and race checks in the pinned Go 1.27.1 Linux container.
- Passed rebuilt native RTX 3060 CUDA correctness tests, including replay, analytical and CPU moment checks, malformed requests, output limits, and process cancellation.
- Recorded five-trial CPU scaling measurements and a separate standard Go PGO experiment with explicit limits in docs/validation.md.
- Passed real GPU tests in a separately identified CUDA 12.4 Linux container and recorded matched five-trial CPU/CUDA batch measurements. The tested runner did not beat CPU median full execution time at those batch sizes.
- Passed GitHub Actions CPU checks on delivery commit `19779c1` (run 43687073691), and authenticated private consumer download at that commit.

### CUDA optimization follow-up in v0.1.1

- CUDA process/device reuse and its new measured results require [v0.1.1](https://github.com/Roonil03/QuantaSpindle/releases/tag/v0.1.1); see the [performance report](docs/cuda-performance.md). In the follow-up shared-host experiment, 524,288 scenarios took 505.46 ms with v0.1.1 CUDA versus 5,997.17 ms with the matched released one-shot runner and 980.90 ms with the fastest measured CPU configuration. CPU won the three smaller cases. These optimized results require v0.1.1 and are not behavior or performance of the v0.1.0 source. The original tag and historical results are unchanged.

### Known limitations

- At the initial release checkpoint, public fetching and pkg.go.dev indexing awaited visibility/publication verification. The v0.1.1 follow-up above records the later successful v0.1.0 checks.
- At that checkpoint, no private vulnerability reporting route had been verified. It was enabled and verified in the later follow-up above.
- CUDA supports one asset with constant GBM parameters, without regimes, correlation or stored paths.
- Market calibration is not provided.
- No FPGA board, vendor emulation, or bitstream has been validated.
