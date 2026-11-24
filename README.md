# QuantaSpindle

![License: GPL version 3](https://img.shields.io/badge/license-GPL--3.0-blue)
![Go version](https://img.shields.io/badge/Go-1.27.1-00ADD8)
[![Checks](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml/badge.svg)](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Roonil03/QuantaSpindle.svg)](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.2)

QuantaSpindle is a Go library and CLI for reproducible synthetic financial market scenarios. It provides bounded CPU simulation and an optional native CUDA backend, with ordered results, cancellation, and explicit model assumptions.

The v0.1.2 CLI chooses CPU or CUDA before generating a run's output. It keeps the requested financial model, seed and output fields. The execution backend determines the random generator, so pin `-backend cpu` or `-backend cuda` when replaying a run. The v0.1.1 CLI defaults to CPU.

## Install and run

Use the official Go 1.27.1 toolchain or a later supported release. There are no external Go module dependencies. CPU-only builds need no CUDA, C compiler, logger, or database.

Install the tagged library in your Go module:

```sh
go get github.com/Roonil03/QuantaSpindle@v0.1.2
```

Install and run the CLI:

```sh
go install github.com/Roonil03/QuantaSpindle/cmd/quantaspindle@v0.1.2
quantaspindle -paths 10000 -steps 252 -seed 42 -workers 4 -batch 128 -summary-only
```

To build and test automatic selection from a checkout, run:

```sh
go test ./...
go run ./examples/basic
go run ./cmd/quantaspindle -backend auto -paths 4096 -steps 252 -seed 42 -summary-only
go run ./cmd/quantaspindle -backend auto -paths 262144 -steps 252 -seed 42 -summary-only
go build ./cmd/quantaspindle
```

The first JSON record identifies the requested and selected backend. A CPU-only machine can use the same tagged CLI installation.

CLI output is newline-delimited JSON. The first record contains configuration and toolchain provenance. Scenario records follow in ID order, then a summary. `-summary-only` omits individual records. A failed run emits an error, returns a nonzero exit status, and may already have delivered a prefix. It never emits a successful final summary after failure. `-timeout 30s` bounds the run. Interrupting the process cancels work.

`-config path.json` loads a strict JSON object over explicit defaults. Unknown fields, extra JSON values, and invalid parameters return errors. Explicit flags override matching config values. `-store-paths` requests asset-major paths including the initial price. Use the Go API for custom models, sinks, portfolio holdings, and risk/impact summaries.

```go
package main

import (
    "context"
    "fmt"
    "log"

    qs "github.com/Roonil03/QuantaSpindle"
)

func main() {
    cfg := qs.DefaultConfig()
    cfg.Paths, cfg.Workers, cfg.BatchSize = 10000, 4, 128
    cfg.Seed = 42
    summary, err := qs.Run(context.Background(), cfg, qs.CPU{}, nil)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(summary.Scenarios, summary.MeanReturns)
}
```

## Results and limits

- **Outputs:** final prices, simple returns, annualized sample volatility of step log returns, and maximum fractional drawdown on the sampled grid. Full paths are optional on CPU.
- **CPU:** single-asset and correlated multiasset GBM, optional discrete Markov regimes, and scenario streams unaffected by worker count or batch partition.
- **CUDA:** actual FP64 kernels for one asset with fixed drift and volatility. No supplied correlation, regimes, or stored paths. An explicit CUDA selection never falls back to CPU.
- **Performance:** a prepared GPU reduced measured request latency at 4,096 and 16,384 scenarios, while CPU was faster when GPU setup was spread across only six requests. Cold automatic calls chose CPU at those sizes and CUDA at 524,288. [Startup methodology and limits](docs/cuda-startup.md) and the separately scoped [v0.1.1 measurements](docs/cuda-performance.md) record the complete timings.
- **Interpretation:** presets are illustrative and uncalibrated. Scenarios follow caller assumptions; they are not forecasts. VaR/expected shortfall are finite-sample estimates, and the impact overlay is model dependent.
- **Scope:** no jumps, volatility clustering, market-data loading, dividends, financing, or transaction matching. No FPGA execution, supported board, vendor emulation, or measured FPGA result.
- **Replay:** scoped to the same implementation, backend, and toolchain; CUDA additionally requires the same runner build and device. CPU and CUDA use different generators and are compared statistically.

## Execution and replay

`Run` schedules a bounded wave of batch jobs, waits for that wave, and calls the sink serially in scenario order. It retains no observations across waves. The sink owns delivered slices and can retain them at its own memory cost. Errors return a summary of the successfully delivered prefix. Exact tail estimates require explicit retained observations. Store complete paths only when needed.

CPU streams use versioned SplitMix64 mixing and Box-Muller normals, with separate regime streams. Run seed and scenario ID determine each stream. Worker count and batch partition do not change CPU results. Replay is scoped to the same implementation, backend, and toolchain. CPU and CUDA use different generators and floating-point functions, so their paths are compared statistically.

Inputs are validated before workers start. Limits include 256 assets, four regimes, 1,000,000 steps, 1,000,000,000 scenarios, 1,024 workers, 16,384 scenarios per batch, and an estimated 512 MiB wave budget. This budget includes CPU correlation setup and estimated live scenario storage. It excludes caller retention, input storage, runtime overhead, and arbitrary allocations by custom backends. Reduce workers or batch size for large stored paths. A Backend must support concurrent batches, owned result slices, valid complete results, and cancellation.

## Optional CUDA

The Go adapter calls a separately built CUDA C++ runner. With `&cuda.Backend{}`, one child process serves all batches in a `Run`. A service handling successive requests can instead call `Prepare` once and pass its result to `hybrid.Run`. This keeps the initialized GPU available until the service closes it. Ordinary CPU builds use the Go standard library alone. [Setup and service example](docs/cuda.md#automatic-selection-and-prepared-cuda-v012)

```powershell
# Run in a Visual Studio x64 developer shell.
./scripts/build-cuda.ps1 -Nvcc 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe'
$env:QS_CUDA_RUNNER = (Resolve-Path ./local/quantaspindle-cuda.exe).Path
go test ./cuda -v
go run ./cmd/quantaspindle -backend cuda -cuda-runner $env:QS_CUDA_RUNNER -workers 1 -batch 16384 -paths 65536 -summary-only
```

The Windows amd64 runner was tested with CUDA 13.2.51, Visual Studio 2022 and an RTX 3060 Laptop GPU (`sm_86`) on driver 621.24. The startup measurements used the same hardware from a Linux CUDA 12.4.131 container. NVIDIA tooling and libraries are separately licensed. [CUDA setup](docs/cuda.md) · [Validation record](docs/validation.md)

## Validation and measurements

```sh
gofmt -l .
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -bench . -benchmem -count 5 ./...
go run ./cmd/qsbench -backend cpu
```

The race detector needs cgo and a supported C compiler. It covers Go concurrency, not GPU memory accesses. Conditional hardware tests skip without `QS_CUDA_RUNNER`; setting it makes device failures fail the tests. Analytical GBM moments, empirical innovation correlations, replay, regimes, cancellation, backend validation, risk, impact, and native protocol errors have synthetic tests.

### v0.1.2 automatic selection

Five trials on an RTX 3060 Laptop GPU (6,144 MiB) and i9-11900H (8 cores/16 logical CPUs), using Go 1.27.1 and CUDA 12.4.131 in a Linux container, compared single-asset FP64 workloads of 252 steps. Cold calls include selection, GPU startup when required, and shutdown; ready calls use an already prepared service. CLI startup and JSON encoding are excluded.

At 4,096 and 16,384 scenarios, ready automatic CUDA took median 2.67 and 10.44 ms per request, versus 6.54 and 26.75 ms for the fastest tested CPU. Cold automatic calls chose CPU. Including preparation and shutdown across six requests raised automatic per-request medians to 25.61 and 34.66 ms, both slower than CPU. At 524,288 scenarios, cold automatic CUDA took 417.57 ms versus 810.01 ms for CPU; ready CUDA took 282.18 ms, or 311.10 ms per request across six requests including setup and shutdown.

Fresh forced CUDA trial ranges overlapped v0.1.1 at every tested size, so no cold CUDA improvement over that release is established. These shared-laptop samples do not establish statistical significance or a general crossover. The [startup report](docs/cuda-startup.md) records complete timings, ranges, configurations and reproduction commands for [source `2db2704`](https://github.com/Roonil03/QuantaSpindle/commit/2db27040b08b6094766196bf0d93b99e2114093b). [All 380 latency samples](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836)

### v0.1.1 full-run latency

In a separate five-trial Linux-container experiment on the same laptop model and toolchain versions, complete single-asset FP64 `Run` medians at 262,144 and 524,288 scenarios were 381.30 and 505.46 ms for v0.1.1 CUDA, versus 509.16 and 980.90 ms for the fastest tested CPU (1.34x and 1.94x ratios). Times include CUDA startup, initialization, transfers and shutdown. CPU won the smaller cases. At 524,288 scenarios, v0.1.1 was 11.86x faster than the matched v0.1.0 one-process-per-batch native runner; these optimized results do not belong to v0.1.0.

The 262,144-scenario CPU/CUDA trial ranges overlapped. These shared-laptop samples do not establish statistical significance, a general crossover, or dedicated-GPU performance; CPU and CUDA random algorithms differ. The [performance report](docs/cuda-performance.md) records exact source identities, workloads, timing boundaries, trial ranges, allocations, reproduction commands and limits. Raw `measure.jsonl` trials are in the [v0.1.1 release assets](https://github.com/Roonil03/QuantaSpindle/releases/tag/v0.1.1). Earlier CPU scaling, single-batch measurements and PGO results remain separately scoped in [validation](docs/validation.md).

## Model contracts

The CPU supports single-asset and correlated multiasset geometric Brownian motion (GBM), a process whose step log returns are normal under fixed parameters. Four illustrative bull, bear, sideways, and stress presets are available through `DefaultRegimeConfig()`. You can supply your own parameters and transition matrix. These presets have no market calibration. A scenario is an outcome under the supplied assumptions, not a forecast.

Each interval uses `S_next = S * exp((mu - sigma*sigma/2)*dt + sigma*sqrt(dt)*z)`. Time is in years. Drift and volatility are annualized. Correlation describes Brownian innovations. Symmetric positive semidefinite matrices, including singular perfect correlations, are accepted within the documented numerical tolerance. No jitter is silently added. A regime governs the current interval and transitions afterward. Transition probabilities apply per step; changing `DT` also changes their time interpretation.

Each scenario reports final prices, simple returns, annualized sample volatility of step log returns, and maximum fractional drawdown on the sampled grid. One interval has zero sample volatility. Intrastep price extrema are not observed. Jumps, volatility clustering, calibration, dividends, financing, transaction matching, and market data loading are outside this release. Prices stay positive in the model; numerical overflow or underflow returns an error.

`RiskFromReturns` and `RiskFromLosses` compute exact empirical VaR and expected shortfall from observations retained by the caller. VaR uses the nearest rank `ceil(confidence*N)`. Expected shortfall averages the worst `ceil((1-confidence)*N)` losses, with at least one observation and documented near-integer rounding. Negative risk values are preserved for all-gain samples. These finite-sample estimates are not regulatory capital calculations.

`SquareRootImpact` is an optional execution overlay. It reports volume, spread, signed order size, daily-volume participation, temporary and permanent impact fractions, execution price, post-trade mid, and cash cost. Its coefficients and liquidity inputs are caller supplied. All impact estimates are model dependent. The overlay does not change simulated paths automatically.

## Documentation

- [CUDA startup and automatic selection](docs/cuda-startup.md) gives small-batch comparisons with v0.1.1 and CPU, including one-off calls and repeated requests.
- [CUDA build and backend guide](docs/cuda.md): optional runner setup, supported inputs, sessions, replay, protocol, and timing definitions.
- [CUDA optimization measurements](docs/cuda-performance.md): matched workloads, repeated trials, outcomes, and measurement limits for the follow-up.
- [Benchmark reproduction](docs/benchmarks.md): commands, environment identities, trial policy, profiling, and timing boundaries.
- [Validation record](docs/validation.md): completed checks, historical baselines, and publication verification, each scoped to its source/version.
- [Design review](docs/design-review.md): mathematical choices, execution contracts, independent origin, and subsequent design amendments.
- [FPGA extension guide](docs/fpga.md) and [configuration template](examples/fpga-v1.json): a future backend contract; no executable FPGA implementation.
- [Competitive comparison](docs/competitive-landscape.md): reference scope and provisional naming limits.
- [Changelog](CHANGELOG.md) and [release procedure](docs/release.md): version-specific changes and publication checks.
- [Contribution guide](CONTRIBUTING.md), [conduct rules](CODE_OF_CONDUCT.md), and [security reporting](SECURITY.md): collaboration and private vulnerability reporting.
- [Credits](CREDITS.md), [notice](NOTICE.md), and [license](LICENSE): attribution, independent origin, and GNU GPL version 3 terms.

The project name remains provisional pending fuller registry and trademark checks. [Go Report Card](https://goreportcard.com/) has retired, so no grade or badge is claimed.

Copyright 2026 QuantaSpindle contributors. Licensed under [GNU GPL version 3](LICENSE).
