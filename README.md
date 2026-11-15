# QuantaSpindle

![License: GPL version 3](https://img.shields.io/badge/license-GPL--3.0-blue)
![Go version](https://img.shields.io/badge/Go-1.27.1-00ADD8)
[![Checks](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml/badge.svg)](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Roonil03/QuantaSpindle.svg)](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.1)

QuantaSpindle is a Go library and CLI for reproducible synthetic financial market scenarios. It provides bounded CPU simulation and an optional native CUDA backend, with ordered results, cancellation, and explicit model assumptions.

The CLI on `main` chooses CPU or CUDA before generating a run's output. It keeps the requested financial model, seed and output fields. The execution backend determines the random generator, so pin `-backend cpu` or `-backend cuda` when replaying a run. The published v0.1.1 CLI defaults to CPU.

## Install and run

Use the official Go 1.27.1 toolchain or a later supported release. There are no external Go module dependencies. CPU-only builds need no CUDA, C compiler, logger, or database.

Install the tagged library in your Go module:

```sh
go get github.com/Roonil03/QuantaSpindle@v0.1.1
```

Install and run the CLI:

```sh
go install github.com/Roonil03/QuantaSpindle/cmd/quantaspindle@v0.1.1
quantaspindle -paths 10000 -steps 252 -seed 42 -workers 4 -batch 128 -summary-only
```

To use automatic selection from a checkout of `main`, run:

```sh
go test ./...
go run ./examples/basic
go run ./cmd/quantaspindle -backend auto -paths 4096 -steps 252 -seed 42 -summary-only
go run ./cmd/quantaspindle -backend auto -paths 262144 -steps 252 -seed 42 -summary-only
go build ./cmd/quantaspindle
```

To install the CLI with automatic selection directly from the public branch, use `go install github.com/Roonil03/QuantaSpindle/cmd/quantaspindle@main`. Its first JSON record identifies the requested and selected backend. A CPU-only machine can use the same command.

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
- **Performance:** the v0.1.1 CUDA runner beat the fastest measured CPU configuration on the two largest tested workloads; CPU won the three smaller cases. Full timings and test settings appear below, with [complete methodology and limits](docs/cuda-performance.md).
- **Interpretation:** presets are illustrative and uncalibrated. Scenarios follow caller assumptions; they are not forecasts. VaR/expected shortfall are finite-sample estimates, and the impact overlay is model dependent.
- **Scope:** no jumps, volatility clustering, market-data loading, dividends, financing, or transaction matching. No FPGA execution, supported board, vendor emulation, or measured FPGA result.
- **Replay:** scoped to the same implementation, backend, and toolchain; CUDA additionally requires the same runner build and device. CPU and CUDA use different generators and are compared statistically.

## Execution and replay

`Run` schedules a bounded wave of batch jobs, waits for that wave, and calls the sink serially in scenario order. It retains no observations across waves. The sink owns delivered slices and can retain them at its own memory cost. Errors return a summary of the successfully delivered prefix. Exact tail estimates require explicit retained observations. Store complete paths only when needed.

CPU streams use versioned SplitMix64 mixing and Box-Muller normals, with separate regime streams. Run seed and scenario ID determine each stream. Worker count and batch partition do not change CPU results. Replay is scoped to the same implementation, backend, and toolchain. CPU and CUDA use different generators and floating-point functions, so their paths are compared statistically.

Inputs are validated before workers start. Limits include 256 assets, four regimes, 1,000,000 steps, 1,000,000,000 scenarios, 1,024 workers, 16,384 scenarios per batch, and an estimated 512 MiB wave budget. This budget includes CPU correlation setup and estimated live scenario storage. It excludes caller retention, input storage, runtime overhead, and arbitrary allocations by custom backends. Reduce workers or batch size for large stored paths. A Backend must support concurrent batches, owned result slices, valid complete results, and cancellation.

## Optional CUDA

The Go adapter calls a separately built CUDA C++ runner. With `&cuda.Backend{}`, one child process serves all batches in a `Run`. A service handling successive requests can instead call `Prepare` once and pass its result to `hybrid.Run`. This keeps the initialized GPU available until the service closes it. Ordinary CPU builds use the Go standard library alone. [Setup and service example](docs/cuda.md#automatic-selection-and-prepared-cuda-on-main)

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

### Automatic selection on main

Five trials on an RTX 3060 Laptop GPU and i9-11900H compared single-asset workloads of 252 steps. Times are median milliseconds per request. Cold calls include backend selection, any required GPU startup, and shutdown. Ready calls use an already prepared service. The last column includes preparing that service once, running six requests, and closing it, divided by six.

| Scenarios | v0.1.1 cold CUDA | Fastest tested CPU | Cold automatic | Ready automatic | Six requests including setup, per request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 139.96 | 0.78 | 0.83 CPU | 0.76 CPU | 22.93 |
| 4,096 | 141.90 | 6.54 | 6.21 CPU | 2.67 CUDA | 25.61 |
| 16,384 | 147.18 | 26.75 | 29.24 CPU | 10.44 CUDA | 34.66 |
| 524,288 | 407.64 | 810.01 | 417.57 CUDA | 282.18 CUDA | 311.10 |

At 4,096 and 16,384 scenarios, routing to an already prepared GPU gave a 2.45× and 2.56× speedup over the fastest measured CPU medians. CPU was faster when setup was spread across only six requests at either size. Fresh forced CUDA took 138.90, 138.80, 139.22 and 395.41 ms in workload order. Its trial ranges overlapped v0.1.1. [Method and trial ranges](docs/cuda-startup.md) · [All 380 latency samples](https://github.com/Roonil03/QuantaSpindle/pull/10#issuecomment-6891525836)

### v0.1.1 full-run latency

Linux container on the i9-11900H (8 cores/16 logical CPUs) and RTX 3060 Laptop GPU: Go 1.27.1, CUDA 12.4.131, FP64, one asset, initial price 100, drift 0.05, volatility 0.2, seed 1, 252 steps, DT 1/252, summaries without stored paths. One warm-up preceded five measured trials per case; case order reversed on alternate trials. CUDA used one worker and batches up to 16,384. Each CPU row is the fastest median among workers 1/4/8/16 and batches 128/1,024. Times include complete `Run`, CUDA startup, initialization, transfers, and process shutdown; compilation and output JSON encoding are excluded.

| Scenarios | Best CPU workers / batch | CPU median ms | v0.1.0 one-shot CUDA median ms | v0.1.1 CUDA median ms |
| --- | --- | --- | --- | --- |
| 256 | 8 / 128 | 1.46 | 145.11 | 159.63 |
| 4,096 | 16 / 128 | 7.84 | 157.99 | 155.66 |
| 16,384 | 16 / 1,024 | 32.25 | 208.47 | 220.07 |
| 262,144 | 16 / 1,024 | 509.16 | 2,942.16 | 381.30 |
| 524,288 | 16 / 1,024 | 980.90 | 5,997.17 | 505.46 |

At 262,144 and 524,288 scenarios, median latency implies 1.34x and 1.94x speedups over the fastest measured CPU configuration. At 524,288 it implies an 11.86x speedup over the matched released one-shot runner. CPU won the smaller cases. The 262,144-scenario trial ranges overlapped; these shared laptop samples do not establish statistical significance, a general crossover, or dedicated-GPU performance. CPU and CUDA random algorithms differ. The released-runner column uses v0.1.0 native source with the current batch-only harness, not a claim that the optimized code exists in v0.1.0.

The [performance report](docs/cuda-performance.md) records source identities, min/max, throughput, allocations, event timings, reproduction commands, and limits. Raw trials are provided as `measure.jsonl` in the [v0.1.1 release assets](https://github.com/Roonil03/QuantaSpindle/releases/tag/v0.1.1). Go allocations exclude native and device allocations; peak memory was not measured. Historical CPU scaling, earlier single-batch CPU/CUDA measurements, and the PGO experiment remain separately labeled in [validation](docs/validation.md).

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
