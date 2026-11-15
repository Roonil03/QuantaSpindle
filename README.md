# QuantaSpindle

![License: GPL version 3](https://img.shields.io/badge/license-GPL--3.0-blue)
![Go version](https://img.shields.io/badge/Go-1.27.1-00ADD8)
[![Checks](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml/badge.svg)](https://github.com/Roonil03/QuantaSpindle/actions/workflows/checks.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Roonil03/QuantaSpindle.svg)](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.0)

QuantaSpindle is a Go library and CLI for reproducible synthetic financial market scenarios. Go runs bounded batches on the CPU and coordinates an optional native CUDA runner. CUDA kernels execute GPU work. Goroutines run on the host.

v0.1.0 is the initial release. QuantaSpindle is independent of the conceptual stock-simulations reference. Its [GitHub repository](https://github.com/Roonil03/QuantaSpindle) is public. An unauthenticated download of v0.1.0 through the public Go proxy and its [published API documentation](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.0) were verified on 2026-11-15. pkg.go.dev indexes fetchable modules; it is not a separate upload service.

## Models and limits

The CPU supports single-asset and correlated multiasset geometric Brownian motion (GBM), a process whose step log returns are normal under fixed parameters. Four illustrative bull, bear, sideways, and stress presets are available through `DefaultRegimeConfig()`. You can supply your own parameters and transition matrix. These presets have no market calibration. A scenario is an outcome under the supplied assumptions, not a forecast.

Each interval uses `S_next = S * exp((mu - sigma*sigma/2)*dt + sigma*sqrt(dt)*z)`. Time is in years. Drift and volatility are annualized. Correlation describes Brownian innovations. Symmetric positive semidefinite matrices, including singular perfect correlations, are accepted within the documented numerical tolerance. No jitter is silently added. A regime governs the current interval and transitions afterward. Transition probabilities apply per step; changing `DT` also changes their time interpretation.

Each scenario reports final prices, simple returns, annualized sample volatility of step log returns, and maximum fractional drawdown on the sampled grid. One interval has zero sample volatility. Intrastep price extrema are not observed. Jumps, volatility clustering, calibration, dividends, financing, transaction matching, and market data loading are outside this release. Prices stay positive in the model; numerical overflow or underflow returns an error.

`RiskFromReturns` and `RiskFromLosses` compute exact empirical VaR and expected shortfall from observations retained by the caller. VaR uses the nearest rank `ceil(confidence*N)`. Expected shortfall averages the worst `ceil((1-confidence)*N)` losses, with at least one observation and documented near-integer rounding. Negative risk values are preserved for all-gain samples. These finite-sample estimates are not regulatory capital calculations.

`SquareRootImpact` is an optional execution overlay. It reports volume, spread, signed order size, daily-volume participation, temporary and permanent impact fractions, execution price, post-trade mid, and cash cost. Its coefficients and liquidity inputs are caller supplied. All impact estimates are model dependent. The overlay does not change simulated paths automatically.

## Install and run

Use the official Go 1.27.1 toolchain or a later supported release. There are no external Go module dependencies. CPU-only builds need no CUDA, C compiler, logger, or database.

Install the library in your Go module:

```sh
go get github.com/Roonil03/QuantaSpindle@v0.1.0
```

Install and run the CLI:

```sh
go install github.com/Roonil03/QuantaSpindle/cmd/quantaspindle@v0.1.0
quantaspindle -paths 10000 -steps 252 -seed 42 -workers 4 -batch 128 -summary-only
```

The public module requires no GitHub credentials or `GOPRIVATE` setting. From a checkout:

```sh
go test ./...
go run ./examples/basic
go run ./cmd/quantaspindle -paths 10000 -steps 252 -seed 42 -workers 4 -batch 128 -summary-only
go build ./cmd/quantaspindle
```

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

## Execution and replay

`Run` schedules a bounded wave of batch jobs, waits for that wave, and calls the sink serially in scenario order. It retains no observations across waves. The sink owns delivered slices and can retain them at its own memory cost. Errors return a summary of the successfully delivered prefix. Exact tail estimates require explicit retained observations. Store complete paths only when needed.

CPU streams use versioned SplitMix64 mixing and Box-Muller normals, with separate regime streams. Run seed and scenario ID determine each stream. Worker count and batch partition do not change CPU results. Replay is scoped to the same implementation, backend, and toolchain. CPU and CUDA use different generators and floating-point functions, so their paths are compared statistically.

Inputs are validated before workers start. Limits include 256 assets, four regimes, 1,000,000 steps, 1,000,000,000 scenarios, 1,024 workers, 16,384 scenarios per batch, and an estimated 512 MiB wave budget. This budget includes CPU correlation setup and estimated live scenario storage. It excludes caller retention, input storage, runtime overhead, and arbitrary allocations by custom backends. Reduce workers or batch size for large stored paths. A Backend must support concurrent batches, owned result slices, valid complete results, and cancellation.

## Optional CUDA

Read [CUDA build, protocol, and tests](docs/cuda.md). The Go adapter uses the maintained standard `os/exec` API and a versioned binary protocol to call a native CUDA C++ runner. This keeps ordinary Go builds free of cgo and vendor dependencies. Each process handles a batch, so startup and device setup can dominate small workloads. Intermediate path state stays on the GPU; only compact summaries return to Go. There is no automatic CPU fallback for an explicit CUDA request.

The tested native target is Windows amd64, CUDA 13.2.51, Visual Studio 2022, and RTX 3060 Laptop GPU (`sm_86`), driver 621.24. See the validation report for container checks and exact versions. Other targets require their own build and device validation. The current CUDA model supports one asset, constant drift and volatility, no supplied correlation or regimes, and no full path output. Unsupported inputs return an error. NVIDIA tools and libraries are separately licensed and are not bundled.

```powershell
# Run in a Visual Studio x64 developer shell.
./scripts/build-cuda.ps1 -Nvcc 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe'
$env:QS_CUDA_RUNNER = (Resolve-Path ./local/quantaspindle-cuda.exe).Path
go test ./cuda -v
go run ./cmd/quantaspindle -backend cuda -cuda-runner $env:QS_CUDA_RUNNER -workers 1 -batch 16384 -paths 16384 -summary-only
```

There are no supported FPGA boards, vendor emulators, hardware designs, or measured FPGA results. The Backend interface leaves room for future implementations. Read the [FPGA adapter guide](docs/fpga.md) and [configuration template](examples/fpga-v1.json) before adding one. The template is not executable support.

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

The corrected CPU baseline ran in the pinned Go 1.27.1 Bookworm image on an i9-11900H (8 cores, 16 logical CPUs), WSL2 Linux, with 8.18 GB assigned memory and GOMAXPROCS 16. Workload: 1,024 scenarios, one asset, 252 steps, DT 1/252, seed 1, price 100, drift 0.05, volatility 0.2, summary output, batch 128. One warm-up preceded five measured 500 ms benchmark trials. Each scenario is one asset path. Times include library setup and aggregation; compilation and JSON output are excluded.

| Mode | Workers | Median scenarios/s and paths/s | Median ms/op (min to max) |
| --- | --- | --- | --- |
| CPU container | 1 | 80,453 | 12.728 (11.893 to 13.345) |
| CPU container | 2 | 116,317 | 8.804 (8.227 to 10.919) |
| CPU container | 4 | 186,598 | 5.488 (5.090 to 6.067) |
| CUDA container, same 1,024-scenario case | None | Not measured | See matched batch sizes below |
| FPGA performance model | None | Not implemented | Not measured |
| FPGA vendor emulation | None | Not supported | Not run |
| Physical FPGA board | None | No supported board | Not run |

A separate matched CPU/CUDA experiment used one worker, one batch, the same model settings, and batch sizes 256, 4,096, and 16,384. Both backends ran in one local CUDA 12.4.131 and Go 1.27.1 container on the same RTX 3060 Laptop host. One warm-up preceded five single-invocation trials per case. These times include full library execution; CUDA also includes child startup, device setup, allocation, transfers, and synchronization. Compilation and JSON encoding occur outside the timed region. Another GPU process was observed, so these are shared-host measurements.

| Backend | Scenarios and batch | Median scenarios/s and paths/s | Full median ms (min to max) | Kernel median ms |
| --- | --- | --- | --- | --- |
| CPU | 256 | 90,093 | 2.841 (2.691 to 2.929) | Not applicable |
| CUDA | 256 | 1,499 | 170.781 (150.714 to 239.890) | 5.208 |
| CPU | 4,096 | 89,414 | 45.810 (45.039 to 46.957) | Not applicable |
| CUDA | 4,096 | 17,435 | 234.936 (176.510 to 269.905) | 5.507 |
| CPU | 16,384 | 82,752 | 197.988 (189.792 to 201.047) | Not applicable |
| CUDA | 16,384 | 73,797 | 222.015 (195.858 to 240.858) | 10.333 |

This runner did not beat the CPU's median full execution time on these cases. Kernel time alone excludes substantial work. These measurements do not establish performance on larger workloads, dedicated hardware, or a persistent GPU runner. The local image identity and its availability limit are in the validation report.

Measured results, trial counts, memory allocations, configuration, environment, commit identities, and limitations are recorded in [validation](docs/validation.md) and [benchmark reproduction](docs/benchmarks.md). No GPU speedup, calibrated financial accuracy, or production-readiness claim is made. CPU, CUDA, and FPGA results must stay separate. [Go Report Card](https://goreportcard.com/) has retired, so no report or grade badge is available.

## Project records

See the [design review](docs/design-review.md), [competitive comparison and naming limits](docs/competitive-landscape.md), [credits](CREDITS.md), [changelog](CHANGELOG.md), [release checklist](docs/release.md), [contribution guide](CONTRIBUTING.md), [conduct rules](CODE_OF_CONDUCT.md), and [security reporting status](SECURITY.md).

Copyright 2026 QuantaSpindle contributors. Licensed under [GNU GPL version 3](LICENSE). [NOTICE.md](NOTICE.md) records the conceptual reference and independent origin. The project name remains provisional pending fuller registry and trademark checks.
