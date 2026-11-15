// Package quantaspindle generates reproducible financial scenarios, not forecasts.
// Prices follow conditional geometric Brownian motion in years. Mu is annual
// arithmetic drift and Sigma is annualized volatility; parameters are supplied by
// callers and are neither estimated nor calibrated. CPU execution supports
// single and correlated multiple assets, with optional discrete Markov regimes.
// Correlation applies to
// Brownian innovations, not directly to terminal prices.
//
// A regime governs the current interval and transitions after that interval.
// Transition probabilities are per step: changing DT changes their interpretation.
// Realized volatility is the sample standard deviation of step log returns,
// annualized by sqrt(1/DT), and is zero for a single step. Maximum drawdown is the
// largest peak-relative fall on the sampled grid, including the initial price.
// It does not observe intrastep extrema.
//
// CPU replay uses independent SplitMix64-v1 streams with Box-Muller normals.
// Scheduling, worker count and batch boundaries do not affect a scenario.
// Bitwise replay is scoped to this implementation, backend and Go toolchain;
// different backends may use different generators and floating-point functions.
// Run invokes its sink serially in ID order and retains only a bounded wave of
// batches. A failed run returns its delivered-prefix summary and an error; it
// never counts undelivered scenarios. Sinks own delivered slices and may retain
// them, in which case storage is the caller's responsibility.
//
// Use DefaultConfig and Run(ctx, cfg, CPU{}, sink) for portable CPU execution.
// Full paths are opt-in; exact empirical VaR/expected shortfall require callers
// to retain observations. SquareRootImpact is a separate execution overlay and
// does not automatically alter simulated paths.
//
// The optional cuda package uses a separately compiled native FP64 runner for
// single-asset, constant-parameter GBM without regimes, supplied correlation or
// stored paths. It requires actual NVIDIA hardware and never falls back to CPU.
// From v0.1.1, Run opens and closes an owned BackendSession when a backend
// implements RunBackend, allowing CUDA to reuse one process and device resources
// across batches. Batch-only custom backends keep their original lifecycle.
// Ordinary CPU builds require neither CUDA nor cgo.
//
// Performance is workload and environment dependent. Measure complete Run
// latency including session startup and close; kernel time alone is not an
// end-to-end comparison. See the repository documentation for tested workloads,
// measured results, reproduction commands, and backend replay limits:
// https://github.com/Roonil03/QuantaSpindle/blob/v0.1.1/README.md.
package quantaspindle
