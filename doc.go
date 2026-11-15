// Package quantaspindle generates reproducible financial scenarios, not forecasts.
// Prices follow conditional geometric Brownian motion in years. Mu is annual
// arithmetic drift and Sigma is annualized volatility; parameters are supplied by
// callers and are neither estimated nor calibrated. Correlation applies to
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
package quantaspindle
