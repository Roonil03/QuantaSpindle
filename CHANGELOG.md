# Changelog

## Unreleased

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

### Known limitations

- Public module fetching and pkg.go.dev indexing are pending repository visibility and release validation.
- No private vulnerability reporting route has been verified.
- CUDA supports one asset with constant GBM parameters, without regimes, correlation or stored paths.
- Market calibration is not provided.
- No FPGA board, vendor emulation, or bitstream has been validated.
