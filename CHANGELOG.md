# Changelog

## Unreleased

Target version: v0.1.0. A version is released only after its checks and review pass. The owner will decide when to make the repository public.

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

### Known limitations

- Public module fetching and pkg.go.dev indexing are pending repository visibility and release validation.
- No private vulnerability reporting route has been verified.
- No FPGA board, vendor emulation, or bitstream has been validated.
