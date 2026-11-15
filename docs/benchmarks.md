# Reproducing measurements

Use identical models, seeds, step counts, scenario counts, batch sizes, output modes, and trial counts for comparisons. In this project a scenario contains one path per asset. Scenarios/s and asset paths/s are equal only for a single asset. Throughput, per-batch latency, and numerical validity are separate quantities.

The pinned CPU environment is `golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66`. It was pulled on 2026-11-15. The module has no external dependencies. Use this local-only Dockerfile at the exact ignored path `Dockerfile.benchmark.local`:

```dockerfile
FROM golang:1.27.1-bookworm@sha256:8d48e12ec56735e9358640898b9d9b9fcca110612ed8a5567438c0a1baa24e66
WORKDIR /work
CMD ["sh", "scripts/benchmark-cpu.sh"]
```

From PowerShell, build and run it with the repository mounted:

```powershell
docker build -f Dockerfile.benchmark.local -t quantaspindle-benchmark:local .
docker run --rm --mount "type=bind,source=$((Get-Location).Path),target=/work" quantaspindle-benchmark:local
```

The Dockerfile is excluded from Git; create it locally using the content above. Container output goes to ignored directories. Preserve raw results with your experiment rather than committing generated logs. For the validation run, see docs/validation.md. Record `git rev-parse HEAD`, `docker image inspect`, the engine version, CPU model and assigned cores, assigned memory, OS/kernel, Go version, model settings, seed, workers, and batch size with any published number. Container resource assignments can differ from physical host resources.

Run one warm-up followed by five measured trials. Report the median and minimum/maximum of elapsed time, paths/s, scenarios/s, B/op, and allocs/op. The benchmarks include workers and batch-size cases. Profiling is a separate run; profiling overhead must not enter the published throughput sample. The saved CPU profile is the starting point for tuning, not evidence of a speedup.

For profile-guided optimization, use the representative profile with `go build -pgo=profiles/cpu.prof` and compare with `go build -pgo=off`. No PGO improvement is claimed without that comparison. Generated profiles are ignored. No custom compiler, runtime, unsafe code, assembly, or unmeasured optimizer changes are included.

CUDA measurements require a working NVIDIA GPU, driver, toolkit, and optional supported container runtime. See docs/cuda.md. Report kernel event time separately from full Go invocation time, including process startup, device setup, transfer, allocation, and synchronization. A native Windows run is not a GPU container run. Mark unavailable container checks explicitly. CPU/CUDA tests compare distributions because their random generators differ; no bitwise cross-backend equality is promised.

FPGA performance model, RTL simulation, vendor emulation, and physical board results are all unavailable. Keep them separate from CPU and CUDA measurements when adapters are added.
