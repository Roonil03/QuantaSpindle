# Optional CUDA backend

The `cuda` Go package launches a separately compiled native CUDA program. Its
GBM step loop really runs on the selected NVIDIA device; it never falls back to
the CPU. The portable library does not import cgo or unsafe, and ordinary builds
and tests do not require an NVIDIA driver, CUDA toolkit, or vendor Go package.

Current device support is deliberately narrow: one asset, fixed drift and
volatility, no regimes or supplied correlation matrix, and `StorePaths=false`.
Terminal price, simple return, realized volatility and maximum drawdown are
returned for every scenario. Multiasset/regime simulation remains available on
the CPU. Execution impact is a separate CPU overlay; the runner does not execute
it. Unsupported requests return `cuda.ErrUnsupported` before starting a child.

## Build and use

Install an NVIDIA CUDA toolkit and a host C++ compiler supported by that toolkit.
For Windows, run the build inside a Visual Studio Developer PowerShell or
Developer Command Prompt with x64 compiler tools initialized. For example:

```powershell
./scripts/build-cuda.ps1 -Nvcc 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe'
$env:QS_CUDA_RUNNER = (Resolve-Path ./local/quantaspindle-cuda.exe).Path
go test ./cuda -v
```

On Linux with a supported host compiler:

```sh
CUDA_ARCH=sm_86 sh scripts/build-cuda.sh
QS_CUDA_RUNNER="$PWD/local/quantaspindle-cuda" go test ./cuda -v
```

The default architecture `sm_86` targets the tested RTX 3060 Laptop GPU; choose
`-Architecture sm_XX` on PowerShell or `CUDA_ARCH=sm_XX` on Linux for your device.
The toolkit's supported architecture list and host compiler requirements govern
other targets. Use an explicit nvcc path when `CUDA_PATH` points at another
toolkit version. In sandboxed Windows environments, nvcc may fail to reopen
generated temporary sources under an unusual sandbox TEMP path; set TEMP and TMP
to an ordinary writable build directory before invoking the build script.

```go
cfg := quantaspindle.DefaultConfig()
cfg.Workers = 1 // one serialized CUDA session per Run
cfg.BatchSize = 16384
backend := &cuda.Backend{Runner: "/absolute/path/to/quantaspindle-cuda"}
summary, err := quantaspindle.Run(ctx, cfg, backend, nil)
// Handle err before using summary. Run closes its owned session before returning.
// backend.Stats() records timings; measure the whole Run for full latency.
```

The complete import paths are `github.com/Roonil03/QuantaSpindle` and
`github.com/Roonil03/QuantaSpindle/cuda`. Empty `Runner` first checks
`QS_CUDA_RUNNER`, then `QUANTASPINDLE_CUDA_RUNNER`, then searches PATH for
`quantaspindle-cuda`. Set these paths only to a trusted executable. A missing
runner returns `ErrUnavailable`; process/device failures return `ErrExecution`;
malformed output returns `ErrProtocol`. The optional hardware tests skip unless
one of the runner environment variables is set. Once selected, missing hardware
or a failing runner fails the tests explicitly.

## Per-run process and resource lifecycle (v0.1.1)

- `Run` detects the optional `quantaspindle.RunBackend` interface and calls `BeginRun(context.Context, Config) (BackendSession, error)`. A `BackendSession` implements `Backend` and `Close() error`. `Run` owns and closes that session on success and every error path.
- CUDA starts one native child with `--stream` per run. Its batches are serialized over one binary stream, and the child retains its CUDA device state, events, and output buffers between requests. Separate runs own separate sessions; there is no global pool or reuse across runs.
- Direct `cuda.Backend.Batch` calls preserve the v1 one-shot interface: one process handles one batch and exits. Existing custom backends that implement only `Backend` keep their original lifecycle.
- Worker count still controls Go scheduling and bounded waves. Multiple workers in one CUDA run queue on the same session; they do not create concurrent kernels. Start with one worker and benchmark the intended batch and scenario counts.
- The stream greeting is bounded by five seconds, or a shorter configured timeout. An older one-shot-only runner is rejected with a rebuild requirement; rebuild the runner together with the Go adapter.
- `Backend.Timeout` bounds each request/response exchange, defaulting to five minutes. Caller cancellation terminates the child. Requests waiting for the session and active work remain subject to cancellation. The whole-run CLI timeout also bounds the run.
- Each response has its exact expected size; retained stderr is bounded to 16 KiB. A failed or canceled batch returns no partial scenarios. `Run` may already have delivered an earlier successful prefix. Failed runs return an error and cannot report a successful final summary.

## Automatic selection and prepared CUDA (v0.1.2)

The CLI defaults to `-backend auto`. It chooses one backend for the entire run. Use `-backend cpu` or `-backend cuda` to request a specific generator. On a CPU selection, the full CPU model is available, including correlations, regimes and stored paths. On a CUDA selection, the requested financial parameters pass through to the FP64 kernel. Which backend executed the run matters for replay.

The cold policy considers CUDA at 262,144 or more scenarios with at least 252 steps. The prepared policy considers it at 4,096 or more scenarios with the same step minimum. These thresholds are heuristics for this implementation. The measurements use 252 steps on one laptop. With `-backend auto`, the CLI uses one worker and batches up to 16,384 for a CUDA selection unless the configuration file or flags supply those settings. Automatic-selection metadata contains the original `config`, the effective `execution_config`, the selected `backend`, and the `selection` reason. Explicit `-backend cuda` preserves the configured worker and batch settings; its metadata contains `config` and `backend`, without `execution_config` or `selection`.

For an eligible CUDA job, selection executes its first batch and keeps those results for delivery. A missing runner or classified device initialization failure can therefore select CPU before the sink sees a scenario. Cancellation, malformed results and numerical failures propagate. A failure on a later batch returns the successfully delivered prefix. One run never mixes CPU and GPU random streams.

For a single library call, use `hybrid.Run(ctx, cfg, hybrid.Options{}, sink)`. It returns the run summary, a routing decision, and an error. With the zero options, it uses the normal runner lookup and respects the worker and batch sizes in `cfg`. The `hybrid` package is at `github.com/Roonil03/QuantaSpindle/hybrid`.

A service can initialize its CUDA device before accepting requests. Keep the prepared object for the service lifetime. Pass it to each request that may use the GPU and close it at shutdown:

```go
gpu := &cuda.Backend{}
prepared, err := gpu.Prepare(serviceCtx)
if err != nil && !errors.Is(err, cuda.ErrUnavailable) {
    return err
}
if prepared != nil {
    defer prepared.Close()
}
options := hybrid.Options{Prepared: prepared, CUDAWorkers: 1, CUDABatchSize: 16384}
for _, cfg := range requests {
    summary, choice, err := hybrid.Run(serviceCtx, cfg, options, nil)
    if err != nil {
        return err
    }
    fmt.Println(choice.Backend, summary.Scenarios)
}
```

`Prepare` opens the runner and executes a one-scenario, one-step initialization probe. The probe contributes to preparation time. User batch and scenario counters start with real requests. A prepared process serves one run at a time, retains allocations up to its largest batch and releases them on close. A caller waiting to use it can cancel. Pass `prepared` directly to `quantaspindle.Run` for explicit CUDA execution. The [startup benchmark](cuda-startup.md) measures preparation, the first real request, later requests and shutdown.

The context passed to `Prepare` governs the service lifetime. Use a separate request context for each `Run`. An exchange failure invalidates the runner. A later explicit CUDA run can reopen it. `Warm()` is a nonblocking snapshot: it is false during recovery and after closure, and becomes true after a successful batch on the current process. When the distinction affects routing, `BeginWarmRun` checks readiness while acquiring the exclusive run session and returns a nil session if the process has cooled. The lease's `WasWarm()` records its state at acquisition. Hybrid uses this distinction to keep a queued small job from silently paying to restart CUDA.

## Mathematics, memory and replay

The runner evaluates the same exact conditional FP64 GBM update, Welford
log-return sample volatility, and peak-to-trough fractional drawdown definitions
as the CPU. It includes the initial price in the drawdown peak, returns zero
sample volatility for one interval, and rejects nonfinite or nonpositive
simulated prices. No fast-math flag is used; fused multiply-add contraction is
disabled with `--fmad=false`. These choices do not imply cross-device bitwise
equivalence of transcendental functions.

One GPU thread owns one scenario. All intermediate prices, running volatility
state, peak and drawdown remain on device; a path is never copied to the host.
Output is four FP64 values (32 bytes) per scenario. Session output buffers are reused between batches. Device output storage is
O(batch size), independent of step count, and live scalar state is per thread.
The host holds one compact native result vector plus the Go response and
independently owned public Scenario slices. Runner limits are 1,000,000
scenarios, 1,000,000 steps and 10,000,000,000 updates per request; core Config/Run
limits may be smaller. Both peers check dimensions and numeric inputs; the
native host also checks multiplication overflow and complete request frames before
allocating or launching work. The one-shot interface additionally rejects trailing
input; the stream interface expects another complete request or clean EOF.

The generator is cuRAND device `Philox4_32_10`, initialized as
`curand_init(config.Seed, scenario.ID, 0, &state)`. `curand_normal_double` supplies
each interval's innovation. The global scenario ID is the cuRAND sequence ID;
worker count and batch partition do not change a scenario. `Scenario.Seed`
retains the library's `ScenarioSeed(config.Seed, ID)` provenance tag; on this
backend the actual cuRAND inputs are the original config seed and ID, recorded
by `Metadata.Generator`. This is a different generator and mapping from the
CPU's SplitMix64/Box-Muller generator.

Replay is promised on the same runner build, device and CUDA toolchain with the
same parameters. Tests verify repeated batches and shifted batch partitions.
CPU/GPU paths are not expected to match, and tests compare analytical GBM
moments and CPU/GPU moments with predeclared finite-sample tolerances instead.
The legacy cuRAND device dependency is isolated in `runner.cu` because NVIDIA
has announced its future deprecation. A replacement requires a new documented
replay version; it must not silently change that contract.

## Timing and validation

The historical v0.1.0 runner passed the conditional suite in a GPU-enabled Ubuntu 22.04.4 container with nvcc 12.4.131, GCC 11.4.0, Go 1.27.1, runtime API 12040, and driver API 13050 on the same RTX 3060 Laptop GPU. Native Windows used CUDA 13.2.51 and MSVC 14.42. These are separate tested environments, not a claim of support for every toolkit in between. Container measurements and the local-image availability limit are recorded in [validation](validation.md) and [reproduction](benchmarks.md).

- `LastMetadata()` identifies the GPU, CUDA runtime/driver API versions, generator, and binary protocol of the last successful batch.
- `KernelTime` uses CUDA events around the kernel; it excludes setup, transfers, and Go work.
- `HostTime` records native service from the completed request frame through device output transfer, in both modes. It excludes process startup, request-read wait and Go sink time. CUDA initialization is included only when it occurs during that batch's service.
- `Stats()` sums successful batch timings. `WallTime` is the one-shot process invocation or streaming request/response exchange; `SessionLifetime` separately sums complete streaming session lifetimes, including startup, idle/sink time, teardown and failed sessions. Inspect session totals after `Run` returns. Concurrent durations are summed, so they are not necessarily elapsed run time.
- Time the caller's complete `Run` for an end-to-end comparison: Go setup, process startup, device initialization, batches, transfers, decoding, aggregation, and close all count. Compilation and output JSON encoding are separate.

An ordinary CUDA backend pays startup once per run. A prepared backend pays it once per process, including any replacement after a failure. Read `SessionLifetime` after the prepared service closes to include its complete lifetime. The [v0.1.1 measurements](cuda-performance.md) compare per-run and one-shot execution. The [startup measurements](cuda-startup.md) compare cold requests, prepared requests and complete service lifetimes against CPU.

Hardware tests cover actual device execution; same-backend repeatability and
batch partition; analytical GBM terminal mean and variance; CPU/GPU distribution
agreement; zero volatility and decreasing-price drawdown; one interval;
numerical overflow; and native truncated/trailing/invalid requests. Ordinary
unit tests cover unavailable selection, capability rejection, invalid ranges,
cancellation, malformed response sizes/numbers and bounded diagnostics/output.

The runner source is covered by this repository's GPL-3.0 license. NVIDIA's CUDA
toolkit, cuRAND and runtime are separate proprietary components under NVIDIA's
terms. Build locally with the toolkit; this repository does not vendor or
redistribute the toolkit libraries. The build currently links the CUDA runtime
statically, so distributing a compiled runner requires separately checking the
applicable NVIDIA redistribution terms and license obligations.

## Binary protocol v1 and stream framing

All integers and IEEE-754 binary64 values are little-endian. The one-shot interface accepts exactly
72 bytes followed by EOF. Fields are: magic `QSPNCU01` at offset 0 (8 bytes),
version uint32 at 8, steps uint32 at 12, count uint64 at 16, starting scenario ID
uint64 at 24, config seed uint64 at 32, initial price double at 40, mu double at
48, sigma double at 56 and DT double at 64.

A successful stdout begins with exactly 176 header bytes: magic `QSPNCR01` at
0, version uint32 at 8, zero status uint32 at 12, count uint64 at 16, kernel
milliseconds double at 24, runner host milliseconds double at 32, CUDA runtime
version uint32 at 40, CUDA driver API version uint32 at 44, and zero-padded GPU
name bytes at 48 through 175. Each of the following count records has four
doubles: terminal price, simple return, realized volatility and maximum
drawdown. No record includes the ID; record i belongs to start+i. There are no
trailing bytes. Validation or device errors produce a nonzero process exit,
diagnostics on stderr, and no successful response for the failed request. The
one-shot Go caller checks process exit status before parsing output. A stream
may already have delivered earlier complete responses; the session checks each
response and reports process/close failures.

In `--stream` mode the child first writes the eight-byte greeting `QSPNCS01`, before CUDA initialization. It then accepts repeated 72-byte v1 requests and writes one complete v1 response per successful request. Clean EOF between requests ends the session; truncated requests fail. Request/response payloads and the Philox scenario-ID mapping remain v1. The greeting identifies the streaming capability and prevents silently using a stale runner.
