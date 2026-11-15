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
cfg.Workers = 1 // avoid concurrent short-lived GPU processes initially
cfg.BatchSize = 16384
backend := &cuda.Backend{Runner: "/absolute/path/to/quantaspindle-cuda"}
summary, err := quantaspindle.Run(ctx, cfg, backend, nil)
// Handle err before using summary. backend.Stats() records successful batches.
```

The complete import paths are `github.com/Roonil03/QuantaSpindle` and
`github.com/Roonil03/QuantaSpindle/cuda`. Empty `Runner` first checks
`QS_CUDA_RUNNER`, then `QUANTASPINDLE_CUDA_RUNNER`, then searches PATH for
`quantaspindle-cuda`. Set these paths only to a trusted executable. A missing
runner returns `ErrUnavailable`; process/device failures return `ErrExecution`;
malformed output returns `ErrProtocol`. The optional hardware tests skip unless
one of the runner environment variables is set. Once selected, missing hardware
or a failing runner fails the tests explicitly.

One process handles one batch and exits. The adapter uses `exec.CommandContext`,
enforces the caller's cancellation and a default five-minute batch timeout,
and bounds stdout to exactly the expected response size and retained stderr to
16 KiB. `Backend.Timeout` can set a different positive timeout. A failed or
canceled batch returns no partial scenarios. `Run` may already have delivered an
earlier successful prefix, as described in the core API. Each batch validates
configuration before resolving or launching the runner.

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
Output is four FP64 values (32 bytes) per scenario. Device output storage is
O(batch size), independent of step count, and live scalar state is per thread.
The host holds one compact native result vector plus the Go response and
independently owned public Scenario slices. Runner limits are 1,000,000
scenarios, 1,000,000 steps and 10,000,000,000 updates per process; core Config/Run
limits may be smaller. Both peers check dimensions and numeric inputs; the
native host also checks multiplication overflow, complete stdin and trailing
bytes before allocating or launching work.

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

The corrected runner also passed the conditional suite in a GPU-enabled Ubuntu 22.04.4 container with nvcc 12.4.131, GCC 11.4.0, Go 1.27.1, runtime API 12040, and driver API 13050 on the same RTX 3060 Laptop GPU. Native Windows used CUDA 13.2.51 and MSVC 14.42. These are separate tested environments, not a claim of support for every toolkit in between. Container measurements and the local-image availability limit are recorded in [validation](validation.md) and [reproduction](benchmarks.md).

`LastMetadata()` identifies the actual GPU, CUDA runtime and driver API versions,
generator and binary protocol. `KernelTime` comes from CUDA events around kernel
execution. `HostTime` is runner wall time from request reading through device
output transfer, excluding final response serialization and teardown.
`WallTime` is measured by Go around process execution, including startup,
allocation, transfers, serialization and teardown. It excludes Go request
encoding and response decoding. `Stats()` sums these times across successful
batches and returns the last device metadata. Concurrent batch totals are sums,
not elapsed run time. Measure the caller's whole `Run` separately for complete
end-to-end comparison. Compilation is always separate.

The process boundary, context initialization, allocation and output transfer
can dominate small batches. No GPU speedup is implied by successful execution.
Use identical scenario counts, steps and output requirements and disclose CPU
workers, GPU batch size, precision, warmup policy and all timing boundaries for
comparisons. This version favors an explicit maintained process boundary; it
does not maintain a persistent CUDA context or pool.

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

## Binary protocol v1

All integers and IEEE-754 binary64 values are little-endian. Stdin is exactly
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
diagnostics on stderr, and no successful response. The Go caller checks exit
status before parsing any response.
