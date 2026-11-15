# Credits

QuantaSpindle is independently written. These entries identify material references and tools, not endorsements. Research was recorded on 2026-11-15. No external Go module dependencies, market datasets, or FPGA bitstreams are bundled.

## Dependencies and libraries

- Go standard library, https://pkg.go.dev/std, Go 1.27.1. Provides concurrency, local random generation, math, JSON, process control, and testing. Go is BSD-3-Clause licensed. Used as a toolchain/runtime dependency; its source was not copied into this repository.
- NVIDIA cuRAND, https://docs.nvidia.com/cuda/archive/13.2.0/curand/device-api-overview.html, CUDA 13.2. Provides GPU random streams in the optional native runner. NVIDIA's toolkit license applies. Vendor source and libraries are not bundled. Runner calls use the documented API.

## Toolchains

- Go, https://go.dev/, 1.27.1 for recorded local checks and the pinned CPU container. Standard supported compiler and runtime, without modifications.
- NVIDIA CUDA Toolkit, https://developer.nvidia.com/cuda-toolkit, installed nvcc 13.2.51. Optional device compilation and runtime. Governed by NVIDIA's proprietary toolkit terms at https://docs.nvidia.com/cuda/eula/index.html. No vendor binaries are redistributed.
- Microsoft Visual C++, https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line, installed Visual Studio 2022 tools for Windows CUDA host compilation. Microsoft license applies; not bundled.
- Docker Engine, https://docs.docker.com/engine/, local 29.9.3. Runs pinned Go CPU checks. Engine components and Docker Desktop have distinct licenses; no Docker software or base image is included in the repository.
- GitHub Actions checkout and setup-go, https://github.com/actions/checkout and https://github.com/actions/setup-go, v4 and v5. CI workflow actions under MIT licenses. Referenced by workflow, not vendored.
- FPGA toolchains: none used. See docs/fpga.md for future adapter requirements.

## Papers, standards, and technical documentation

- Columbia Monte Carlo notes, https://www.columbia.edu/~mh2078/MonteCarlo/MCS_Generate_RVars.pdf. Consulted for normal sampling and GBM moment checks. No text, code, or figures reused.
- Go rand/v2 design discussion, https://go.dev/blog/randv2. Consulted for local seeded random generators. No code copied.
- Gonum matrix documentation, https://pkg.go.dev/gonum.org/v1/gonum/mat. Consulted to compare numerical infrastructure and Cholesky factorization. Gonum is BSD-3-Clause licensed. No dependency or source copied.
- markovchain authors' vignette, https://cran.r-project.org/web/packages/markovchain/vignettes/an_introduction_to_markovchain_package.pdf. Consulted for finite-state transition conventions. No code or documentation copied.
- Almgren and Chriss, Optimal Execution of Portfolio Transactions, https://static1.squarespace.com/static/5df018ec4c9d61288094246c/t/5e39db229a16b072d97fce9b/1580849954431/Optimal%2BExecution%2Bof%2BPortfolio%2BTransaction%2B_%2BAlmgren%26Chriss%2B1999.pdf. Consulted for separation of execution impact from path dynamics. The project's simple impact estimate is not the paper's complete model. No code, data, figures, or text copied.
- Basel market-risk framework, https://www.bis.org/committees/bcbs/basel-framework/standard/mar. Consulted to distinguish empirical risk summaries from regulatory calculations. No material copied.
- Go module publication guide, https://go.dev/doc/modules/publishing. Informs the release checklist. No text copied.

## Data and benchmarks

All test inputs are synthetic. There is no bundled market data, calibration data, or third-party benchmark dataset. Benchmark source generates its own fixed-seed workload. Local generated results stay in ignored directories; measured summaries identify their environment and limits.

## Conceptual project references

- stock-simulations, https://github.com/e-wrobel/stock-simulations and https://pkg.go.dev/github.com/e-wrobel/stock-simulations. Conceptual reference through accessible indexed documentation only. Repository access returned 404. Indexed GPL-3.0 metadata was observed. No source code, tests, data, assets, configuration, documentation, or structure were copied or modified. Its copyright notices were not inherited.
- QuantLib, https://www.quantlib.org/. Official project description used for scope comparison. Modified BSD license. No code, data, assets, or documentation copied.
- MSwM, https://cran.r-project.org/package=MSwM. Scope comparison for regime estimation, which QuantaSpindle does not provide. No dependency or material reused.

See docs/competitive-landscape.md for naming-search sources and limits. No trademark clearance is claimed.
