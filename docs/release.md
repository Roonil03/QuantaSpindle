# Release procedure

The repository is public. v0.1.0 is the initial immutable release; v0.1.1 adds per-run CUDA reuse; v0.1.2 adds automatic selection and application-owned prepared CUDA. Never move, reuse, or overwrite a published tag. A GitHub release-note edit does not change tagged Go module contents.

## Prepare a reviewable version

- Confirm the module path is `github.com/Roonil03/QuantaSpindle` in go.mod, imports, examples, and README.
- Run formatting, vet, CPU tests, supported race checks, and the conditional CUDA tests in [CUDA setup](cuda.md). Rebuild the native runner from the same source as the Go adapter, including the streaming capability and device-initialization failure classification.
- Record exact source identity, machine, toolchains, commands, outcomes, and any failed or unavailable checks in [validation](validation.md) and the relevant performance report.
- Compare repeated matched workloads before claiming improvement. Record complete run latency separately from kernel/request time, and keep historical v0.1.0 measurements identifiable.
- Review the final changes and resolve correctness findings before publication. Update the changelog with actual completed behavior and validation; do not convert a pending check into a pass.
- Keep README installation examples and versioned API links aligned with the version actually being published. Verify them after the tag is available.
- Use the completed `v0.1.2` section in [CHANGELOG.md](../CHANGELOG.md#v012---2026-11-15) as the release-note source. Preserve its hardware, source links, timing boundaries, one-off versus ready distinction, six-request setup cost and measurement limits. Extract that section to a temporary file for the GitHub release body; no separate tracked release-note file is needed.

Inspect publication contents:

```sh
git status --short
git ls-files
git check-ignore Dockerfile.benchmark.local compose.benchmark.local.yml local/results.txt benchmarks/generated/results.txt
go list -m all
```

The Go module has no external Go dependencies. Preserve the canonical GPL version 3 LICENSE, attribution, and notices. NVIDIA tooling is separately licensed and is not bundled. Review redistribution terms separately before distributing compiled native runners.

## Publish and verify

After checks and review pass, commit the release changes, create the new semantic version tag on that commit, push it, and create a matching GitHub release. Never tag incomplete work to fill a checklist. For automatic selection and prepared CUDA use `v0.1.2`. Keep `v0.1.0` at `3ffad625663d2ec24a54de6aabc41252491c9210` and `v0.1.1` at `ed503e730f6740315d32add48c087d60414479d1`.

Verify without credentials from a clean consumer using the public Go proxy and an isolated module cache. Clear private-module bypass settings for this public module:

```powershell
$env:GOPROXY = 'https://proxy.golang.org'
$env:GOPRIVATE = ''
$env:GONOSUMDB = ''
$env:GOMODCACHE = Join-Path (Get-Location) 'local/release-consumer-cache'
go list -m github.com/Roonil03/QuantaSpindle@v0.1.2
go mod download -json github.com/Roonil03/QuantaSpindle@v0.1.2
```

Use a fresh shell or restore these environment settings after verification. Follow [Go's publication process](https://go.dev/doc/modules/publishing). Check the versioned [proxy metadata](https://proxy.golang.org/github.com/!roonil03/!quanta!spindle/@v/v0.1.2.info) and [pkg.go.dev documentation](https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.2), recording origin commit and checksum. Go proxy paths escape uppercase letters with `!` followed by lowercase. pkg.go.dev indexes fetchable modules; it is not a separate upload service. An indexing request alone does not establish that documentation is available: verify the rendered versioned page. Updated README links and installation examples are publication targets, not evidence that these checks passed.

Attach the three original startup-experiment files to the v0.1.2 GitHub release: `baseline.jsonl` (20 tagged records), `measure.jsonl` (340 measured-implementation records), and `alternative.jsonl` (20 alternative-runner records). Verify their SHA-256 hashes against [the startup report](cuda-startup.md#hardware-validation-and-source) before uploading, then verify unauthenticated public downloads and the same hashes afterward. Preserve these original logs and their recorded source/build identities; the release-documentation commit does not replace the measured source identity. Generated logs need not be committed. The separate 250-record v0.1.1 experiment remains with that historical release.

If updating the original v0.1.0 GitHub release notes, add a clearly dated **v0.1.1 follow-up** section linking the new release, source, and measurements. State that the optimized behavior/results require v0.1.1. Retain the original v0.1.0 results and limitations; never attribute follow-up code or speed to that tag.

Verify CI, public documentation links, and the private advisory route in SECURITY.md. Go Report Card has retired; do not add a grade badge. Record pre-tag checks in validation.md and the verified public commit, module checksum and package links in the published release notes.
