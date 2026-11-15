# Releasing v0.1.0

Keep CHANGELOG.md under Unreleased until checks pass and the release is made. Never move or reuse an existing published tag. This repository remains private until its owner changes its visibility.

Confirm the module path is `github.com/Roonil03/QuantaSpindle` in go.mod, package examples, README, and CLI imports. Run formatting, vet, CPU tests, race checks, benchmarks, and conditional CUDA tests described in README and docs/cuda.md. Record the exact commit, versions, machine, commands, and failures in docs/validation.md. Review changes through separate pull requests. Do not merge unresolved correctness findings.

Inspect publication contents:

```sh
git status --short
git ls-files
git check-ignore Dockerfile.benchmark.local compose.benchmark.local.yml local/results.txt benchmarks/generated/results.txt
go list -m all
```

No external Go module dependencies are required. The existing LICENSE contains the canonical GNU GPL version 3 text. Review any future dependency or native runtime distribution before release. NVIDIA tooling is separately licensed and is not bundled.

After checks and review pass, update the changelog with the actual release date, commit that change, create `v0.1.0`, push it, and create a matching GitHub release from the completed changelog entries. Do not tag an incomplete release to fill a checklist.

For private use, configure Git credentials and `GOPRIVATE=github.com/Roonil03/QuantaSpindle`. Do not request public indexing while private. After the owner makes the repository public, follow https://go.dev/doc/modules/publishing and verify from a clean consumer:

```sh
go list -m github.com/Roonil03/QuantaSpindle@v0.1.0
go mod download -json github.com/Roonil03/QuantaSpindle@v0.1.0
```

Then request `https://proxy.golang.org/github.com/!roonil03/!quanta!spindle/@v/v0.1.0.info` and check `https://pkg.go.dev/github.com/Roonil03/QuantaSpindle@v0.1.0`. Go module proxy paths escape each uppercase letter with `!` followed by lowercase. pkg.go.dev indexes fetchable modules; it is not a separate upload service. Add a documentation badge only after its page is confirmed. Verify Go Report Card and CI before adding their status badges.
