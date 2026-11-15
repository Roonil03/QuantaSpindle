# Contributing

Use Go 1.27.1 or a later supported Go release. The CPU package needs only the Go standard library. Run these commands from the repository root:

```sh
gofmt -w .
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -bench . -benchmem -count 5 ./...
```

The race check needs cgo and a supported C compiler. It checks Go concurrency, not GPU device memory. Read [CUDA setup](docs/cuda.md) for the optional NVIDIA toolkit, driver, compiler, and hardware requirements. A CPU contribution does not require CUDA.

Open a pull request against main. Describe the input that changes behavior and the result after the change. Include the commands you ran and their results. For a failed check, include the command, tool versions, and a small synthetic example. Never attach credentials or private data.

For model changes, state the formula, units, assumptions, and analytical or statistical checks. Set statistical tolerances before measuring the result. For performance changes, include the same workload before and after, repeated trials, machine and toolchain details, and full execution time including transfers. Report correctness and speed separately. Keep the immutable v0.1.0 baselines labeled with their source and record follow-up optimization experiments separately. CUDA lifecycle changes must exercise cancellation, failure cleanup, malformed/truncated streams, stale runner rejection, and repeated multibatch execution, in addition to analytical device checks. See [performance reproduction](docs/cuda-performance.md) for the follow-up workload.

Keep required tests and benchmark source in Git. Keep generated benchmark output, local container files, profiles, credentials, private market data, and vendor build output out of Git. Inspect `git status --short` before committing. Local Docker files have explicit entries in .gitignore.

Contributions must be compatible with GNU GPL version 3. Record any new dependency, dataset, or material technical source in CREDITS.md. Check its license before adding it. Do not copy the conceptual reference implementation.
