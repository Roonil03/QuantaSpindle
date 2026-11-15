#!/bin/sh
set -eu
# Run inside the pinned Go container from docs/benchmarks.md.
mkdir -p benchmarks/generated profiles
{
  go version
  go env GOOS GOARCH CGO_ENABLED
  uname -a
  cat /proc/cpuinfo
  cat /proc/meminfo
} > benchmarks/generated/environment.txt
go test ./...
go test -race ./...
go test -run '^$' -bench Benchmark -benchtime 1x -benchmem ./...
go test -run '^$' -bench Benchmark -benchtime 500ms -benchmem -count 5 ./... > benchmarks/generated/cpu.txt
go test -run '^$' -bench 'BenchmarkRun/workers=1/batch=128$' -benchtime 2s -cpuprofile profiles/cpu.prof -memprofile profiles/mem.prof .
go tool pprof -top profiles/cpu.prof > benchmarks/generated/cpu-profile.txt
