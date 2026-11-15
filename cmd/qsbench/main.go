// Command qsbench measures complete fixed-seed Runs after an unreported warm-up.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	backendNames := flag.String("backend", "cpu", "comma-separated cpu, cuda-batch (one process per batch), or cuda")
	runner := flag.String("cuda-runner", "", "trusted CUDA runner path")
	batchRunner := flag.String("cuda-batch-runner", "", "optional separate runner for cuda-batch, e.g. the release baseline")
	batches := flag.String("batches", "256,4096,16384", "comma-separated batch sizes; also scenario counts when paths is empty")
	paths := flag.String("paths", "", "comma-separated total scenario counts; permits multi-batch Runs")
	workers := flag.String("workers", "1", "comma-separated CPU worker counts; CUDA uses one worker")
	cpuBatch := flag.Int("cpu-batch", 0, "optional CPU batch size override for parallel utilization")
	steps := flag.Int("steps", 252, "steps per scenario")
	trials := flag.Int("trials", 5, "measured trials after one warm-up per batch size")
	flag.Parse()
	if flag.NArg() != 0 || *trials < 1 || *trials > 100 {
		return fmt.Errorf("use no positional arguments and trials in [1,100]")
	}
	batchSizes, err := positiveInts(*batches)
	if err != nil {
		return err
	}
	workerCounts, err := positiveInts(*workers)
	if err != nil {
		return err
	}
	var pathCounts []int
	if *paths != "" {
		pathCounts, err = positiveInts(*paths)
		if err != nil {
			return err
		}
	}
	if *cpuBatch < 0 {
		return fmt.Errorf("cpu-batch must be nonnegative")
	}
	type benchmarkCase struct {
		mode   string
		config qs.Config
	}
	var cases []benchmarkCase
	for _, batch := range batchSizes {
		counts := pathCounts
		if len(counts) == 0 {
			counts = []int{batch}
		}
		for _, count := range counts {
			for _, mode := range strings.Split(*backendNames, ",") {
				if mode != "cpu" && mode != "cuda" && mode != "cuda-batch" {
					return fmt.Errorf("unknown backend %q", mode)
				}
				ws := []int{1}
				if mode == "cpu" {
					ws = workerCounts
				}
				for _, worker := range ws {
					cfg := qs.DefaultConfig()
					cfg.Paths, cfg.BatchSize, cfg.Workers, cfg.Steps = count, batch, worker, *steps
					if mode == "cpu" && *cpuBatch > 0 {
						cfg.BatchSize = *cpuBatch
					}
					if err := cfg.Validate(); err != nil {
						return err
					}
					cases = append(cases, benchmarkCase{mode, cfg})
				}
			}
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	// Warm each case once. Reverse the measured case order on alternate trials,
	// so a backend is not always measured first or last on a shared laptop.
	for trial := -1; trial < *trials; trial++ {
		for position := range cases {
			index := position
			if trial >= 0 && trial%2 == 1 {
				index = len(cases) - 1 - position
			}
			item := cases[index]
			cfg := item.config
			var backend qs.Backend
			var gpu *cuda.Backend
			switch item.mode {
			case "cpu":
				backend = qs.CPU{}
			case "cuda", "cuda-batch":
				gpu = &cuda.Backend{Runner: *runner}
				backend = gpu
				if item.mode == "cuda-batch" {
					if *batchRunner != "" {
						gpu.Runner = *batchRunner
					}
					backend = batchOnly{gpu}
				}
			}
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			started := time.Now()
			summary, err := qs.Run(context.Background(), cfg, backend, nil)
			elapsed := time.Since(started)
			runtime.ReadMemStats(&after)
			if err != nil {
				return err
			}
			if trial < 0 {
				continue
			}
			if elapsed <= 0 {
				return fmt.Errorf("elapsed time below clock resolution")
			}
			record := map[string]any{"backend": backend.Name(), "mode": item.mode, "trial": trial + 1, "order": position + 1, "config": cfg, "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "gomaxprocs": runtime.GOMAXPROCS(0), "elapsed_ns": elapsed.Nanoseconds(), "scenarios_per_second": float64(cfg.Paths) / elapsed.Seconds(), "asset_paths_per_second": float64(cfg.Paths*len(cfg.Assets)) / elapsed.Seconds(), "go_allocated_bytes": after.TotalAlloc - before.TotalAlloc, "go_allocations": after.Mallocs - before.Mallocs, "summary": summary}
			if gpu != nil {
				record["cuda"] = gpu.Stats()
			}
			if err := encoder.Encode(record); err != nil {
				return err
			}
		}
	}
	return nil
}

// Deliberately do not embed Backend: optional Run lifecycle methods must remain
// hidden so this measures the public one-shot Batch execution path.
type batchOnly struct{ gpu *cuda.Backend }

func (b batchOnly) Name() string { return b.gpu.Name() }
func (b batchOnly) Batch(ctx context.Context, cfg qs.Config, start uint64, count int) ([]qs.Scenario, error) {
	return b.gpu.Batch(ctx, cfg, start, count)
}

func positiveInts(value string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(value, ",") {
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("expected comma-separated positive integers, got %q", value)
		}
		out = append(out, n)
	}
	return out, nil
}
