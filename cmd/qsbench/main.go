// Command qsbench measures complete fixed-seed batches after an unreported warm-up.
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
	backendName := flag.String("backend", "cpu", "cpu or cuda")
	runner := flag.String("cuda-runner", "", "trusted CUDA runner path")
	batches := flag.String("batches", "256,4096,16384", "comma-separated scenario counts, one batch each")
	trials := flag.Int("trials", 5, "measured trials after one warm-up per batch size")
	flag.Parse()
	if flag.NArg() != 0 || *trials < 1 || *trials > 100 {
		return fmt.Errorf("use no positional arguments and trials in [1,100]")
	}
	encoder := json.NewEncoder(os.Stdout)
	for _, value := range strings.Split(*batches, ",") {
		count, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		cfg := qs.DefaultConfig()
		cfg.Paths = count
		cfg.BatchSize = count
		cfg.Workers = 1
		if err := cfg.Validate(); err != nil {
			return err
		}
		for trial := -1; trial < *trials; trial++ {
			var backend qs.Backend
			var gpu *cuda.Backend
			switch *backendName {
			case "cpu":
				backend = qs.CPU{}
			case "cuda":
				gpu = &cuda.Backend{Runner: *runner}
				backend = gpu
			default:
				return fmt.Errorf("unknown backend %q", *backendName)
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
			record := map[string]any{"backend": backend.Name(), "trial": trial + 1, "config": cfg, "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "gomaxprocs": runtime.GOMAXPROCS(0), "elapsed_ns": elapsed.Nanoseconds(), "scenarios_per_second": float64(count) / elapsed.Seconds(), "asset_paths_per_second": float64(count) / elapsed.Seconds(), "go_allocated_bytes": after.TotalAlloc - before.TotalAlloc, "go_allocations": after.Mallocs - before.Mallocs, "summary": summary}
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
