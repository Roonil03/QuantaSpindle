// Command qsstartup separates one-off Run latency from reusable-backend latency.
// JSONL retains every Run clock and the complete reusable lifecycle, including
// preparation (including its device probe), first workload, and final Close.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
	"github.com/Roonil03/QuantaSpindle/hybrid"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type benchCase struct {
	mode   string
	config qs.Config
}

func run() error {
	modes := flag.String("modes", "cpu-default,cpu,cuda-cold,cuda-reuse,auto-cold,auto-reuse", "comma-separated measurement modes")
	runner := flag.String("cuda-runner", "", "trusted CUDA executable")
	paths := flag.String("paths", "256,4096,16384,524288", "scenario counts")
	workers := flag.String("workers", "1,4,8,16", "CPU worker sweep")
	batches := flag.String("cpu-batches", "64,128,1024", "CPU batch sweep")
	trials := flag.Int("trials", 5, "measured lifecycle trials after one discarded lifecycle")
	repeats := flag.Int("repeats", 5, "additional ready Runs per reusable lifecycle")
	label := flag.String("label", "working-tree", "build/source identity (also captured by script hashes)")
	flag.Parse()
	if flag.NArg() != 0 || *trials < 1 || *trials > 100 || *repeats < 1 || *repeats > 100 {
		return fmt.Errorf("trials/repeats must be in [1,100], with no positional arguments")
	}
	counts, err := positiveInts(*paths)
	if err != nil {
		return err
	}
	ws, err := positiveInts(*workers)
	if err != nil {
		return err
	}
	bs, err := positiveInts(*batches)
	if err != nil {
		return err
	}
	var cases []benchCase
	for _, count := range counts {
		for _, mode := range strings.Split(*modes, ",") {
			cfg := qs.DefaultConfig()
			cfg.Paths = count
			switch mode {
			case "cpu-default", "auto-cold", "auto-reuse":
				cases = append(cases, benchCase{mode, cfg})
			case "cpu":
				for _, w := range ws {
					for _, b := range bs {
						c := cfg
						c.Workers, c.BatchSize = w, b
						cases = append(cases, benchCase{mode, c})
					}
				}
			case "cuda-cold", "cuda-reuse":
				cfg.Workers, cfg.BatchSize = 1, 16384
				cases = append(cases, benchCase{mode, cfg})
			default:
				return fmt.Errorf("unknown mode %q", mode)
			}
		}
	}
	for _, c := range cases {
		if err := c.config.Validate(); err != nil {
			return err
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	for trial := -1; trial < *trials; trial++ {
		for position := range cases {
			index := position
			if trial >= 0 && trial%2 == 1 {
				index = len(cases) - 1 - position
			}
			item := cases[index]
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			record, err := measure(item, *runner, *repeats)
			runtime.ReadMemStats(&after)
			if err != nil {
				return fmt.Errorf("%s paths=%d: %w", item.mode, item.config.Paths, err)
			}
			if trial < 0 {
				continue
			}
			record["trial"], record["order"], record["label"] = trial+1, position+1, *label
			record["go"], record["os"], record["arch"], record["gomaxprocs"] = runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0)
			record["go_allocated_bytes"], record["go_allocations"] = after.TotalAlloc-before.TotalAlloc, after.Mallocs-before.Mallocs
			if err := encoder.Encode(record); err != nil {
				return err
			}
		}
	}
	return nil
}

func measure(item benchCase, runner string, repeats int) (map[string]any, error) {
	ctx := context.Background()
	cfg := item.config
	record := map[string]any{"mode": item.mode, "config": cfg}
	var backend qs.Backend = qs.CPU{}
	var closeBackend func() error
	var stats func() cuda.Statistics
	var preparedAuto *cuda.Prepared
	operations := 1
	if strings.HasSuffix(item.mode, "reuse") {
		operations += repeats
	}
	started := time.Now()
	switch item.mode {
	case "cuda-cold":
		gpu := &cuda.Backend{Runner: runner}
		backend, stats = gpu, gpu.Stats
	case "cuda-reuse", "auto-reuse":
		gpu := &cuda.Backend{Runner: runner}
		p, err := gpu.Prepare(ctx)
		if err != nil {
			return nil, err
		}
		backend, closeBackend, stats = p, p.Close, gpu.Stats
		if item.mode == "auto-reuse" {
			preparedAuto = p
			record["bootstrap"] = "Prepare includes an internal one-scenario/one-step CUDA probe; no previous user workload"
		}
	case "auto-cold":
		selection, err := hybrid.Select(ctx, cfg, hybrid.Options{Runner: runner, CUDAWorkers: 1, CUDABatchSize: 16384})
		if err != nil {
			return nil, err
		}
		cfg, backend, closeBackend = selection.Config, selection.Backend, selection.Close
		record["decision"] = selection.Decision
		stats = func() cuda.Statistics { value, _ := selection.CUDAStats(); return value }
	}
	prepared := time.Now()
	var elapsed []int64
	var decisions []hybrid.Decision
	var summary qs.RunSummary
	var summaries []qs.RunSummary
	for i := 0; i < operations; i++ {
		runStart := time.Now()
		var err error
		if preparedAuto != nil {
			var selection *hybrid.Selection
			selection, err = hybrid.Select(ctx, item.config, hybrid.Options{Prepared: preparedAuto, CUDAWorkers: 1, CUDABatchSize: 16384})
			if err == nil {
				cfg = selection.Config
				summary, err = qs.Run(ctx, cfg, selection.Backend, nil)
				closeErr := selection.Close()
				if err == nil {
					err = closeErr
				}
				decisions = append(decisions, selection.Decision)
			}
		} else {
			summary, err = qs.Run(ctx, cfg, backend, nil)
		}
		elapsed = append(elapsed, time.Since(runStart).Nanoseconds())
		if err != nil {
			if closeBackend != nil {
				_ = closeBackend()
			}
			return nil, err
		}
		summaries = append(summaries, summary)
	}
	closing := time.Now()
	if closeBackend != nil {
		if err := closeBackend(); err != nil {
			return nil, err
		}
	}
	finished := time.Now()
	for _, other := range summaries[1:] {
		if !reflect.DeepEqual(summaries[0], other) {
			return nil, fmt.Errorf("repeated fixed-seed Run changed summary")
		}
	}
	total := finished.Sub(started)
	record["execution_config"], record["backend"], record["summary"] = cfg, summary.Backend, summary
	record["operations"], record["elapsed_ns"], record["amortized_ns"] = operations, total.Nanoseconds(), total.Nanoseconds()/int64(operations)
	record["prepare_ns"], record["close_ns"] = prepared.Sub(started).Nanoseconds(), finished.Sub(closing).Nanoseconds()
	record["run_elapsed_ns"], record["first_run_ns"] = elapsed, elapsed[0]
	if operations > 1 {
		record["ready_run_ns"] = elapsed[1:]
	}
	if len(decisions) > 0 {
		record["decisions"] = decisions
	}
	record["scenarios_per_second"] = float64(cfg.Paths) * float64(operations) / total.Seconds()
	if stats != nil {
		if value := stats(); value.Batches > 0 {
			record["cuda"] = value
		}
	}
	return record, nil
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
