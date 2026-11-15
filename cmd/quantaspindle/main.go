// Command quantaspindle streams reproducible synthetic market scenarios as JSON.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := execute(ctx, os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "quantaspindle:", err)
		os.Exit(1)
	}
}

func execute(ctx context.Context, args []string, output, diagnostic io.Writer) error {
	cfg := qs.DefaultConfig()
	fs := flag.NewFlagSet("quantaspindle", flag.ContinueOnError)
	fs.SetOutput(diagnostic)
	configPath := fs.String("config", "", "strict JSON Config file; flags override explicitly supplied fields")
	backendName := fs.String("backend", "cpu", "cpu or cuda; no automatic fallback")
	runner := fs.String("cuda-runner", "", "path to the trusted native CUDA executable")
	paths := fs.Int("paths", cfg.Paths, "scenario count, one path per asset per scenario")
	steps := fs.Int("steps", cfg.Steps, "number of time intervals")
	dt := fs.Float64("dt", cfg.DT, "interval length in years")
	seed := fs.Uint64("seed", cfg.Seed, "run seed")
	workers := fs.Int("workers", cfg.Workers, "maximum concurrent batches")
	batch := fs.Int("batch", cfg.BatchSize, "scenarios per batch")
	storePaths := fs.Bool("store-paths", false, "include complete paths; increases bounded working memory")
	summaryOnly := fs.Bool("summary-only", false, "omit individual scenario records")
	timeout := fs.Duration("timeout", 0, "whole run deadline; zero disables")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", fs.Args())
	}
	if *configPath != "" {
		f, err := os.Open(*configPath)
		if err != nil {
			return err
		}
		data, readErr := io.ReadAll(io.LimitReader(f, (4<<20)+1))
		if readErr != nil {
			_ = f.Close()
			return readErr
		}
		trimmed := bytes.TrimSpace(data)
		if len(data) > 4<<20 || len(trimmed) == 0 || trimmed[0] != '{' {
			_ = f.Close()
			return errors.New("config must be a JSON object no larger than 4 MiB")
		}
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.DisallowUnknownFields()
		err = decoder.Decode(&cfg)
		if err == nil {
			var trailing any
			if extra := decoder.Decode(&trailing); extra != io.EOF {
				err = errors.New("config must contain exactly one JSON object")
			}
		}
		closeErr := f.Close()
		if err != nil {
			return fmt.Errorf("config: %w", err)
		}
		if closeErr != nil {
			return closeErr
		}
	}
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "paths":
			cfg.Paths = *paths
		case "steps":
			cfg.Steps = *steps
		case "dt":
			cfg.DT = *dt
		case "seed":
			cfg.Seed = *seed
		case "workers":
			cfg.Workers = *workers
		case "batch":
			cfg.BatchSize = *batch
		case "store-paths":
			cfg.StorePaths = *storePaths
		}
	})
	if *timeout < 0 {
		return errors.New("timeout must be nonnegative")
	}
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	var backend qs.Backend
	var gpu *cuda.Backend
	switch *backendName {
	case "cpu":
		backend = qs.CPU{}
	case "cuda":
		gpu = &cuda.Backend{Runner: *runner}
		backend = gpu
	default:
		return fmt.Errorf("unknown backend %q; use cpu or cuda", *backendName)
	}
	writer := bufio.NewWriter(output)
	encoder := json.NewEncoder(writer)
	metadata := map[string]any{"type": "metadata", "config": cfg, "backend": backend.Name(), "precision": "float64", "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "gomaxprocs": runtime.GOMAXPROCS(0), "version": buildVersion(), "started_utc": time.Now().UTC().Format(time.RFC3339Nano)}
	if err := encoder.Encode(metadata); err != nil {
		return err
	}
	started := time.Now()
	summary, runErr := qs.Run(ctx, cfg, backend, func(s qs.Scenario) error {
		if *summaryOnly {
			return nil
		}
		return encoder.Encode(struct {
			Type     string      `json:"type"`
			Scenario qs.Scenario `json:"scenario"`
		}{"scenario", s})
	})
	elapsed := time.Since(started)
	if runErr != nil {
		_ = encoder.Encode(map[string]any{"type": "error", "error": runErr.Error(), "completed": summary.Scenarios})
		_ = writer.Flush()
		return runErr
	}
	result := map[string]any{"type": "summary", "summary": summary, "elapsed_ns": elapsed.Nanoseconds()}
	// Very short runs can be below a platform clock's resolution.
	if elapsed > 0 {
		result["scenarios_per_second"] = float64(summary.Scenarios) / elapsed.Seconds()
		result["asset_paths_per_second"] = float64(summary.Scenarios) * float64(len(cfg.Assets)) / elapsed.Seconds()
	}
	if gpu != nil {
		result["cuda"] = gpu.Stats()
	}
	if err := encoder.Encode(result); err != nil {
		return err
	}
	return writer.Flush()
}

func buildVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	version := info.Main.Version
	for _, setting := range info.Settings {
		if setting.Key == "vcs.revision" {
			version += "; commit=" + setting.Value
		}
		if setting.Key == "vcs.modified" {
			version += "; modified=" + setting.Value
		}
	}
	return version
}
