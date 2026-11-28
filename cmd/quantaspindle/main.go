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
	"strings"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
	"github.com/Roonil03/QuantaSpindle/hybrid"
)

type scenarioRange struct {
	StartID uint64 `json:"start_id"`
	Count   uint64 `json:"count"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := execute(ctx, os.Args[1:], os.Stdout, os.Stderr); err != nil && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, "quantaspindle:", err)
		os.Exit(1)
	}
}

func execute(ctx context.Context, args []string, output, diagnostic io.Writer) (resultErr error) {
	cfg := qs.DefaultConfig()
	fs := flag.NewFlagSet("quantaspindle", flag.ContinueOnError)
	fs.SetOutput(diagnostic)
	configPath := fs.String("config", "", "strict JSON Config file; flags override explicitly supplied fields")
	backendName := fs.String("backend", "auto", "auto, cpu or cuda; explicit modes pin the backend")
	runner := fs.String("cuda-runner", "", "path to the trusted native CUDA executable")
	paths := fs.Int("paths", cfg.Paths, "scenario count, one path per asset per scenario")
	startID := fs.Uint64("start-id", 0, "first scenario ID in an explicit cpu or cuda range")
	rangeCount := fs.Uint64("count", 0, "scenarios in an explicit backend range; defaults to paths minus start-id")
	steps := fs.Int("steps", cfg.Steps, "number of time intervals")
	dt := fs.Float64("dt", cfg.DT, "interval length in years")
	seed := fs.Uint64("seed", cfg.Seed, "run seed")
	workers := fs.Int("workers", cfg.Workers, "maximum concurrent batches")
	batch := fs.Int("batch", cfg.BatchSize, "scenarios per batch")
	storePaths := fs.Bool("store-paths", false, "include complete paths; increases bounded working memory")
	summaryOnly := fs.Bool("summary-only", false, "omit individual scenario records")
	progressEvery := fs.Int("progress-every", 0, "report delivered scenario counts to stderr every N scenarios; zero disables")
	validateOnly := fs.Bool("validate-only", false, "emit resolved config and validation status without selecting a backend or running")
	selectOnly := fs.Bool("select-only", false, "preview automatic selection, probing CUDA when eligible, then close resources without running remaining scenarios")
	fs.Bool("json-errors", false, "also emit startup failures as JSON on stdout; stderr diagnostics are retained")
	timeout := fs.Duration("timeout", 0, "whole run deadline; zero disables")
	jsonErrors := scanJSONErrors(args, fs)
	outputStarted := false
	defer func() {
		if jsonErrors && !outputStarted && resultErr != nil && !errors.Is(resultErr, flag.ErrHelp) {
			if err := json.NewEncoder(output).Encode(map[string]any{"type": "error", "error": resultErr.Error()}); err != nil {
				resultErr = errors.Join(resultErr, fmt.Errorf("write JSON error: %w", err))
			}
		}
	}()
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", fs.Args())
	}
	explicitWorkers, explicitBatch, explicitRange, explicitCount := false, false, false, false
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
		// A supplied array replaces the defaults. Clear its existing backing
		// slice so json.Decoder cannot reuse a populated default asset element.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			_ = f.Close()
			return fmt.Errorf("config: %w", err)
		}
		for name := range fields {
			if strings.EqualFold(name, "Workers") {
				explicitWorkers = true
			}
			if strings.EqualFold(name, "BatchSize") {
				explicitBatch = true
			}
			if strings.EqualFold(name, "Assets") {
				cfg.Assets = nil
			}
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
		case "start-id", "count":
			explicitRange = true
			if f.Name == "count" {
				explicitCount = true
			}
		case "paths":
			cfg.Paths = *paths
		case "steps":
			cfg.Steps = *steps
		case "dt":
			cfg.DT = *dt
		case "seed":
			cfg.Seed = *seed
		case "workers":
			explicitWorkers = true
			cfg.Workers = *workers
		case "batch":
			explicitBatch = true
			cfg.BatchSize = *batch
		case "store-paths":
			cfg.StorePaths = *storePaths
		}
	})
	if *timeout < 0 {
		return errors.New("timeout must be nonnegative")
	}
	if *progressEvery < 0 {
		return errors.New("progress-every must be nonnegative")
	}
	if *validateOnly && *selectOnly {
		return errors.New("validate-only and select-only are mutually exclusive")
	}
	if *selectOnly && *backendName != "auto" {
		return errors.New("select-only requires -backend auto")
	}
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *timeout)
		defer cancel()
	}
	selectedStart, selectedCount := uint64(0), uint64(cfg.Paths)
	var requestedRange *scenarioRange
	if explicitRange {
		selectedStart, selectedCount = *startID, *rangeCount
		if !explicitCount && cfg.Paths > 0 && selectedStart < uint64(cfg.Paths) {
			selectedCount = uint64(cfg.Paths) - selectedStart
		}
		requestedRange = &scenarioRange{StartID: selectedStart, Count: selectedCount}
	}
	var validationErr error
	if explicitRange {
		validationErr = cfg.ValidateRange(selectedStart, selectedCount)
	} else {
		validationErr = cfg.Validate()
	}
	if validationErr != nil {
		return validationErr
	}
	if explicitRange && *backendName != "cpu" && *backendName != "cuda" {
		return errors.New("scenario ranges require -backend cpu or -backend cuda; auto is unsupported")
	}
	switch *backendName {
	case "auto", "cpu", "cuda":
	default:
		return fmt.Errorf("unknown backend %q; use auto, cpu or cuda", *backendName)
	}
	if *validateOnly {
		outputStarted = true
		return json.NewEncoder(output).Encode(struct {
			Type   string         `json:"type"`
			Valid  bool           `json:"valid"`
			Config qs.Config      `json:"config"`
			Range  *scenarioRange `json:"range,omitempty"`
		}{"validation", true, cfg, requestedRange})
	}
	var backend qs.Backend
	var gpu *cuda.Backend
	var selected *hybrid.Selection
	executionConfig := cfg
	started := time.Now()
	switch *backendName {
	case "auto":
		options := hybrid.Options{Runner: *runner}
		if !explicitWorkers {
			options.CUDAWorkers = 1
		}
		if !explicitBatch {
			options.CUDABatchSize = 16384
		}
		var err error
		selected, err = hybrid.Select(ctx, cfg, options)
		if err != nil {
			return err
		}
		defer selected.Close()
		backend, executionConfig = selected.Backend, selected.Config
	case "cpu":
		backend = qs.CPU{}
	case "cuda":
		gpu = &cuda.Backend{Runner: *runner}
		backend = gpu
	}
	writer := bufio.NewWriter(output)
	encoder := json.NewEncoder(writer)
	metadata := map[string]any{"type": "metadata", "config": cfg, "backend": backend.Name(), "precision": "float64", "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "gomaxprocs": runtime.GOMAXPROCS(0), "version": buildVersion(), "started_utc": started.UTC().Format(time.RFC3339Nano)}
	metadata["requested_backend"] = *backendName
	if requestedRange != nil {
		metadata["range"] = requestedRange
	}
	if selected != nil {
		metadata["selection"] = selected.Decision
		metadata["execution_config"] = executionConfig
	}
	if *selectOnly {
		// Close before claiming a successful preview, including shutdown errors.
		// The shared deferred Close remains the fallback for every earlier exit.
		if err := selected.Close(); err != nil {
			return err
		}
		metadata["type"] = "selection"
		outputStarted = true
		return json.NewEncoder(output).Encode(metadata)
	}
	// Once output is attempted, avoid appending a startup error to a partial
	// metadata record if the destination fails.
	outputStarted = true
	if err := encoder.Encode(metadata); err != nil {
		return err
	}
	var delivered, lastReport uint64
	reportProgress := func(count uint64) {
		if *progressEvery > 0 && count != lastReport {
			fmt.Fprintf(diagnostic, "quantaspindle: progress delivered=%d\n", count)
			lastReport = count
		}
	}
	sink := func(s qs.Scenario) error {
		if !*summaryOnly {
			if err := encoder.Encode(struct {
				Type     string      `json:"type"`
				Scenario qs.Scenario `json:"scenario"`
			}{"scenario", s}); err != nil {
				return err
			}
		}
		delivered++
		if *progressEvery > 0 && delivered%uint64(*progressEvery) == 0 {
			reportProgress(delivered)
		}
		return nil
	}
	var summary qs.RunSummary
	var runErr error
	if explicitRange {
		summary, runErr = qs.RunRange(ctx, executionConfig, backend, selectedStart, selectedCount, sink)
	} else {
		summary, runErr = qs.Run(ctx, executionConfig, backend, sink)
	}
	reportProgress(summary.Scenarios)
	if selected != nil {
		runErr = errors.Join(runErr, selected.Close())
	}
	elapsed := time.Since(started)
	addRunDetails := func(record map[string]any) {
		record["elapsed_ns"] = elapsed.Nanoseconds()
		if gpu != nil {
			record["cuda"] = gpu.Stats()
		}
		if selected != nil {
			record["selection"] = selected.Decision
			if stats, ok := selected.CUDAStats(); ok {
				record["cuda"] = stats
			}
		}
	}
	if runErr != nil {
		record := map[string]any{"type": "error", "error": runErr.Error(), "completed": summary.Scenarios}
		addRunDetails(record)
		_ = encoder.Encode(record)
		_ = writer.Flush()
		return runErr
	}
	result := map[string]any{"type": "summary", "summary": summary}
	addRunDetails(result)
	if requestedRange != nil {
		result["range"] = requestedRange
		result["scope"] = "range"
	}
	// Very short runs can be below a platform clock's resolution.
	if elapsed > 0 {
		result["scenarios_per_second"] = float64(summary.Scenarios) / elapsed.Seconds()
		result["asset_paths_per_second"] = float64(summary.Scenarios) * float64(len(cfg.Assets)) / elapsed.Seconds()
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
