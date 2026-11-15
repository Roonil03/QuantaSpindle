// Package hybrid selects the portable CPU or optional CUDA backend once per
// run, before delivering scenarios. Financial inputs are never changed.
// Replay is deterministic within the selected backend; CPU and CUDA use
// different random generators, so pin a backend for cross-machine replay.
package hybrid

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
)

const (
	// ColdMinPaths is a conservative routing boundary anchored to complete Run
	// measurements at 252 steps. The interval 16,384..262,144 was not measured.
	ColdMinPaths = 262144
	// WarmMinPaths applies only to an already initialized reusable CUDA service.
	WarmMinPaths = 4096
	// MinSteps is the measured horizon. Longer horizons use the same heuristic;
	// this is not a universal hardware crossover or an online timing estimator.
	MinSteps = 252
	Policy   = "measured-gbm-v1"
)

// Options configures selection. Its zero value requires no CUDA installation.
type Options struct {
	Runner string
	// Prepared optionally reuses a caller-owned CUDA service across Runs. Run
	// closes its lease, never the service. The caller must close Prepared.
	Prepared *cuda.Prepared
	// ProbeTimeout optionally bounds the first CUDA exchange after acquiring a
	// run session; zero preserves the CUDA backend's normal exchange timeout.
	// Session startup/lease acquisition respects ctx. Small and unsupported jobs
	// never start or probe an accelerator.
	ProbeTimeout time.Duration
	// Positive execution overrides opt into CUDA-specific tuning of a Config
	// copy. Zero preserves the caller's worker/batch settings. For the measured
	// cold CUDA configuration, set CUDAWorkers=1 and CUDABatchSize=16384.
	CUDAWorkers   int
	CUDABatchSize int
}

// Decision records the actual backend and the transparent routing heuristic.
type Decision struct {
	Backend      string
	Reason       string
	Policy       string
	MinCUDAPaths int
	MinSteps     int
	Warm         bool
	// Detail explains accelerator unavailability, if it caused CPU selection.
	Detail string `json:",omitempty"`
}

// Selection owns a single run's prepared resources. Use Backend and Config
// together for exactly one qs.Run and then Close, including on output errors.
// Prefer Run when pre-scenario selection metadata is not needed.
type Selection struct {
	Backend   qs.Backend
	Config    qs.Config
	Decision  Decision
	close     func() error
	stats     func() cuda.Statistics
	closeOnce sync.Once
	closeErr  error
}

// CUDAStats returns cumulative accelerator statistics when CUDA was selected.
// A caller-owned Prepared service can include earlier runs in these totals.
func (s *Selection) CUDAStats() (cuda.Statistics, bool) {
	if s == nil || s.stats == nil {
		return cuda.Statistics{}, false
	}
	return s.stats(), true
}

// Close releases the run's accelerator lease. CPU selections need no cleanup.
func (s *Selection) Close() error {
	if s == nil || s.close == nil {
		return nil
	}
	s.closeOnce.Do(func() { s.closeErr = s.close() })
	return s.closeErr
}

// Select chooses before any caller-visible result. Only classified accelerator
// unavailability permits CPU fallback. Cancellation, protocol and model errors
// propagate. A selected CUDA run never falls back after partial delivery.
func Select(ctx context.Context, c qs.Config, o Options) (*Selection, error) {
	b := &cuda.Backend{Runner: o.Runner}
	var backend qs.RunBackend = b
	stats := b.Stats
	warm := false
	if o.Prepared != nil {
		backend = o.Prepared
		warm = o.Prepared.Warm()
		stats = o.Prepared.Stats
	}
	s, err := selectBackend(ctx, c, o, backend, warm)
	if err == nil && s.close != nil {
		s.stats = stats
	}
	return s, err
}

func selectBackend(ctx context.Context, c qs.Config, o Options, accelerator qs.RunBackend, warm bool) (*Selection, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if o.ProbeTimeout < 0 || o.CUDAWorkers < 0 || o.CUDABatchSize < 0 {
		return nil, fmt.Errorf("hybrid options must be nonnegative")
	}
	threshold := ColdMinPaths
	warmLeaser, supportsWarmLease := accelerator.(interface {
		BeginWarmRun(context.Context, qs.Config) (qs.BackendSession, bool, error)
	})
	if warm {
		threshold = WarmMinPaths
	}
	s := &Selection{Backend: qs.CPU{}, Config: c, Decision: Decision{Backend: qs.CPU{}.Name(), Policy: Policy, MinCUDAPaths: threshold, MinSteps: MinSteps, Warm: warm}}
	if len(c.Assets) != 1 || len(c.Regimes) != 0 || len(c.Transition) != 0 || len(c.Correlation) != 0 || c.StorePaths {
		s.Decision.Reason = "unsupported-model"
		return s, nil
	}
	eligibilityThreshold := threshold
	if supportsWarmLease {
		eligibilityThreshold = WarmMinPaths
	}
	if c.Paths < eligibilityThreshold || c.Steps < MinSteps {
		s.Decision.Reason = "below-threshold"
		return s, nil
	}
	effective := c
	if o.CUDAWorkers > 0 {
		effective.Workers = o.CUDAWorkers
	}
	if o.CUDABatchSize > 0 {
		effective.BatchSize = o.CUDABatchSize
	}
	if err := effective.Validate(); err != nil {
		return nil, fmt.Errorf("CUDA execution settings: %w", err)
	}
	if uint64(min(effective.BatchSize, effective.Paths))*uint64(effective.Steps) > 10_000_000_000 {
		s.Decision.Reason = "cuda-dimension-limit"
		return s, nil
	}
	var session qs.BackendSession
	var err error
	if supportsWarmLease && c.Paths < ColdMinPaths {
		var acquiredWarm bool
		session, acquiredWarm, err = warmLeaser.BeginWarmRun(ctx, effective)
		if err == nil && !acquiredWarm {
			s.Decision.Warm = false
			s.Decision.MinCUDAPaths = ColdMinPaths
			s.Decision.Reason = "below-threshold"
			return s, nil
		}
	} else {
		session, err = accelerator.BeginRun(ctx, effective)
	}
	if actual, ok := session.(interface{ WasWarm() bool }); ok {
		s.Decision.Warm = actual.WasWarm()
		s.Decision.MinCUDAPaths = ColdMinPaths
		if s.Decision.Warm {
			s.Decision.MinCUDAPaths = WarmMinPaths
		}
	}
	if err == nil && session == nil {
		return nil, fmt.Errorf("CUDA returned nil run session")
	}
	var first []qs.Scenario
	if err == nil {
		probeCtx := ctx
		cancel := func() {}
		if o.ProbeTimeout > 0 {
			probeCtx, cancel = context.WithTimeout(ctx, o.ProbeTimeout)
		}
		first, err = session.Batch(probeCtx, effective, 0, min(effective.BatchSize, effective.Paths))
		cancel()
	}
	if err != nil {
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if errors.Is(err, cuda.ErrProtocol) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, cuda.ErrUnavailable) {
			return nil, err
		}
		s.Decision.Reason = "cuda-unavailable"
		s.Decision.Detail = err.Error()
		return s, nil
	}
	s.Backend = &cachedFirst{Backend: session, first: first}
	s.Config = effective
	s.close = session.Close
	s.Decision.Backend = session.Name()
	s.Decision.Reason = "cuda-threshold"
	return s, nil
}

// Run selects a backend, runs once and closes its resources. Selection overhead,
// first-batch preflight and cleanup all belong to the caller's elapsed time.
func Run(ctx context.Context, c qs.Config, o Options, sink func(qs.Scenario) error) (summary qs.RunSummary, decision Decision, err error) {
	s, err := Select(ctx, c, o)
	if err != nil {
		return summary, decision, err
	}
	decision = s.Decision
	defer func() { err = errors.Join(err, s.Close()) }()
	summary, err = qs.Run(ctx, s.Config, s.Backend, sink)
	return
}

// cachedFirst consumes the successful preflight exactly once. It deliberately
// exposes no BeginRun hook: the underlying session already belongs to this run.
type cachedFirst struct {
	qs.Backend
	mu    sync.Mutex
	first []qs.Scenario
}

func (b *cachedFirst) Batch(ctx context.Context, c qs.Config, first uint64, count int) ([]qs.Scenario, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if first == 0 {
		b.mu.Lock()
		if b.first != nil && len(b.first) == count {
			out := b.first
			b.first = nil
			b.mu.Unlock()
			return out, nil
		}
		b.mu.Unlock()
	}
	return b.Backend.Batch(ctx, c, first, count)
}
