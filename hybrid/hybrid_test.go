package hybrid

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
)

type fakeAccelerator struct {
	beginErr, batchErr error
	begins, closes     int
	mu                 sync.Mutex
	calls              map[uint64]int
}

type queuedAccelerator struct {
	fakeAccelerator
	gate chan struct{}
	warm bool
}

func (f *queuedAccelerator) BeginWarmRun(ctx context.Context, c qs.Config) (qs.BackendSession, bool, error) {
	select {
	case <-f.gate:
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}
	if !f.warm {
		return nil, false, nil
	}
	return &warmFakeLease{fakeAccelerator: &f.fakeAccelerator, warm: true}, true, nil
}

func (f *queuedAccelerator) BeginRun(ctx context.Context, c qs.Config) (qs.BackendSession, error) {
	select {
	case <-f.gate:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &warmFakeLease{fakeAccelerator: &f.fakeAccelerator, warm: f.warm}, nil
}

type warmFakeLease struct {
	*fakeAccelerator
	warm bool
}

func (f *warmFakeLease) WasWarm() bool { return f.warm }

func TestQueuedSelectionUsesAcquiredWarmth(t *testing.T) {
	for _, paths := range []int{WarmMinPaths, ColdMinPaths} {
		c := candidate()
		c.Paths = paths
		f := &queuedAccelerator{gate: make(chan struct{})}
		type result struct {
			selection *Selection
			err       error
		}
		done := make(chan result, 1)
		go func() { s, err := selectBackend(context.Background(), c, Options{}, f, true); done <- result{s, err} }()
		// The previous lease failed and invalidated its process before release.
		close(f.gate)
		r := <-done
		if r.err != nil {
			t.Fatal(r.err)
		}
		defer r.selection.Close()
		if r.selection.Decision.Warm || r.selection.Decision.MinCUDAPaths != ColdMinPaths {
			t.Fatalf("stale warm decision: %+v", r.selection.Decision)
		}
		if (r.selection.Decision.Reason == "cuda-threshold") != (paths == ColdMinPaths) {
			t.Fatalf("wrong route: %+v", r.selection.Decision)
		}
	}
}

func (f *fakeAccelerator) BeginRun(context.Context, qs.Config) (qs.BackendSession, error) {
	f.begins++
	if f.beginErr != nil {
		return nil, f.beginErr
	}
	return f, nil
}
func (*fakeAccelerator) Name() string   { return "cuda-test" }
func (f *fakeAccelerator) Close() error { f.closes++; return nil }
func (f *fakeAccelerator) Batch(ctx context.Context, c qs.Config, first uint64, count int) ([]qs.Scenario, error) {
	f.mu.Lock()
	if f.calls == nil {
		f.calls = map[uint64]int{}
	}
	f.calls[first]++
	f.mu.Unlock()
	if f.batchErr != nil {
		return nil, f.batchErr
	}
	return (qs.CPU{}).Batch(ctx, c, first, count)
}

func candidate() qs.Config {
	c := qs.DefaultConfig()
	c.Paths = ColdMinPaths
	c.Workers = 1
	c.BatchSize = 1
	return c
}

func TestRoutingDoesNotStartAcceleratorForSmallOrUnsupportedJobs(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*qs.Config)
	}{
		{"small", func(c *qs.Config) { c.Paths = ColdMinPaths - 1 }},
		{"short", func(c *qs.Config) { c.Steps = MinSteps - 1 }},
		{"assets", func(c *qs.Config) { c.Assets = append(c.Assets, c.Assets[0]) }},
		{"correlation", func(c *qs.Config) { c.Correlation = [][]float64{{1}} }},
		{"paths", func(c *qs.Config) { c.StorePaths = true }},
		{"regimes", func(c *qs.Config) {
			c.Regimes = []qs.Regime{{Mu: []float64{.05}, Sigma: []float64{.2}}}
			c.Transition = [][]float64{{1}}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := candidate()
			tc.change(&c)
			f := &fakeAccelerator{}
			s, err := selectBackend(context.Background(), c, Options{}, f, false)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			if f.begins != 0 || s.Backend.Name() != (qs.CPU{}).Name() {
				t.Fatalf("unexpected accelerator selection: %+v", s.Decision)
			}
		})
	}
}

func TestThresholdUsesWarmthAndLongerHorizons(t *testing.T) {
	for _, tc := range []struct {
		paths, steps   int
		warm, wantCUDA bool
	}{
		{ColdMinPaths, 252, false, true}, {ColdMinPaths, 253, false, true}, {ColdMinPaths - 1, 252, false, false},
		{WarmMinPaths, 252, true, true}, {WarmMinPaths - 1, 252, true, false}, {WarmMinPaths, 251, true, false}, {WarmMinPaths, 252, false, false},
	} {
		c := candidate()
		c.Paths = tc.paths
		c.Steps = tc.steps
		f := &fakeAccelerator{}
		s, err := selectBackend(context.Background(), c, Options{}, f, tc.warm)
		if err != nil {
			t.Fatal(err)
		}
		s.Close()
		if (f.begins == 1) != tc.wantCUDA {
			t.Fatalf("%+v => %+v", tc, s.Decision)
		}
	}
}

func TestOnlyUnavailableFallsBackAndCancellationPropagates(t *testing.T) {
	for _, atBegin := range []bool{true, false} {
		for _, failure := range []error{cuda.ErrUnavailable, cuda.ErrExecution, cuda.ErrProtocol, errors.Join(cuda.ErrUnavailable, cuda.ErrProtocol), context.DeadlineExceeded} {
			f := &fakeAccelerator{}
			if atBegin {
				f.beginErr = failure
			} else {
				f.batchErr = failure
			}
			s, err := selectBackend(context.Background(), candidate(), Options{}, f, false)
			if errors.Is(failure, cuda.ErrUnavailable) && !errors.Is(failure, cuda.ErrProtocol) {
				if err != nil || s.Decision.Reason != "cuda-unavailable" || s.Decision.Detail == "" {
					t.Fatalf("fallback: %+v %v", s, err)
				}
			} else if !errors.Is(err, failure) {
				t.Fatalf("error %v lost: %v", failure, err)
			}
			if !atBegin && f.closes != 1 {
				t.Fatal("failed probe leaked session")
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	f := &fakeAccelerator{}
	if _, err := selectBackend(ctx, candidate(), Options{}, f, false); !errors.Is(err, context.Canceled) || f.begins != 0 {
		t.Fatal("canceled selection started work", err)
	}
}

func TestPublicSelectionFallsBackForMissingRunner(t *testing.T) {
	s, err := Select(context.Background(), candidate(), Options{Runner: "missing-quantaspindle-runner-for-test"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.Decision.Reason != "cuda-unavailable" || s.Backend.Name() != (qs.CPU{}).Name() {
		t.Fatalf("missing runner did not route to CPU: %+v", s.Decision)
	}
}

func TestCUDAUpdateLimitRoutesCPUWithoutStartingRunner(t *testing.T) {
	c := candidate()
	c.Steps = 1_000_000
	c.BatchSize = 16384
	f := &fakeAccelerator{}
	s, err := selectBackend(context.Background(), c, Options{}, f, false)
	if err != nil {
		t.Fatal(err)
	}
	if f.begins != 0 || s.Decision.Reason != "cuda-dimension-limit" {
		t.Fatalf("invalid CUDA shape started runner: %+v", s.Decision)
	}
}

func TestPreflightIsCachedAndSinkFailureNeverReruns(t *testing.T) {
	c := candidate()
	f := &fakeAccelerator{}
	s, err := selectBackend(context.Background(), c, Options{}, f, false)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("sink stopped")
	seen := 0
	summary, err := qs.Run(context.Background(), s.Config, s.Backend, func(s qs.Scenario) error {
		seen++
		if seen == 3 {
			return want
		}
		return nil
	})
	if !errors.Is(err, want) || summary.Scenarios != 2 || summary.Backend != "cuda-test" {
		t.Fatalf("partial run: %+v %v", summary, err)
	}
	if f.calls[0] != 1 || f.begins != 1 || seen != 3 {
		t.Fatalf("reran work: %+v", f.calls)
	}
	s.Close()
	s.Close()
	if f.closes != 1 {
		t.Fatal("Close is not idempotent")
	}
}

func TestTuningUsesCopyAndCPUHasNoOverrides(t *testing.T) {
	c := candidate()
	before := c
	f := &fakeAccelerator{}
	s, err := selectBackend(context.Background(), c, Options{CUDAWorkers: 2, CUDABatchSize: 3}, f, false)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if !reflect.DeepEqual(c, before) || s.Config.Workers != 2 || s.Config.BatchSize != 3 || !reflect.DeepEqual(c.Assets, s.Config.Assets) {
		t.Fatal("changed model or ignored execution override")
	}
	c.Paths = 1
	s, err = selectBackend(context.Background(), c, Options{CUDAWorkers: 2, CUDABatchSize: 3}, f, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.Config, c) {
		t.Fatal("CPU settings changed")
	}
}

func TestAutomaticCPUReplayMatchesExplicitCPU(t *testing.T) {
	c := qs.DefaultRegimeConfig()
	c.Paths = 7
	c.Steps = 3
	c.BatchSize = 2
	var automatic, explicit []qs.Scenario
	summary, decision, err := Run(context.Background(), c, Options{Runner: "missing-runner-must-not-start"}, func(s qs.Scenario) error { automatic = append(automatic, s); return nil })
	if err != nil {
		t.Fatal(err)
	}
	other, err := qs.Run(context.Background(), c, qs.CPU{}, func(s qs.Scenario) error { explicit = append(explicit, s); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(automatic, explicit) || !reflect.DeepEqual(summary, other) || decision.Backend != other.Backend {
		t.Fatal("automatic CPU changed replay or metadata")
	}
}
