package cuda

import (
	"context"
	"errors"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
)

func TestPreparedReusesProcessAcrossIndependentRuns(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	log := t.TempDir() + "/processes.log"
	t.Setenv("QS_TEST_STREAM_LOG", log)
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if !p.Warm() || p.Stats().Batches != 0 || p.Stats().Scenarios != 0 {
		t.Fatalf("probe leaked into caller statistics: %+v", p.Stats())
	}
	var want []qs.Scenario
	for _, batch := range []int{2, 3, 7} {
		cfg.BatchSize = batch
		var got []qs.Scenario
		_, err := qs.Run(context.Background(), cfg, p, func(s qs.Scenario) error { got = append(got, s); return nil })
		if err != nil {
			t.Fatal(err)
		}
		if want == nil {
			want = got
		} else if !reflect.DeepEqual(want, got) {
			t.Fatal("independent Run replay changed")
		}
	}
	if p.Stats().Scenarios != 21 || p.Stats().Batches != 8 {
		t.Fatalf("caller statistics: %+v", p.Stats())
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if p.Warm() || p.Stats().SessionLifetime <= 0 {
		t.Fatal("closed process still warm or lifetime missing")
	}
	data, _ := os.ReadFile(log)
	if string(data) != "start\nexit\n" {
		t.Fatalf("expected one process across three Runs: %q", data)
	}
	if _, err := p.BeginRun(context.Background(), cfg); err == nil {
		t.Fatal("closed prepared runner reopened")
	}
}

func TestPreparedQueuedCancellationAndLeaseClose(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	lease, err := p.BeginRun(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := p.BeginRun(ctx, cfg); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued cancellation: %v", err)
	}
	if !p.Warm() {
		t.Fatal("queued cancellation invalidated active process")
	}
	_ = lease.Close()
	_ = lease.Close()
	if _, err := lease.Batch(context.Background(), cfg, 0, 1); !errors.Is(err, ErrExecution) {
		t.Fatalf("stale lease accepted batch: %v", err)
	}
	if _, err := qs.Run(context.Background(), cfg, p, nil); err != nil {
		t.Fatal(err)
	}
}

func TestPreparedCancellationRecoversOnNextRun(t *testing.T) {
	b, cfg := fakeBackend(t, "hang-workload")
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := qs.Run(ctx, cfg, p, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("failed exchange: %v", err)
	}
	if p.Warm() {
		t.Fatal("failed process remained warm")
	}
	t.Setenv("QS_TEST_STREAM_HELPER", "valid")
	if _, err := qs.Run(context.Background(), cfg, p, nil); err != nil {
		t.Fatal(err)
	}
	if !p.Warm() || p.Stats().Scenarios != uint64(cfg.Paths) {
		t.Fatalf("fresh process did not recover: %+v", p.Stats())
	}
}

func TestWarmLeaseWaitsForFailureWithoutColdRestart(t *testing.T) {
	b, cfg := fakeBackend(t, "hang-workload")
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	active, err := p.BeginRun(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		lease qs.BackendSession
		warm  bool
		err   error
	}
	ready := make(chan result, 1)
	go func() {
		lease, warm, err := p.BeginWarmRun(context.Background(), cfg)
		ready <- result{lease, warm, err}
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := active.Batch(ctx, cfg, 0, 1); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	if err := active.Close(); err != nil {
		t.Fatal(err)
	}
	r := <-ready
	if r.err != nil || r.lease != nil || r.warm || p.Warm() {
		t.Fatalf("cold service leased as warm: %+v", r)
	}
}

func TestPreparedCloseDoesNotWaitForAbandonedLease(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	lease, err := p.BeginRun(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Close()
	done := make(chan struct{})
	go func() { _ = p.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Close waited for abandoned lease")
	}
	if _, err := lease.Batch(context.Background(), cfg, 0, 1); err == nil {
		t.Fatal("closed process accepted a batch")
	}
}

func TestPrepareDeviceFailuresStayExplicit(t *testing.T) {
	for _, tc := range []struct {
		mode string
		want error
	}{
		{"unavailable", ErrUnavailable}, {"malformed", ErrProtocol}, {"exit", ErrExecution},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			b, _ := fakeBackend(t, tc.mode)
			_, err := b.Prepare(context.Background())
			if !errors.Is(err, tc.want) {
				t.Fatalf("Prepare: %v", err)
			}
			if tc.mode != "unavailable" && errors.Is(err, ErrUnavailable) {
				t.Fatalf("execution/protocol failure misclassified: %v", err)
			}
		})
	}
}

func TestPreparedConcurrentRunsAndWarmSnapshots(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = p.Warm()
			if _, err := qs.Run(context.Background(), cfg, p, nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if p.Stats().Scenarios != 4*uint64(cfg.Paths) {
		t.Fatal(p.Stats())
	}
}

func TestCUDAPreparedIntegration(t *testing.T) {
	b := integrationBackend(t)
	p, err := b.Prepare(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	cfg := qs.DefaultConfig()
	cfg.Paths, cfg.Steps, cfg.BatchSize = 257, 64, 61
	want, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	for _, batch := range []int{37, 101} {
		cfg.BatchSize = batch
		var got []qs.Scenario
		_, err := qs.Run(context.Background(), cfg, p, func(s qs.Scenario) error { got = append(got, s); return nil })
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("prepared replay differs: %v", err)
		}
	}
}
