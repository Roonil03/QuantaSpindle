package cuda

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	qs "github.com/Roonil03/QuantaSpindle"
)

// Prepared keeps one native process and its CUDA allocations across independent
// Runs. Runs are serialized; queued callers can cancel without disrupting the
// active Run. Call Close to release the process, including after cancellation.
// A failed exchange invalidates the process; the next Run starts a fresh one.
// Retained device memory grows to the largest batch until Close or a failure.
type Prepared struct {
	owner    *Backend
	ctx      context.Context
	cancel   context.CancelFunc
	stopping context.Context
	stop     context.CancelFunc
	gate     chan struct{}
	mu       sync.Mutex
	stream   *session
	ready    atomic.Pointer[session]
	closed   bool
	once     sync.Once
	err      error
}

// Prepare starts a reusable streaming runner. The context owns its lifetime;
// each Run also observes its own context. A fixed one-scenario, one-step device
// probe initializes CUDA and validates a real result without updating caller
// batch statistics. Include Prepare in setup costs. A first full workload may
// still grow device storage. The v0.1.1 runner protocol is sufficient.
func (b *Backend) Prepare(ctx context.Context) (*Prepared, error) {
	if ctx == nil {
		return nil, fmt.Errorf("CUDA: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lifetime, cancel := context.WithCancel(ctx)
	stopping, stop := context.WithCancel(context.Background())
	p := &Prepared{owner: b, ctx: lifetime, cancel: cancel, stopping: stopping, stop: stop, gate: make(chan struct{}, 1)}
	s, err := b.startSession(lifetime, ctx)
	if err != nil {
		cancel()
		stop()
		return nil, err
	}
	p.stream = s
	probe := qs.DefaultConfig()
	probe.Paths, probe.BatchSize, probe.Workers, probe.Steps = 1, 1, 1, 1
	if _, err := s.batch(ctx, probe, 0, 1, false); err != nil {
		_ = p.Close()
		return nil, err
	}
	p.ready.Store(s)
	return p, nil
}

func (*Prepared) Name() string { return "cuda" }

// Stats reports cumulative statistics on the Backend that created this runner.
// SessionLifetime is finalized only when a process is closed, not per Run.
func (p *Prepared) Stats() Statistics { return p.owner.Stats() }

func (p *Prepared) LastMetadata() Metadata { return p.owner.LastMetadata() }

// Warm reports whether the current live process completed a successful batch.
// It is a nonblocking snapshot, false during recovery or after closure; it does
// not predict performance or reserve the process for a future Run.
func (p *Prepared) Warm() bool {
	s := p.ready.Load()
	return p.ctx.Err() == nil && s != nil && s.warm.Load()
}

// BeginRun leases this runner exclusively until the returned session is closed.
// Closing the lease retains the native process for the next independent Run.
func (p *Prepared) BeginRun(ctx context.Context, cfg qs.Config) (qs.BackendSession, error) {
	lease, _, err := p.beginRun(ctx, cfg, false)
	return lease, err
}

// BeginWarmRun atomically leases an already warm process without starting a
// replacement. A nil lease and false indicate that no warm process is available.
func (p *Prepared) BeginWarmRun(ctx context.Context, cfg qs.Config) (qs.BackendSession, bool, error) {
	return p.beginRun(ctx, cfg, true)
}

func (p *Prepared) beginRun(ctx context.Context, cfg qs.Config, warmOnly bool) (qs.BackendSession, bool, error) {
	if err := validateBatch(ctx, cfg, 0, min(cfg.BatchSize, cfg.Paths)); err != nil {
		return nil, false, err
	}
	select {
	case p.gate <- struct{}{}:
	case <-ctx.Done():
		return nil, false, ctx.Err()
	case <-p.ctx.Done():
		return nil, false, p.ctx.Err()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	fail := func(err error) (qs.BackendSession, bool, error) { <-p.gate; return nil, false, err }
	if p.closed {
		return fail(fmt.Errorf("%w: prepared runner closed", ErrExecution))
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	if err := p.ctx.Err(); err != nil {
		return fail(err)
	}
	warm := p.stream != nil && !p.stream.closed && p.stream.warm.Load()
	if warmOnly && !warm {
		return fail(nil)
	}
	if p.stream == nil || p.stream.closed {
		p.ready.Store(nil)
		startup, cancel := context.WithCancel(ctx)
		stop := context.AfterFunc(p.stopping, cancel)
		s, err := p.owner.startSession(p.ctx, startup)
		stop()
		cancel()
		if err != nil {
			return fail(err)
		}
		p.stream = s
		p.ready.Store(s)
	}
	return &preparedLease{parent: p, stream: p.stream, warm: warm}, warm, nil
}

// Batch leases the runner for one direct batch. Concurrent calls are serialized.
func (p *Prepared) Batch(ctx context.Context, cfg qs.Config, first uint64, count int) ([]qs.Scenario, error) {
	if err := validateBatch(ctx, cfg, first, count); err != nil {
		return nil, err
	}
	lease, err := p.BeginRun(ctx, cfg)
	if err != nil {
		return nil, err
	}
	defer lease.Close()
	return lease.Batch(ctx, cfg, first, count)
}

// Close is idempotent. Idle runners exit on EOF; active exchanges are interrupted
// so Close does not wait for a Run's timeout or a blocked user sink. Native
// shutdown is bounded by the streaming session's two-second kill timer.
func (p *Prepared) Close() error {
	p.once.Do(func() {
		p.ready.Store(nil)
		// Interrupt a recovery handshake before waiting for mu. Idle processes
		// still exit on EOF, preserving their native shutdown diagnostics.
		p.stop()
		p.mu.Lock()
		p.closed = true
		p.ready.Store(nil)
		s := p.stream
		active := len(p.gate) != 0
		p.mu.Unlock()
		if s != nil {
			if active {
				s.cancel()
				_ = s.stdin.Close()
				_ = s.stdout.Close()
			}
			p.err = s.Close()
		}
		p.cancel()
	})
	return p.err
}

type preparedLease struct {
	parent *Prepared
	stream *session
	mu     sync.RWMutex
	closed bool
	warm   bool
}

func (*preparedLease) Name() string { return "cuda" }

// WasWarm reports the process state when this exclusive lease was acquired.
func (s *preparedLease) WasWarm() bool { return s.warm }

func (s *preparedLease) Batch(ctx context.Context, cfg qs.Config, first uint64, count int) ([]qs.Scenario, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, fmt.Errorf("%w: prepared Run session closed", ErrExecution)
	}
	return s.stream.Batch(ctx, cfg, first, count)
}

func (s *preparedLease) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		<-s.parent.gate
	}
	return nil
}
