package cuda

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
)

// session is private to one Run. The token serializes stream transactions while
// allowing queued callers to observe cancellation without waiting for a mutex.
type session struct {
	owner       *Backend
	command     *exec.Cmd
	stdin       io.WriteCloser
	stdout      io.ReadCloser
	cancel      context.CancelFunc
	timeout     time.Duration
	token       chan struct{}
	diagnostics limitedBuffer
	once        sync.Once
	closed      bool
	err         error
	started     time.Time
	batchWall   time.Duration
}

// BeginRun starts one native streaming runner for all batches in a Run. Native
// runners predating streaming must be rebuilt; direct Batch remains v1 compatible.
func (b *Backend) BeginRun(ctx context.Context, cfg qs.Config) (qs.BackendSession, error) {
	if err := validateBatch(ctx, cfg, 0, min(cfg.BatchSize, cfg.Paths)); err != nil {
		return nil, err
	}
	path, timeout, err := b.runner()
	if err != nil {
		return nil, err
	}
	childCtx, cancel := context.WithCancel(ctx)
	s := &session{owner: b, cancel: cancel, timeout: timeout, token: make(chan struct{}, 1), started: time.Now(), diagnostics: limitedBuffer{limit: 16 * 1024, truncate: true}}
	s.command = exec.CommandContext(childCtx, path, "--stream")
	s.command.WaitDelay = 2 * time.Second
	s.command.Stderr = &s.diagnostics
	if s.stdin, err = s.command.StdinPipe(); err != nil {
		cancel()
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	if s.stdout, err = s.command.StdoutPipe(); err != nil {
		_ = s.stdin.Close()
		cancel()
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	if err = s.command.Start(); err != nil {
		_ = s.stdin.Close()
		_ = s.stdout.Close()
		cancel()
		return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	// The greeting precedes CUDA initialization. A short bound catches old
	// runners, which ignore --stream and wait for a one-shot request until EOF.
	startupCtx, stop := context.WithTimeout(ctx, min(timeout, 5*time.Second))
	defer stop()
	ready := make(chan error, 1)
	go func() {
		var greeting [8]byte
		_, err := io.ReadFull(s.stdout, greeting[:])
		if err == nil && string(greeting[:]) != "QSPNCS01" {
			err = ErrProtocol
		}
		ready <- err
	}()
	select {
	case err = <-ready:
	case <-startupCtx.Done():
		err = startupCtx.Err()
	}
	if err != nil {
		s.cancel()
		_ = s.stdout.Close()
		_ = s.finish()
		return nil, fmt.Errorf("%w: streaming startup: %w; rebuild the CUDA runner for --stream", ErrUnavailable, err)
	}
	return s, nil
}

func (*session) Name() string { return "cuda" }

func (s *session) Batch(ctx context.Context, cfg qs.Config, first uint64, count int) ([]qs.Scenario, error) {
	if err := validateBatch(ctx, cfg, first, count); err != nil {
		return nil, err
	}
	select {
	case s.token <- struct{}{}:
		defer func() { <-s.token }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if s.closed {
		return nil, fmt.Errorf("%w: session closed", ErrExecution)
	}
	childCtx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	started := time.Now()
	response := make([]byte, headerSize+recordSize*count)
	done := make(chan error, 1)
	go func() {
		_, err := s.stdin.Write(encodeRequest(cfg, first, count))
		if err == nil {
			_, err = io.ReadFull(s.stdout, response)
		}
		done <- err
	}()
	var err error
	select {
	case err = <-done:
	case <-childCtx.Done():
		err = childCtx.Err()
		s.cancel()
		_ = s.stdin.Close()
		_ = s.stdout.Close()
		<-done
	}
	if err != nil {
		s.cancel()
		_ = s.stdout.Close()
		closeErr := s.finish()
		if childCtx.Err() != nil {
			return nil, childCtx.Err()
		}
		return nil, errors.Join(fmt.Errorf("%w: streaming exchange: %v", ErrExecution, err), closeErr)
	}
	scenarios, metadata, err := decodeResponse(response, cfg, first, count)
	if err != nil {
		s.cancel()
		_ = s.stdout.Close()
		_ = s.finish()
		return nil, err
	}
	metadata.WallTime = time.Since(started)
	s.batchWall += metadata.WallTime
	s.owner.record(metadata, count)
	return scenarios, nil
}

// Close waits for graceful EOF shutdown, then kills a stuck child after two
// seconds. It is idempotent and includes process startup/teardown in Stats.
func (s *session) Close() error {
	s.token <- struct{}{}
	defer func() { <-s.token }()
	return s.finish()
}

func (s *session) finish() error {
	s.once.Do(func() {
		s.closed = true
		_ = s.stdin.Close()
		// Drain at most one trailing byte. Additional bytes violate the fixed
		// frame contract; never buffer an untrusted stream until EOF.
		trailing := make(chan error, 1)
		go func() {
			var data [1]byte
			n, err := s.stdout.Read(data[:])
			if n != 0 {
				err = fmt.Errorf("%w: trailing stream output", ErrProtocol)
			}
			if err == io.EOF {
				err = nil
			}
			trailing <- err
		}()
		timer := time.AfterFunc(2*time.Second, func() { s.cancel(); _ = s.stdout.Close() })
		readErr := <-trailing
		waitErr := s.command.Wait()
		timer.Stop()
		s.cancel()
		_ = s.stdout.Close()
		if waitErr != nil {
			s.err = fmt.Errorf("%w: %v: %s", ErrExecution, waitErr, strings.TrimSpace(s.diagnostics.String()))
		}
		if errors.Is(readErr, ErrProtocol) {
			s.err = errors.Join(s.err, readErr)
		}
		s.owner.mu.Lock()
		s.owner.stats.WallTime += time.Since(s.started) - s.batchWall
		s.owner.mu.Unlock()
	})
	return s.err
}
