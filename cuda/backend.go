// Package cuda executes optional FP64 GBM kernels through a native CUDA runner.
// It supports one asset with constant drift and volatility, no supplied
// correlation or regimes, and no stored paths. Unsupported requests fail
// explicitly; there is no CPU fallback. Ordinary Go builds need neither CUDA
// nor cgo, but execution needs a locally built runner and a working NVIDIA GPU.
//
// Pass &Backend{Runner: trustedExecutablePath} to quantaspindle.Run. From v0.1.1,
// Run owns a streaming process for its lifetime and reuses CUDA resources across
// serialized batches. Direct Backend.Batch calls retain the one-shot interface.
// Rebuild the native runner with the adapter to provide the --stream capability.
// Runner may instead be selected through QS_CUDA_RUNNER,
// QUANTASPINDLE_CUDA_RUNNER, or PATH. Only select a trusted executable.
//
// Calling [Backend.Prepare] initializes one GPU process for repeated runs.
// Pass the returned [Prepared] to quantaspindle.Run for explicit CUDA or to
// the hybrid subpackage for automatic selection. Close it when the service
// stops. Its context must live as long as the service.
//
// CUDA replay is scoped to the same runner build, device and toolchain. The
// Philox seed/scenario-ID mapping differs from the CPU generator, so compare
// distributions rather than paths across backends. Stats separates successful
// batch timings from SessionLifetime, which includes process startup, idle/sink
// time, shutdown and failed sessions after closure. Measure the entire Run for
// complete caller latency; Go allocation counts exclude native/device memory.
// See the repository CUDA guide and performance report for build commands,
// protocol, actual measurements, and their limits:
// https://github.com/Roonil03/QuantaSpindle/blob/main/docs/cuda.md.
package cuda

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	quantaspindle "github.com/Roonil03/QuantaSpindle"
)

const (
	requestSize         = 72
	headerSize          = 176
	recordSize          = 32
	maxScenarios        = 1000000
	maxSteps            = 1000000
	maxUpdates   uint64 = 10000000000
)

var (
	// ErrUnsupported means the requested model exceeds CUDA's current capabilities.
	ErrUnsupported = errors.New("CUDA capability unsupported")
	// ErrUnavailable means the native runner could not be located or started.
	ErrUnavailable = errors.New("CUDA runner unavailable")
	// ErrProtocol means the runner returned malformed or excessive output.
	ErrProtocol = errors.New("invalid CUDA runner response")
	// ErrExecution means the native process or device reported a failure.
	ErrExecution = errors.New("CUDA execution failed")
)

// Metadata describes the actual device and timing of the last successful batch.
// KernelTime excludes transfers and startup; HostTime is measured inside the
// runner through output transfer. WallTime covers the successful one-shot
// process invocation, or the successful streaming request/response exchange.
type Metadata struct {
	Device          string
	RuntimeVersion  int
	DriverVersion   int
	KernelTime      time.Duration
	HostTime        time.Duration
	WallTime        time.Duration
	Generator       string
	ProtocolVersion int
}

// Backend executes single-asset, constant-parameter terminal GBM batches on a
// real CUDA device. Configure its fields before concurrent use. No CPU fallback
// occurs. Direct Batch calls start fresh processes; Run reuses one process.
type Backend struct {
	// Runner is the executable path. Empty uses QS_CUDA_RUNNER, then
	// QUANTASPINDLE_CUDA_RUNNER, then searches PATH for quantaspindle-cuda.
	Runner string
	// Timeout bounds a Batch exchange (including startup); zero selects five minutes.
	Timeout  time.Duration
	mu       sync.RWMutex
	metadata Metadata
	stats    Statistics
}

// Name returns the backend's stable identifier.
func (*Backend) Name() string { return "cuda" }

// LastMetadata returns a synchronized snapshot of the last successful batch.
// Concurrent batches overwrite the snapshot in completion order.
func (b *Backend) LastMetadata() Metadata {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.metadata
}

// Statistics aggregates successful batches since this Backend was created.
// KernelTime, HostTime and WallTime sum successful batch timings. SessionLifetime
// sums streaming process lifetimes, including startup, idle/sink time, teardown
// and failed sessions. Concurrent durations are summed, not elapsed run time.
type Statistics struct {
	Batches         uint64
	Scenarios       uint64
	KernelTime      time.Duration
	HostTime        time.Duration
	WallTime        time.Duration
	SessionLifetime time.Duration
	Last            Metadata
}

// Stats returns a synchronized snapshot of cumulative successful batch metadata.
func (b *Backend) Stats() Statistics {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.stats
}

// Batch returns scenarios in increasing ID order, or an error and no partial
// results. It validates both common inputs and CUDA-specific capabilities.
func (b *Backend) Batch(ctx context.Context, cfg quantaspindle.Config, start uint64, count int) ([]quantaspindle.Scenario, error) {
	if err := validateBatch(ctx, cfg, start, count); err != nil {
		return nil, err
	}
	return b.oneShot(ctx, cfg, start, count)
}

func validateBatch(ctx context.Context, cfg quantaspindle.Config, start uint64, count int) error {
	if ctx == nil {
		return errors.New("CUDA: nil context")
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if len(cfg.Assets) != 1 || len(cfg.Regimes) != 0 || len(cfg.Transition) != 0 || len(cfg.Correlation) != 0 || cfg.StorePaths {
		return fmt.Errorf("%w: requires one asset, no correlation/regimes, and StorePaths=false", ErrUnsupported)
	}
	if count <= 0 || count > maxScenarios || cfg.Steps > maxSteps || uint64(count) > maxUpdates/uint64(cfg.Steps) || start >= uint64(cfg.Paths) || uint64(count) > uint64(cfg.Paths)-start {
		return fmt.Errorf("CUDA: invalid batch range or runner dimension limits (count<=%d, steps<=%d, updates<=%d)", maxScenarios, maxSteps, maxUpdates)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

func (b *Backend) runner() (string, time.Duration, error) {
	path := b.Runner
	if path == "" {
		path = os.Getenv("QS_CUDA_RUNNER")
	}
	if path == "" {
		path = os.Getenv("QUANTASPINDLE_CUDA_RUNNER")
	}
	if path == "" {
		path = "quantaspindle-cuda"
	}
	resolved, err := exec.LookPath(path)
	if err != nil {
		return "", 0, fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	timeout := b.Timeout
	if timeout == 0 {
		timeout = 5 * time.Minute
	}
	if timeout < 0 {
		return "", 0, errors.New("CUDA: timeout must be nonnegative")
	}
	return resolved, timeout, nil
}

func (b *Backend) oneShot(ctx context.Context, cfg quantaspindle.Config, start uint64, count int) ([]quantaspindle.Scenario, error) {
	resolved, timeout, err := b.runner()
	if err != nil {
		return nil, err
	}
	childCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	command := exec.CommandContext(childCtx, resolved)
	command.WaitDelay = 2 * time.Second
	request := encodeRequest(cfg, start, count)
	command.Stdin = bytes.NewReader(request)
	output := limitedBuffer{limit: headerSize + recordSize*count}
	diagnostics := limitedBuffer{limit: 16 * 1024, truncate: true}
	command.Stdout = &output
	command.Stderr = &diagnostics
	wallStart := time.Now()
	err = command.Run()
	wallTime := time.Since(wallStart)
	if childCtx.Err() != nil {
		return nil, childCtx.Err()
	}
	if output.exceeded {
		return nil, fmt.Errorf("%w: output exceeds expected length", ErrProtocol)
	}
	if err != nil {
		var exit *exec.ExitError
		if !errors.As(err, &exit) {
			return nil, fmt.Errorf("%w: %v", ErrUnavailable, err)
		}
		return nil, executionError(err, strings.TrimSpace(diagnostics.String()))
	}
	scenarios, metadata, err := decodeResponse(output.Bytes(), cfg, start, count)
	if err != nil {
		return nil, err
	}
	metadata.WallTime = wallTime
	b.record(metadata, count)
	return scenarios, nil
}

func executionError(err error, diagnostics string) error {
	result := fmt.Errorf("%w: %v: %s", ErrExecution, err, diagnostics)
	// Only the runner's explicit initialization marker is eligible for
	// unavailable-device handling. Numerical and protocol errors stay fatal.
	if strings.HasPrefix(diagnostics, "quantaspindle-cuda: CUDA unavailable:") ||
		strings.HasPrefix(diagnostics, "quantaspindle-cuda: device enumeration:") ||
		strings.HasPrefix(diagnostics, "quantaspindle-cuda: device selection:") ||
		diagnostics == "quantaspindle-cuda: no CUDA device" {
		return errors.Join(ErrUnavailable, result)
	}
	return result
}

func (b *Backend) record(metadata Metadata, count int) {
	b.mu.Lock()
	b.metadata = metadata
	b.stats.Batches++
	b.stats.Scenarios += uint64(count)
	b.stats.KernelTime += metadata.KernelTime
	b.stats.HostTime += metadata.HostTime
	b.stats.WallTime += metadata.WallTime
	b.stats.Last = metadata
	b.mu.Unlock()
}

func encodeRequest(cfg quantaspindle.Config, start uint64, count int) []byte {
	data := make([]byte, requestSize)
	copy(data, "QSPNCU01")
	binary.LittleEndian.PutUint32(data[8:], 1)
	binary.LittleEndian.PutUint32(data[12:], uint32(cfg.Steps))
	binary.LittleEndian.PutUint64(data[16:], uint64(count))
	binary.LittleEndian.PutUint64(data[24:], start)
	binary.LittleEndian.PutUint64(data[32:], cfg.Seed)
	asset := cfg.Assets[0]
	for i, value := range []float64{asset.InitialPrice, asset.Mu, asset.Sigma, cfg.DT} {
		binary.LittleEndian.PutUint64(data[40+8*i:], math.Float64bits(value))
	}
	return data
}

func decodeResponse(data []byte, cfg quantaspindle.Config, start uint64, count int) ([]quantaspindle.Scenario, Metadata, error) {
	bad := func(message string) ([]quantaspindle.Scenario, Metadata, error) {
		return nil, Metadata{}, fmt.Errorf("%w: %s", ErrProtocol, message)
	}
	if len(data) != headerSize+recordSize*count {
		return bad("unexpected response size")
	}
	if string(data[:8]) != "QSPNCR01" || binary.LittleEndian.Uint32(data[8:]) != 1 || binary.LittleEndian.Uint32(data[12:]) != 0 || binary.LittleEndian.Uint64(data[16:]) != uint64(count) {
		return bad("invalid header")
	}
	readFloat := func(offset int) float64 { return math.Float64frombits(binary.LittleEndian.Uint64(data[offset:])) }
	kernel, host := readFloat(24), readFloat(32)
	if !finite(kernel) || !finite(host) || kernel < 0 || host < kernel || host > float64((24*time.Hour)/time.Millisecond) {
		return bad("invalid timing")
	}
	device := strings.TrimRight(string(data[48:176]), "\x00")
	if device == "" {
		return bad("empty device identity")
	}
	metadata := Metadata{Device: device, RuntimeVersion: int(binary.LittleEndian.Uint32(data[40:])), DriverVersion: int(binary.LittleEndian.Uint32(data[44:])), KernelTime: time.Duration(kernel * float64(time.Millisecond)), HostTime: time.Duration(host * float64(time.Millisecond)), Generator: "cuRAND Philox4_32_10; seed=config.Seed, sequence=scenario.ID, offset=0", ProtocolVersion: 1}
	scenarios := make([]quantaspindle.Scenario, count)
	for i := range scenarios {
		offset := headerSize + i*recordSize
		terminal, totalReturn, vol, drawdown := readFloat(offset), readFloat(offset+8), readFloat(offset+16), readFloat(offset+24)
		if !finite(terminal) || terminal <= 0 || !finite(totalReturn) || totalReturn < -1 || !finite(vol) || vol < 0 || !finite(drawdown) || drawdown < 0 || drawdown > 1 {
			return bad("invalid scenario numbers")
		}
		id := start + uint64(i)
		scenarios[i] = quantaspindle.Scenario{ID: id, Seed: quantaspindle.ScenarioSeed(cfg.Seed, id), FinalPrices: []float64{terminal}, Returns: []float64{totalReturn}, RealizedVolatility: []float64{vol}, MaxDrawdown: []float64{drawdown}}
	}
	return scenarios, metadata, nil
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

type limitedBuffer struct {
	buffer   bytes.Buffer
	limit    int
	truncate bool
	exceeded bool
}

func (b *limitedBuffer) Len() int       { return b.buffer.Len() }
func (b *limitedBuffer) Bytes() []byte  { return b.buffer.Bytes() }
func (b *limitedBuffer) String() string { return b.buffer.String() }

func (b *limitedBuffer) Write(data []byte) (int, error) {
	remaining := b.limit - b.Len()
	if len(data) <= remaining {
		return b.buffer.Write(data)
	}
	b.exceeded = true
	b.buffer.Write(data[:remaining])
	if b.truncate {
		return len(data), nil
	}
	return remaining, errors.New("CUDA output size limit exceeded")
}
