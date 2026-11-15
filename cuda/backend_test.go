package cuda

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"os/exec"
	"reflect"
	"testing"
	"time"

	quantaspindle "github.com/Roonil03/QuantaSpindle"
)

func TestCapabilityAndInputValidation(t *testing.T) {
	b := &Backend{Runner: "intentionally-missing-quantaspindle-runner"}
	cfg := quantaspindle.DefaultConfig()
	cfg.Workers = 1
	for _, mutate := range []func(*quantaspindle.Config){
		func(c *quantaspindle.Config) { c.StorePaths = true },
		func(c *quantaspindle.Config) { c.Assets = append(c.Assets, c.Assets[0]) },
		func(c *quantaspindle.Config) { c.Correlation = [][]float64{{1}} },
		func(c *quantaspindle.Config) {
			c.Regimes = []quantaspindle.Regime{{Mu: []float64{0}, Sigma: []float64{0}}}
			c.Transition = [][]float64{{1}}
		},
	} {
		c := cfg
		mutate(&c)
		if _, err := b.Batch(context.Background(), c, 0, 1); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("capability: %v", err)
		}
	}
	for _, batch := range []struct {
		start uint64
		count int
	}{{0, 0}, {0, -1}, {1000, 1}, {999, 2}, {math.MaxUint64, 1}} {
		if _, err := b.Batch(context.Background(), cfg, batch.start, batch.count); err == nil || errors.Is(err, ErrUnavailable) {
			t.Fatalf("range not rejected: %+v %v", batch, err)
		}
	}
	if _, err := b.Batch(context.Background(), cfg, 0, 1); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("missing runner: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := b.Batch(ctx, cfg, 0, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if _, err := b.Batch(nil, cfg, 0, 1); err == nil {
		t.Fatal("nil context accepted")
	}
}

func validResponse() []byte {
	data := make([]byte, headerSize+recordSize)
	copy(data, "QSPNCR01")
	binary.LittleEndian.PutUint32(data[8:], 1)
	binary.LittleEndian.PutUint64(data[16:], 1)
	copy(data[48:], "test-device")
	for offset, value := range map[int]float64{24: 1, 32: 2, 176: 100, 184: 0, 192: 0.2, 200: 0.1} {
		binary.LittleEndian.PutUint64(data[offset:], math.Float64bits(value))
	}
	return data
}

func TestProtocolRejectsMalformedOutput(t *testing.T) {
	cfg := quantaspindle.DefaultConfig()
	for _, mutate := range []func([]byte) []byte{
		func(b []byte) []byte { return b[:len(b)-1] },
		func(b []byte) []byte { return append(b, 0) },
		func(b []byte) []byte { b[0] = 'x'; return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[16:], 0); return b },
		func(b []byte) []byte { b[8] = 2; return b },
		func(b []byte) []byte { b[12] = 1; return b },
		func(b []byte) []byte { b[16] = 2; return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[176:], math.Float64bits(math.NaN())); return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[200:], math.Float64bits(1.1)); return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[24:], math.Float64bits(-1)); return b },
	} {
		if _, _, err := decodeResponse(mutate(validResponse()), cfg, 0, 1); !errors.Is(err, ErrProtocol) {
			t.Fatalf("malformed output accepted: %v", err)
		}
	}
	s, m, err := decodeResponse(validResponse(), cfg, 0, 1)
	if err != nil || s[0].FinalPrices[0] != 100 || m.Device != "test-device" {
		t.Fatalf("valid response: %+v %+v %v", s, m, err)
	}
}

func TestBoundedBuffers(t *testing.T) {
	for _, truncate := range []bool{false, true} {
		b := limitedBuffer{limit: 4, truncate: truncate}
		n, err := b.Write([]byte("123456"))
		if b.Len() != 4 || !b.exceeded || (truncate && (n != 6 || err != nil)) || (!truncate && err == nil) {
			t.Fatalf("buffer bound: %+v %d %v", b, n, err)
		}
		_, _ = b.Write([]byte("789"))
		if b.Len() != 4 {
			t.Fatal("buffer grew past bound")
		}
	}
}

func TestProcessOutputLimits(t *testing.T) {
	if os.Getenv("QS_TEST_FLOOD") == "1" {
		chunk := bytes.Repeat([]byte("x"), 4096)
		for i := 0; i < 32; i++ {
			_, _ = os.Stderr.Write(chunk)
		}
		for i := 0; i < 32; i++ {
			_, _ = os.Stdout.Write(chunk)
		}
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestProcessOutputLimits$")
	cmd.Env = append(os.Environ(), "QS_TEST_FLOOD=1")
	output := limitedBuffer{limit: 208}
	diagnostics := limitedBuffer{limit: 16 * 1024, truncate: true}
	cmd.Stdout, cmd.Stderr = &output, &diagnostics
	_ = cmd.Run() // stdout is closed when the limit is reached.
	if !output.exceeded || output.Len() != output.limit || !diagnostics.exceeded || diagnostics.Len() != diagnostics.limit {
		t.Fatalf("process output bounds: stdout=%d exceeded=%v stderr=%d exceeded=%v", output.Len(), output.exceeded, diagnostics.Len(), diagnostics.exceeded)
	}
}

func integrationBackend(t *testing.T) *Backend {
	t.Helper()
	path := os.Getenv("QS_CUDA_RUNNER")
	if path == "" {
		path = os.Getenv("QUANTASPINDLE_CUDA_RUNNER")
	}
	if path == "" {
		t.Skip("set QS_CUDA_RUNNER to a compiled runner for actual-device checks")
	}
	return &Backend{Runner: path, Timeout: time.Minute}
}

func TestCUDAIntegration(t *testing.T) {
	b := integrationBackend(t)
	cfg := quantaspindle.DefaultConfig()
	cfg.Paths = 16384
	cfg.Steps = 64
	cfg.DT = 1.0 / 64
	cfg.Workers = 1
	whole, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual CUDA device: %+v", b.LastMetadata())
	again, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(whole, again) {
		t.Fatal("same backend replay changed")
	}
	subset, err := b.Batch(context.Background(), cfg, 71, 37)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(whole[71:108], subset) {
		t.Fatal("batch partition changed random streams")
	}
	mean, variance := moments(whole)
	a := cfg.Assets[0]
	horizon := float64(cfg.Steps) * cfg.DT
	expectedMean := a.InitialPrice * math.Exp(a.Mu*horizon)
	expectedVariance := a.InitialPrice * a.InitialPrice * math.Exp(2*a.Mu*horizon) * math.Expm1(a.Sigma*a.Sigma*horizon)
	if math.Abs(mean-expectedMean) > 6*math.Sqrt(expectedVariance/float64(cfg.Paths)) {
		t.Fatalf("analytical mean %.8g want %.8g", mean, expectedMean)
	}
	if math.Abs(variance/expectedVariance-1) > 0.10 {
		t.Fatalf("analytical variance %.8g want %.8g", variance, expectedVariance)
	}
	cpu, err := (quantaspindle.CPU{}).Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	cpuMean, cpuVariance := moments(cpu)
	if math.Abs(mean-cpuMean) > 6*math.Sqrt(2*expectedVariance/float64(cfg.Paths)) || math.Abs(variance-cpuVariance) > 0.14*expectedVariance {
		t.Fatalf("CPU/GPU moments disagree: GPU=(%g,%g) CPU=(%g,%g)", mean, variance, cpuMean, cpuVariance)
	}
	cfg.Paths = 2
	cfg.Assets[0].Sigma = 0
	cfg.Assets[0].Mu = -0.1
	deterministic, err := b.Batch(context.Background(), cfg, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := 100 * math.Exp(-0.1)
	for _, s := range deterministic {
		if math.Abs(s.FinalPrices[0]-want) > 1e-10 || s.RealizedVolatility[0] > 1e-14 || math.Abs(s.MaxDrawdown[0]-(1-want/100)) > 1e-12 {
			t.Fatalf("zero volatility: %+v", s)
		}
	}
	cfg.Steps = 1
	single, err := b.Batch(context.Background(), cfg, 0, 1)
	if err != nil || single[0].RealizedVolatility[0] != 0 {
		t.Fatalf("one-step volatility: %v %v", single, err)
	}
	cfg.Assets[0].Mu = 1e308
	cfg.DT = 1
	if _, err := b.Batch(context.Background(), cfg, 0, 1); !errors.Is(err, ErrExecution) {
		t.Fatalf("numerical overflow: %v", err)
	}
	stats := b.Stats()
	if stats.Batches != 5 || stats.Scenarios != 2*16384+37+2+1 || stats.Last.Device == "" {
		t.Fatalf("aggregate statistics: %+v", stats)
	}
}

func moments(scenarios []quantaspindle.Scenario) (float64, float64) {
	mean, m2 := 0.0, 0.0
	for i, s := range scenarios {
		delta := s.FinalPrices[0] - mean
		mean += delta / float64(i+1)
		m2 += delta * (s.FinalPrices[0] - mean)
	}
	return mean, m2 / float64(len(scenarios)-1)
}

func TestNativeRejectsMalformedRequests(t *testing.T) {
	b := integrationBackend(t)
	cfg := quantaspindle.DefaultConfig()
	for _, mutate := range []func([]byte) []byte{
		func(b []byte) []byte { return b[:10] },
		func(b []byte) []byte { return append(b, 1) },
		func(b []byte) []byte { binary.LittleEndian.PutUint32(b[12:], 0); return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[16:], 0); return b },
		func(b []byte) []byte { b[0] = 'x'; return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[16:], math.MaxUint64); return b },
		func(b []byte) []byte {
			binary.LittleEndian.PutUint64(b[24:], math.MaxUint64)
			binary.LittleEndian.PutUint64(b[16:], 2)
			return b
		},
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[40:], math.Float64bits(math.NaN())); return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[56:], math.Float64bits(-1)); return b },
		func(b []byte) []byte { binary.LittleEndian.PutUint64(b[64:], 0); return b },
		func(b []byte) []byte {
			binary.LittleEndian.PutUint32(b[12:], 2)
			binary.LittleEndian.PutUint64(b[48:], 0)
			binary.LittleEndian.PutUint64(b[56:], 0)
			binary.LittleEndian.PutUint64(b[64:], math.Float64bits(math.MaxFloat64))
			return b
		},
	} {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		cmd := exec.CommandContext(ctx, b.Runner)
		cmd.Stdin = bytes.NewReader(mutate(encodeRequest(cfg, 0, 1)))
		output, err := cmd.Output()
		cancel()
		if err == nil || len(output) != 0 {
			t.Fatalf("invalid request accepted: output=%d error=%v", len(output), err)
		}
	}
}

func TestCUDAProcessDeadline(t *testing.T) {
	b := integrationBackend(t)
	b.Timeout = time.Millisecond
	cfg := quantaspindle.DefaultConfig()
	cfg.Workers = 1
	started := time.Now()
	if _, err := b.Batch(context.Background(), cfg, 0, 1); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("native process deadline: %v", err)
	}
	if time.Since(started) > 5*time.Second {
		t.Fatal("deadline did not stop child promptly")
	}
}
