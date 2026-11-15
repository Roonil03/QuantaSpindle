package cuda

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
)

// The test executable doubles as a portable native runner, avoiding a shell or
// GPU dependency while exercising actual process/pipe/cancellation lifetimes.
func init() {
	if mode := os.Getenv("QS_TEST_STREAM_HELPER"); mode != "" {
		os.Exit(streamHelper(mode))
	}
}

func streamHelper(mode string) int {
	log := func(line string) {
		if path := os.Getenv("QS_TEST_STREAM_LOG"); path != "" {
			f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
			if err == nil {
				_, _ = fmt.Fprintln(f, line)
				_ = f.Close()
			}
		}
	}
	log("start")
	defer log("exit")
	stream := len(os.Args) == 2 && os.Args[1] == "--stream"
	if mode == "old" {
		_, _ = io.Copy(io.Discard, os.Stdin)
		return 0
	}
	if stream {
		_, _ = os.Stdout.Write([]byte("QSPNCS01"))
	}
	for {
		var request [requestSize]byte
		_, err := io.ReadFull(os.Stdin, request[:])
		if err == io.EOF {
			if mode == "shutdown" {
				return 7
			}
			return 0
		}
		if err != nil {
			return 2
		}
		if mode == "hang" {
			time.Sleep(time.Hour)
		}
		if mode == "exit" {
			_, _ = os.Stderr.Write([]byte("device failure"))
			return 7
		}
		if mode == "stderr" {
			_, _ = os.Stderr.Write([]byte(strings.Repeat("x", 64*1024)))
		}
		count := int(binary.LittleEndian.Uint64(request[16:]))
		first := binary.LittleEndian.Uint64(request[24:])
		response := make([]byte, headerSize+recordSize*count)
		copy(response, validResponse()[:headerSize])
		binary.LittleEndian.PutUint64(response[16:], uint64(count))
		for i := 0; i < count; i++ {
			price := 100 + float64(first+uint64(i))
			for j, number := range []float64{price, price/100 - 1, 0.2, 0.1} {
				binary.LittleEndian.PutUint64(response[headerSize+i*recordSize+j*8:], math.Float64bits(number))
			}
		}
		if mode == "truncated" {
			_, _ = os.Stdout.Write(response[:20])
			return 0
		}
		if mode == "malformed" {
			response[0] = 'x'
		}
		_, _ = os.Stdout.Write(response)
		if mode == "extra" {
			_, _ = os.Stdout.Write([]byte("x"))
		}
		if !stream {
			return 0
		}
	}
}

func fakeBackend(t *testing.T, mode string) (*Backend, qs.Config) {
	t.Helper()
	t.Setenv("QS_TEST_STREAM_HELPER", mode)
	b := &Backend{Runner: os.Args[0], Timeout: 5 * time.Second}
	cfg := qs.DefaultConfig()
	cfg.Paths, cfg.BatchSize, cfg.Workers = 7, 2, 3
	return b, cfg
}

func TestRunUsesOneProcessAndPreservesBatchResults(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	log := t.TempDir() + "/processes.log"
	t.Setenv("QS_TEST_STREAM_LOG", log)
	want, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	var got []qs.Scenario
	started := time.Now()
	summary, err := qs.Run(context.Background(), cfg, b, func(s qs.Scenario) error { got = append(got, s); return nil })
	elapsed := time.Since(started)
	if err != nil || summary.Scenarios != uint64(cfg.Paths) || !reflect.DeepEqual(got, want) {
		t.Fatalf("stream differs from one-shot: summary=%+v err=%v", summary, err)
	}
	data, err := os.ReadFile(log)
	if err != nil || string(data) != "start\nexit\nstart\nexit\n" {
		t.Fatalf("expected exactly one process per Run plus direct Batch: %q %v", data, err)
	}
	stats := b.Stats()
	if stats.Batches != 5 || stats.Scenarios != 14 || stats.WallTime <= 0 || stats.WallTime > elapsed+time.Second {
		t.Fatalf("stats: %+v", stats)
	}
}

func TestSessionFailuresAndBoundedDiagnostics(t *testing.T) {
	for _, mode := range []string{"truncated", "malformed", "extra", "exit", "shutdown", "stderr"} {
		t.Run(mode, func(t *testing.T) {
			b, cfg := fakeBackend(t, mode)
			cfg.Paths = 1
			_, err := qs.Run(context.Background(), cfg, b, nil)
			if mode == "stderr" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil {
				t.Fatal("runner failure accepted")
			}
		})
	}
}

func TestSessionCancellationReapsRunner(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(fmt.Sprint(deadline), func(t *testing.T) {
			b, cfg := fakeBackend(t, "hang")
			scoped, err := b.BeginRun(context.Background(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			s := scoped.(*session)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			want := context.Canceled
			if deadline {
				s.timeout = 30 * time.Millisecond
				want = context.DeadlineExceeded
			} else {
				time.AfterFunc(30*time.Millisecond, cancel)
			}
			started := time.Now()
			if _, err = s.Batch(ctx, cfg, 0, 1); !errors.Is(err, want) {
				t.Fatalf("cancellation: %v", err)
			}
			_ = s.Close()
			if time.Since(started) > 3*time.Second || s.command.ProcessState == nil {
				t.Fatal("child was not promptly reaped")
			}
		})
	}
}

func TestRunSinkFailureClosesSession(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	log := t.TempDir() + "/processes.log"
	t.Setenv("QS_TEST_STREAM_LOG", log)
	want := errors.New("sink stopped")
	_, err := qs.Run(context.Background(), cfg, b, func(qs.Scenario) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("sink error lost: %v", err)
	}
	data, _ := os.ReadFile(log)
	if string(data) != "start\nexit\n" {
		t.Fatalf("sink failure leaked runner: %q", data)
	}
}

func TestOldRunnerStartupIsBounded(t *testing.T) {
	b, cfg := fakeBackend(t, "old")
	b.Timeout = 80 * time.Millisecond
	started := time.Now()
	_, err := b.BeginRun(context.Background(), cfg)
	if !errors.Is(err, ErrUnavailable) || !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "rebuild") || time.Since(started) > 3*time.Second {
		t.Fatalf("legacy runner startup: %v", err)
	}
}

func TestCUDAStreamingIntegration(t *testing.T) {
	b := integrationBackend(t)
	cfg := qs.DefaultConfig()
	cfg.Paths, cfg.BatchSize, cfg.Workers, cfg.Steps = 257, 61, 3, 64
	want, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	var got []qs.Scenario
	_, err = qs.Run(context.Background(), cfg, b, func(s qs.Scenario) error { got = append(got, s); return nil })
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("native streaming replay differs: %v", err)
	}
}
