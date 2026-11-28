package cuda

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
)

func TestRunRangeReplaysFakeCUDAAbsoluteIDs(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	cfg.Paths, cfg.Steps = 8, 12
	want, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	for _, prepared := range []bool{false, true} {
		var backend qs.Backend = b
		if prepared {
			p, err := b.Prepare(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			backend = p
		}
		for _, setting := range [][2]int{{1, 1}, {2, 2}, {3, 5}, {8, 8}} {
			cfg.Workers, cfg.BatchSize = setting[0], setting[1]
			var joined []qs.Scenario
			for _, bounds := range [][2]uint64{{0, 3}, {3, 5}} {
				summary, err := qs.RunRange(context.Background(), cfg, backend, bounds[0], bounds[1], func(s qs.Scenario) error { joined = append(joined, s); return nil })
				if err != nil || summary.Scenarios != bounds[1] || summary.Backend != "cuda" {
					t.Fatalf("prepared=%v bounds=%v: summary=%+v err=%v", prepared, bounds, summary, err)
				}
			}
			if !reflect.DeepEqual(joined, want) {
				t.Fatalf("absolute IDs/seeds/results changed: prepared=%v setting=%v", prepared, setting)
			}
		}
	}
}

func TestCUDARangeStartupAndBatchesUseSelectedCount(t *testing.T) {
	for _, prepared := range []bool{false, true} {
		for _, tc := range []struct {
			workers, steps int
		}{{1024, 252}, {1, 1000000}, {1024, 1000000}} {
			b, cfg := fakeBackend(t, "valid")
			cfg.Paths, cfg.BatchSize, cfg.Workers, cfg.Steps = 1000000000, 16384, tc.workers, tc.steps
			if _, err := b.BeginRun(context.Background(), cfg); err == nil {
				t.Fatal("full-run startup should reject storage/work limits")
			}
			var backend qs.Backend = b
			var p *Prepared
			if prepared {
				var err error
				p, err = b.Prepare(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				backend = p
			}
			var ids []uint64
			summary, err := qs.RunRange(context.Background(), cfg, backend, 999999997, 3, func(s qs.Scenario) error { ids = append(ids, s.ID); return nil })
			if p != nil {
				err = errors.Join(err, p.Close())
			}
			if err != nil || summary.Scenarios != 3 || !reflect.DeepEqual(ids, []uint64{999999997, 999999998, 999999999}) {
				t.Fatalf("selected count lost: prepared=%v setting=%+v ids=%v summary=%+v err=%v", prepared, tc, ids, summary, err)
			}
		}
	}
}

func TestCUDARangeRejectsInvalidOrUnsupportedBeforeStartup(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	log := filepath.Join(t.TempDir(), "runner.log")
	t.Setenv("QS_TEST_STREAM_LOG", log)
	for _, bounds := range [][2]uint64{{0, 0}, {7, 1}, {6, 2}, {math.MaxUint64, 1}, {1, math.MaxUint64}} {
		if _, err := b.BeginRunRange(context.Background(), cfg, bounds[0], bounds[1]); err == nil {
			t.Fatal("accepted invalid range", bounds)
		}
	}
	for _, change := range []func(*qs.Config){
		func(c *qs.Config) { c.StorePaths = true },
		func(c *qs.Config) { c.Assets = append(c.Assets, c.Assets[0]) },
		func(c *qs.Config) { c.Correlation = [][]float64{{1}} },
	} {
		unsupported := cfg
		change(&unsupported)
		if _, err := b.BeginRunRange(context.Background(), unsupported, 3, 2); !errors.Is(err, ErrUnsupported) {
			t.Fatalf("capability check: %v", err)
		}
	}
	if _, err := os.Stat(log); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("invalid range/model started a runner: %v", err)
	}
}

func TestCUDARangeSessionRejectsOutsideBatches(t *testing.T) {
	b, cfg := fakeBackend(t, "valid")
	s, err := b.BeginRunRange(context.Background(), cfg, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, bounds := range [][2]uint64{{2, 1}, {5, 1}, {4, 2}, {math.MaxUint64, 1}} {
		if _, err := s.Batch(context.Background(), cfg, bounds[0], int(bounds[1])); err == nil || !strings.Contains(err.Error(), "outside selected") {
			t.Fatalf("range session accepted outside batch %v: %v", bounds, err)
		}
	}
	got, err := s.Batch(context.Background(), cfg, 3, 2)
	if err != nil || len(got) != 2 || got[0].ID != 3 || got[1].ID != 4 {
		t.Fatalf("range validation disrupted session: %+v %v", got, err)
	}
}

func TestCUDAIntegrationRanges(t *testing.T) {
	b := integrationBackend(t)
	cfg := qs.DefaultConfig()
	cfg.Paths, cfg.Steps, cfg.Workers, cfg.BatchSize = 257, 64, 2, 61
	want, err := b.Batch(context.Background(), cfg, 0, cfg.Paths)
	if err != nil {
		t.Fatal(err)
	}
	var joined []qs.Scenario
	for _, bounds := range [][2]uint64{{0, 71}, {71, 186}} {
		_, err := qs.RunRange(context.Background(), cfg, b, bounds[0], bounds[1], func(s qs.Scenario) error { joined = append(joined, s); return nil })
		if err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(joined, want) {
		t.Fatal("actual-device split ranges changed IDs, seeds, or results")
	}
	t.Logf("actual CUDA device: %+v", b.LastMetadata())
}
