package cuda

import (
	"context"
	"errors"
	"reflect"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
)

func TestRangeSummaryStateUsesSelectedCUDAStartup(t *testing.T) {
	// A full-run session rejects these storage/work limits. The optional state
	// API must preserve explicit range initialization for both backend lifetimes.
	for _, prepared := range []bool{false, true} {
		b, cfg := fakeBackend(t, "valid")
		cfg.Paths, cfg.Steps, cfg.Workers, cfg.BatchSize = 1_000_000_000, 1_000_000, 1024, 16384
		if _, err := b.BeginRun(context.Background(), cfg); err == nil {
			t.Fatal("full-run startup unexpectedly accepted the oversized workload")
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
		summary, state, err := qs.RunRangeWithState(context.Background(), cfg, backend, 999999997, 3, "fake-cuda-range/v1", func(s qs.Scenario) error {
			ids = append(ids, s.ID)
			s.Returns[0] = 99 // State must use the validated pre-sink value.
			return nil
		})
		if p != nil {
			err = errors.Join(err, p.Close())
		}
		if err != nil || summary.Scenarios != 3 || state.Count() != 3 || !reflect.DeepEqual(ids, []uint64{999999997, 999999998, 999999999}) || !reflect.DeepEqual(state.Ranges(), []qs.ScenarioRange{{StartID: 999999997, Count: 3}}) {
			t.Fatalf("selected state startup failed: prepared=%v summary=%+v IDs=%v error=%v", prepared, summary, ids, err)
		}
		finalized, err := state.Summary()
		if err != nil || !reflect.DeepEqual(finalized, summary) {
			t.Fatalf("state disagrees with accepted CUDA summary: %+v %+v %v", finalized, summary, err)
		}
	}
}
