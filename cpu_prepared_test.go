package quantaspindle

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
)

func preparedCPUConfig() Config {
	c := tinyConfig()
	c.Assets = []Asset{{InitialPrice: 100, Mu: .03, Sigma: .2}, {InitialPrice: 80, Mu: -.02, Sigma: .4}, {InitialPrice: 150, Mu: .04, Sigma: .1}}
	c.Correlation = [][]float64{{1, .3, -.2}, {.3, 1, .1}, {-.2, .1, 1}}
	c.Regimes = []Regime{{Mu: []float64{.02, .03, -.04}, Sigma: []float64{.2, .3, .4}}, {Mu: []float64{-.1, -.2, -.3}, Sigma: []float64{.5, .6, .7}}}
	c.Transition = [][]float64{{.8, .2}, {.3, .7}}
	c.InitialRegime = 1
	c.StorePaths = true
	return c
}

func ExampleCPU_Prepare() {
	c := DefaultConfig()
	c.Paths, c.Steps = 100, 1
	p, err := (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		panic(err)
	}
	for _, first := range []uint64{80, 3, 42} {
		batch, err := p.Batch(context.Background(), first, 1)
		if err != nil {
			panic(err)
		}
		fmt.Println(batch[0].ID)
	}
	// Output:
	// 80
	// 3
	// 42
}

func TestPreparedCPUArbitraryBatchesReplay(t *testing.T) {
	for name, correlation := range map[string][][]float64{
		"independent": nil,
		"correlated":  {{1, .3, -.2}, {.3, 1, .1}, {-.2, .1, 1}},
		"singular":    {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		"precise":     nearSingularCorrelation(),
	} {
		for _, store := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/paths=%t", name, store), func(t *testing.T) {
				c := preparedCPUConfig()
				c.Correlation, c.StorePaths = correlation, store
				p, err := (CPU{}).Prepare(context.Background(), c)
				if err != nil {
					t.Fatal(err)
				}
				all, _ := collect(t, c)
				for _, bounds := range [][2]int{{11, 2}, {0, 1}, {5, 4}, {18, 1}, {5, 4}} {
					got, err := p.Batch(context.Background(), uint64(bounds[0]), bounds[1])
					if err != nil {
						t.Fatal(err)
					}
					direct, err := (CPU{}).Batch(context.Background(), c, uint64(bounds[0]), bounds[1])
					if err != nil || !reflect.DeepEqual(got, direct) || !reflect.DeepEqual(got, all[bounds[0]:bounds[0]+bounds[1]]) {
						t.Fatalf("batch %v differed from CPU.Batch or Run: %v", bounds, err)
					}
					for _, s := range got {
						if err := validateResult(s, s.ID, c); err != nil {
							t.Fatal(err)
						}
					}
				}
			})
		}
	}
}

func TestPreparedCPUSnapshotAndResultOwnership(t *testing.T) {
	c := preparedCPUConfig()
	want, err := (CPU{}).Batch(context.Background(), c, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	p, err := (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	// Mutate every slice type, including nested regime/transition rows, after
	// preparation. The prepared model must neither retain nor reread them.
	c.Assets[0] = Asset{}
	c.Correlation[0][0] = math.NaN()
	c.Correlation[1] = nil
	c.Regimes[0].Mu[0] = math.NaN()
	c.Regimes[1].Sigma[0] = -1
	c.Regimes[0] = Regime{}
	c.Transition[0][0] = 0
	c.Transition[1] = nil
	c.Paths, c.Steps, c.DT, c.Seed, c.StorePaths = 1, 1, 1, 999, false
	got, err := p.Batch(context.Background(), 4, 2)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("caller mutation changed snapshot: %v", err)
	}
	s := &got[0]
	s.FinalPrices[0], s.Returns[0], s.RealizedVolatility[0], s.MaxDrawdown[0] = -1, -1, -1, -1
	s.Prices[0][0], s.Regimes[0] = -1, -1
	if !reflect.DeepEqual(got[1], want[1]) {
		t.Fatal("scenarios share result storage")
	}
	again, err := p.Batch(context.Background(), 4, 2)
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatalf("results share storage with another call or the snapshot: %v", err)
	}
}

func TestPreparedCPUConcurrentBatches(t *testing.T) {
	c := preparedCPUConfig()
	p, err := (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	all, _ := collect(t, c)
	var wg sync.WaitGroup
	for _, first := range []uint64{13, 1, 7, 13, 2, 0, 7, 17} {
		wg.Go(func() {
			got, err := p.Batch(context.Background(), first, 2)
			if err != nil || !reflect.DeepEqual(got, all[first:first+2]) {
				t.Errorf("concurrent batch %d: %v", first, err)
				return
			}
			// Other simultaneous calls may overlap the same scenario IDs.
			got[0].FinalPrices[0] = -1
			got[0].Prices[0][0] = -1
		})
	}
	wg.Wait()
}

func TestPreparedCPUValidation(t *testing.T) {
	c := tinyConfig()
	if p, err := (CPU{}).Prepare(nil, c); err == nil || p != nil {
		t.Fatal("accepted nil preparation context")
	}
	for _, matrix := range [][][]float64{
		{{1, .5}, {0, 1}},                         // asymmetric
		{{1, .9, .9}, {.9, 1, -.9}, {.9, -.9, 1}}, // indefinite
	} {
		bad := c
		bad.Assets = make([]Asset, len(matrix))
		for i := range bad.Assets {
			bad.Assets[i] = c.Assets[0]
		}
		bad.Correlation = matrix
		if p, err := (CPU{}).Prepare(context.Background(), bad); err == nil || p != nil {
			t.Fatal("accepted invalid correlation")
		}
	}
	p, err := (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Batch(nil, 0, 1); err == nil {
		t.Fatal("accepted nil batch context")
	}
	for _, bounds := range []struct {
		first uint64
		count int
	}{{0, 0}, {0, -1}, {0, 16385}, {19, 1}, {18, 2}, {math.MaxUint64, 1}} {
		if out, err := p.Batch(context.Background(), bounds.first, bounds.count); err == nil || out != nil {
			t.Fatalf("accepted invalid batch %+v", bounds)
		}
	}
	for _, zero := range []*PreparedCPU{nil, {}} {
		if _, err := zero.Batch(context.Background(), 0, 1); err == nil {
			t.Fatal("accepted uninitialized prepared CPU")
		}
	}
	// A valid small configured wave does not authorize an oversized direct batch.
	c.Paths, c.Workers, c.BatchSize, c.Steps, c.StorePaths = 100, 1, 1, 1_000_000, true
	p, err = (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := p.Batch(context.Background(), 0, 100); err == nil || out != nil {
		t.Fatal("accepted batch exceeding storage limit")
	}
	c = onePathConfig()
	c.DT, c.Assets[0].Mu, c.Assets[0].Sigma = 1, 1000, 0
	p, err = (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	if out, err := p.Batch(context.Background(), 0, 1); err == nil || out != nil {
		t.Fatal("accepted numerical overflow")
	}
}

// Cancel on a selected error check to exercise cancellation within simulation
// deterministically, after its first 64 steps rather than a timing-based sleep.
type cancelCPUContext struct {
	context.Context
	checks atomic.Int32
	cancel context.CancelFunc
}

func (c *cancelCPUContext) Err() error {
	if c.checks.Add(-1) == 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestPreparedCPUCancellationIsPerCall(t *testing.T) {
	c := tinyConfig()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if p, err := (CPU{}).Prepare(ctx, c); !errors.Is(err, context.Canceled) || p != nil {
		t.Fatalf("preparation ignored cancellation: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	p, err := (CPU{}).Prepare(ctx, c)
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	if out, err := p.Batch(ctx, 0, 1); !errors.Is(err, context.Canceled) || out != nil {
		t.Fatalf("batch ignored cancellation: %v", err)
	}
	if _, err := p.Batch(context.Background(), 0, 1); err != nil {
		t.Fatalf("preparation context retained after return: %v", err)
	}
	c.Steps = 128
	p, err = (CPU{}).Prepare(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	active := &cancelCPUContext{Context: ctx, cancel: cancel}
	active.checks.Store(3)
	if out, err := p.Batch(active, 4, 2); !errors.Is(err, context.Canceled) || out != nil {
		t.Fatalf("mid-batch cancellation returned %v, %v", out, err)
	}
	if _, err := p.Batch(context.Background(), 4, 2); err != nil {
		t.Fatalf("canceled batch invalidated prepared CPU: %v", err)
	}
}

func BenchmarkCPURandomAccessBatches(b *testing.B) {
	// Each operation requests the same sixteen one-scenario, noncontiguous
	// batches. Separate reused and prepare-plus-batches cases expose setup cost.
	c := DefaultConfig()
	c.Paths, c.Steps, c.Workers, c.BatchSize = 32, 1, 1, 1
	c.Assets = make([]Asset, 128)
	c.Correlation = make([][]float64, 128)
	for i := range c.Assets {
		c.Assets[i] = Asset{InitialPrice: 100, Mu: .01, Sigma: .2}
		c.Correlation[i] = make([]float64, 128)
		c.Correlation[i][i] = 1
	}
	ctx := context.Background()
	for _, mode := range []string{"CPU.Batch", "prepared_reused", "prepare_and_batches"} {
		b.Run(mode, func(b *testing.B) {
			p, err := (CPU{}).Prepare(ctx, c)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if mode == "prepare_and_batches" {
					p, err = (CPU{}).Prepare(ctx, c)
					if err != nil {
						b.Fatal(err)
					}
				}
				for batch := 0; batch < 16; batch++ {
					first := uint64(30 - 2*batch)
					if mode == "CPU.Batch" {
						_, err = (CPU{}).Batch(ctx, c, first, 1)
					} else {
						_, err = p.Batch(ctx, first, 1)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}
