package quantaspindle

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"reflect"
	"strings"
	"testing"
)

type stateReference struct {
	count uint64
	sums  []big.Rat
}

func (r *stateReference) add(s Scenario) {
	r.count++
	if r.sums == nil {
		r.sums = make([]big.Rat, 3*len(s.Returns))
	}
	for metric, values := range [][]float64{s.Returns, s.RealizedVolatility, s.MaxDrawdown} {
		for i, value := range values {
			r.sums[metric*len(values)+i].Add(&r.sums[metric*len(values)+i], new(big.Rat).SetFloat64(value))
		}
	}
}

func checkStateReference(t *testing.T, summary RunSummary, ref *stateReference) {
	t.Helper()
	if summary.Scenarios != ref.count {
		t.Fatalf("count %d, want %d", summary.Scenarios, ref.count)
	}
	for metric, values := range [][]float64{summary.MeanReturns, summary.MeanRealizedVolatility, summary.MeanMaxDrawdown} {
		for i, got := range values {
			exact := new(big.Rat).Quo(&ref.sums[metric*len(values)+i], new(big.Rat).SetInt(new(big.Int).SetUint64(ref.count)))
			want, _ := exact.Float64()
			// Predeclared tolerance: eight adjacent-float steps for normal
			// results, exact rounding for subnormal means and zero.
			tolerance := 0.0
			if math.Abs(want) >= 0x1p-1022 {
				tolerance = 8 * math.Abs(want-math.Nextafter(want, 0))
			}
			if !finite(got) || math.Abs(got-want) > tolerance {
				t.Fatalf("metric %d asset %d mean %.17g, rational reference %.17g (tolerance %g)", metric, i, got, want, tolerance)
			}
		}
	}
}

func transportState(t *testing.T, state *RunSummaryState) *RunSummaryState {
	t.Helper()
	data, err := state.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var restored RunSummaryState
	if err := restored.UnmarshalBinary(data); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state, &restored) {
		t.Fatal("transport changed compensated components or provenance")
	}
	return &restored
}

func TestRunSummaryStateUnequalCPUShards(t *testing.T) {
	c := tinyConfig()
	c.Paths, c.Steps = 137, 24
	c.Assets = []Asset{{Name: "first", InitialPrice: 100, Mu: -.1, Sigma: .3}, {Name: "second", InitialPrice: 23, Mu: .2, Sigma: .1}}
	var ref stateReference
	direct, err := Run(context.Background(), c, CPU{}, func(s Scenario) error { ref.add(s); return nil })
	if err != nil {
		t.Fatal(err)
	}
	checkStateReference(t, direct, &ref)
	var shards []*RunSummaryState
	for i, bounds := range [][2]uint64{{0, 1}, {1, 4}, {5, 32}, {37, 100}} {
		c.Workers, c.BatchSize = i+1, i*5+1
		summary, state, err := RunRangeWithState(context.Background(), c, CPU{}, bounds[0], bounds[1], "cpu-fixture/toolchain-v1", nil)
		if err != nil {
			t.Fatal(err)
		}
		finalized, err := state.Summary()
		if err != nil || !reflect.DeepEqual(finalized, summary) {
			t.Fatalf("range state diverged from returned summary: %+v %+v %v", finalized, summary, err)
		}
		shards = append(shards, transportState(t, state))
	}
	for _, order := range [][]int{{0, 1, 2, 3}, {3, 2, 1, 0}} {
		merged, err := NewRunSummaryState(c, (CPU{}).Name(), "cpu-fixture/toolchain-v1")
		if err != nil {
			t.Fatal(err)
		}
		for _, i := range order {
			if err := merged.Merge(shards[i]); err != nil {
				t.Fatal(err)
			}
		}
		got, err := merged.Summary()
		if err != nil {
			t.Fatal(err)
		}
		checkStateReference(t, got, &ref)
		if !reflect.DeepEqual(merged.Ranges(), []ScenarioRange{{StartID: 0, Count: 137}}) {
			t.Fatal("adjacent ranges did not coalesce", merged.Ranges())
		}
		ranges := merged.Ranges()
		ranges[0].Count = 0
		got.MeanReturns[0] = math.NaN()
		again, _ := merged.Summary()
		checkStateReference(t, again, &ref)
	}
	left, right := transportState(t, shards[0]), transportState(t, shards[2])
	if err := left.Merge(shards[1]); err != nil {
		t.Fatal(err)
	}
	if err := right.Merge(shards[3]); err != nil {
		t.Fatal(err)
	}
	if err := left.Merge(right); err != nil {
		t.Fatal(err)
	}
	tree, _ := left.Summary()
	checkStateReference(t, tree, &ref)
}

func TestRunSummaryStateCompensationAcrossScales(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	for name, returns := range map[string][]float64{
		"cancellation": {.5, 1.7763568394002506e-16, -.5},
		"overflow":     {math.MaxFloat64, math.MaxFloat64, 0},
		"subnormal":    {tiny, tiny, 2 * tiny},
		"signed tiny":  {-1e-300, 2e-300, 3e-300},
	} {
		t.Run(name, func(t *testing.T) {
			c := onePathConfig()
			c.Paths = 1_000_000_000 // Deliberately much larger than each shard.
			values := make([]Scenario, len(returns))
			var ref stateReference
			for i, value := range returns {
				values[i] = Scenario{FinalPrices: []float64{1}, Returns: []float64{value}, RealizedVolatility: []float64{math.Abs(value)}, MaxDrawdown: []float64{math.Min(math.Abs(value), 1)}}
				ref.add(values[i])
			}
			backend := precisionScenarios(values)
			_, first, err := RunRangeWithState(context.Background(), c, backend, 0, 2, "numerics-v1", nil)
			if err != nil {
				t.Fatal(err)
			}
			_, second, err := RunRangeWithState(context.Background(), c, backend, 2, 1, "numerics-v1", nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, reversed := range []bool{false, true} {
				a, b := transportState(t, first), transportState(t, second)
				if reversed {
					a, b = b, a
				}
				if err := a.Merge(b); err != nil {
					t.Fatal(err)
				}
				got, _ := a.Summary()
				checkStateReference(t, got, &ref)
			}
		})
	}
}

func TestRunSummaryStateAcceptedPrefixAndOwnership(t *testing.T) {
	c := tinyConfig()
	c.Paths, c.Workers, c.BatchSize = 20, 1, 1
	for _, failure := range []string{"sink", "cancellation", "backend", "invalid scenario", "bad batch"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			sentinel := errors.New("stop shard")
			backend := precisionBackend(func(ctx context.Context, c Config, first uint64, count int) ([]Scenario, error) {
				if first == 6 && failure == "backend" {
					return nil, sentinel
				}
				out, err := (CPU{}).Batch(ctx, c, first, count)
				if first == 6 && failure == "invalid scenario" {
					out[0].Returns[0] = math.NaN()
				}
				if first == 6 && failure == "bad batch" {
					return nil, nil
				}
				return out, err
			})
			var ref stateReference
			summary, state, err := RunRangeWithState(ctx, c, backend, 4, 8, "partial-v1", func(s Scenario) error {
				if s.ID == 6 && failure == "sink" {
					return sentinel
				}
				ref.add(s)
				if s.ID == 5 && failure == "cancellation" {
					cancel()
				}
				s.Returns[0], s.RealizedVolatility[0], s.MaxDrawdown[0] = math.NaN(), math.NaN(), math.NaN()
				return nil
			})
			if err == nil || state.Count() != 2 || !reflect.DeepEqual(state.Ranges(), []ScenarioRange{{StartID: 4, Count: 2}}) {
				t.Fatalf("wrong accepted prefix: %+v %v", state, err)
			}
			if (failure == "sink" || failure == "backend") && !errors.Is(err, sentinel) {
				t.Fatal("original failure lost", err)
			}
			if failure == "cancellation" && !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation lost", err)
			}
			checkStateReference(t, summary, &ref)
			finalized, stateErr := transportState(t, state).Summary()
			if stateErr != nil {
				t.Fatal(stateErr)
			}
			checkStateReference(t, finalized, &ref)
			// A retry of the unaccepted suffix may merge despite overlap with
			// the originally requested (but not accepted) range.
			_, retry, err := RunRangeWithState(context.Background(), c, backend, 6, 6, "partial-v1", nil)
			if failure == "backend" || failure == "invalid scenario" || failure == "bad batch" {
				// These persistent fixture failures intentionally accept no retry.
				if err == nil || retry.Count() != 0 {
					t.Fatal("failed retry unexpectedly accepted observations")
				}
			} else if err != nil || state.Merge(retry) != nil || state.Count() != 8 || !reflect.DeepEqual(state.Ranges(), []ScenarioRange{{4, 8}}) {
				t.Fatalf("retry provenance failed: %+v %v", state, err)
			}
		})
	}
}

func TestRunSummaryStateEmptyAndCompatibility(t *testing.T) {
	c := tinyConfig()
	c.Paths = 10
	c.Assets = []Asset{{Name: "a", InitialPrice: 1}, {Name: "b", InitialPrice: 2}}
	empty, err := NewRunSummaryState(c, (CPU{}).Name(), "scope")
	if err != nil {
		t.Fatal(err)
	}
	zero, err := empty.Summary()
	if err != nil || zero.Scenarios != 0 || len(zero.MeanReturns) != 2 || len(empty.Ranges()) != 0 {
		t.Fatalf("invalid empty identity: %+v %v", zero, err)
	}
	if err := empty.Merge(transportState(t, empty)); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, canceled, err := RunRangeWithState(ctx, c, CPU{}, 2, 3, "scope", nil)
	if !errors.Is(err, context.Canceled) || canceled.Count() != 0 || len(canceled.Ranges()) != 0 || empty.Merge(canceled) != nil {
		t.Fatal("pre-cancellation empty state", err)
	}
	_, full, err := RunRangeWithState(context.Background(), c, CPU{}, 2, 3, "scope", nil)
	if err != nil || empty.Merge(full) != nil {
		t.Fatal(err)
	}
	before, _ := empty.MarshalBinary()
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Assets = c.Assets[:1] },
		func(c *Config) { c.Assets = []Asset{c.Assets[1], c.Assets[0]} },
		func(c *Config) { c.Seed++ },
		func(c *Config) { c.Steps++ },
		func(c *Config) { c.Paths++ },
	} {
		changed := c
		mutate(&changed)
		other, err := NewRunSummaryState(changed, (CPU{}).Name(), "scope")
		if err != nil || empty.Merge(other) == nil {
			t.Fatalf("incompatible configuration accepted: %v", err)
		}
	}
	wrongScope, _ := NewRunSummaryState(c, (CPU{}).Name(), "another-job")
	wrongBackend, _ := NewRunSummaryState(c, "cuda", "scope")
	malformed := transportState(t, full)
	malformed.means = malformed.means[:len(malformed.means)-1]
	for _, other := range []*RunSummaryState{nil, {}, wrongScope, wrongBackend, malformed, full, empty} {
		if empty.Merge(other) == nil {
			t.Fatal("incompatible, malformed or overlapping state accepted")
		}
		after, _ := empty.MarshalBinary()
		if string(after) != string(before) {
			t.Fatal("rejected merge changed destination")
		}
	}
	if _, _, err := RunRangeWithState(context.Background(), c, CPU{}, 0, 1, "", nil); err == nil {
		t.Fatal("missing explicit scope accepted")
	}
}

func TestRunSummaryStateTransportValidationIsAtomic(t *testing.T) {
	c := tinyConfig()
	_, state, err := RunRangeWithState(context.Background(), c, CPU{}, 3, 5, "transport/v1", nil)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := state.MarshalBinary()
	for name, mutate := range map[string]func(*summaryStateWire){
		"version":           func(w *summaryStateWire) { w.Version++ },
		"asset count":       func(w *summaryStateWire) { w.Assets++ },
		"metric dimensions": func(w *summaryStateWire) { w.Means = w.Means[:len(w.Means)-1] },
		"empty metrics":     func(w *summaryStateWire) { w.Count = 0; w.Ranges = nil },
		"scope":             func(w *summaryStateWire) { w.Scope = "" },
		"scale":             func(w *summaryStateWire) { w.Means[0].Exponent++ },
		"range count":       func(w *summaryStateWire) { w.Ranges[0].Count-- },
		"range overflow":    func(w *summaryStateWire) { w.Ranges[0].StartID = math.MaxUint64 },
		"count overflow":    func(w *summaryStateWire) { w.Count = math.MaxUint64 },
		"overlap":           func(w *summaryStateWire) { w.Ranges = append(w.Ranges, w.Ranges[0]) },
	} {
		t.Run(name, func(t *testing.T) {
			var wire summaryStateWire
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}
			mutate(&wire)
			invalid, _ := json.Marshal(wire)
			if err := state.UnmarshalBinary(invalid); err == nil {
				t.Fatal("malformed state accepted")
			}
			after, _ := state.MarshalBinary()
			if string(after) != string(data) {
				t.Fatal("failed decode mutated destination")
			}
		})
	}
	for _, invalid := range [][]byte{data[:len(data)-1], append(append([]byte(nil), data...), []byte(" {}")...), []byte(`{"Version":1,"Unknown":true}`), []byte(`null`), []byte(strings.Replace(string(data), "transport/v1", `\ud800`, 1)), append([]byte{0xff}, data...)} {
		if err := state.UnmarshalBinary(invalid); err == nil {
			t.Fatal("malformed/trailing transport accepted")
		}
	}
}

func TestRunSummaryStateIdentityIsLossless(t *testing.T) {
	c := tinyConfig()
	for _, malformed := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
		if _, err := NewRunSummaryState(c, "cpu", malformed); err == nil {
			t.Fatal("invalid UTF-8 scope accepted")
		}
		if _, err := NewRunSummaryState(c, malformed, "scope"); err == nil {
			t.Fatal("invalid UTF-8 backend accepted")
		}
		changed := c
		changed.Assets = append([]Asset(nil), c.Assets...)
		changed.Assets[0].Name = malformed
		if _, err := NewRunSummaryState(changed, "cpu", "scope"); err == nil {
			t.Fatal("invalid UTF-8 asset identity accepted")
		}
		changed = DefaultRegimeConfig()
		changed.Regimes[0].Name = malformed
		if _, err := NewRunSummaryState(changed, "cpu", "scope"); err == nil {
			t.Fatal("invalid UTF-8 regime identity accepted")
		}
	}
	c.Assets[0].Name = "資産-α"
	s, err := NewRunSummaryState(c, "計算-cpu", "実験/é/🙂")
	if err != nil {
		t.Fatal(err)
	}
	restored := transportState(t, s)
	if restored.scope != s.scope || restored.backend != s.backend || restored.fingerprint != s.fingerprint {
		t.Fatal("valid Unicode identity changed during transport")
	}
}

func TestRunSummaryStateLargeUnequalShardsBoundedStorage(t *testing.T) {
	c := onePathConfig()
	c.Paths, c.BatchSize = 100_001, 1024
	backend := precisionBackend(func(_ context.Context, c Config, first uint64, count int) ([]Scenario, error) {
		out := make([]Scenario, count)
		for i := range out {
			id := first + uint64(i)
			value := float64(int(id%3)-1) * .5
			out[i] = Scenario{ID: id, FinalPrices: []float64{1}, Returns: []float64{value}, RealizedVolatility: []float64{math.Abs(value)}, MaxDrawdown: []float64{math.Max(-value, 0)}}
		}
		return out, nil
	})
	var ref stateReference
	var merged *RunSummaryState
	for _, bounds := range [][2]uint64{{0, 1}, {1, 100_000}} {
		_, state, err := RunRangeWithState(context.Background(), c, backend, bounds[0], bounds[1], "large/v1", func(s Scenario) error { ref.add(s); return nil })
		if err != nil {
			t.Fatal(err)
		}
		if len(state.means) != 3 || len(state.ranges) != 1 {
			t.Fatal("state grew with observation count")
		}
		if merged == nil {
			merged = state
		} else if err := merged.Merge(state); err != nil {
			t.Fatal(err)
		}
	}
	got, err := merged.Summary()
	if err != nil {
		t.Fatal(err)
	}
	checkStateReference(t, got, &ref)
	encoded, err := merged.MarshalBinary()
	if err != nil || len(encoded) > 4096 || len(merged.ranges) != 1 || len(merged.means) != 3 {
		t.Fatalf("bounded state exceeded expected small footprint: %d bytes %v", len(encoded), err)
	}
	if got.Scenarios != 100_001 {
		t.Fatalf("wrong large shard count: %d", got.Scenarios)
	}
}

type stateLifecycleBackend struct {
	CPU
	beginErr, closeErr error
}

func (b stateLifecycleBackend) BeginRun(context.Context, Config) (BackendSession, error) {
	if b.beginErr != nil {
		return nil, b.beginErr
	}
	return &stateLifecycleSession{closeErr: b.closeErr}, nil
}

type stateLifecycleSession struct {
	CPU
	closeErr error
}

func (s *stateLifecycleSession) Close() error { return s.closeErr }

func TestRunSummaryStateLifecycleFailures(t *testing.T) {
	c := tinyConfig()
	sentinel := errors.New("session lifecycle failed")
	for _, begin := range []bool{false, true} {
		backend := stateLifecycleBackend{}
		if begin {
			backend.beginErr = sentinel
		} else {
			backend.closeErr = sentinel
		}
		summary, state, err := RunRangeWithState(context.Background(), c, backend, 4, 5, "lifecycle/v1", nil)
		wantCount := uint64(5)
		if begin {
			wantCount = 0
		}
		if !errors.Is(err, sentinel) || state.Count() != wantCount || summary.Scenarios != wantCount {
			t.Fatalf("wrong lifecycle prefix: %+v %v", state, err)
		}
		if _, err := transportState(t, state).Summary(); err != nil {
			t.Fatal(err)
		}
	}
}
