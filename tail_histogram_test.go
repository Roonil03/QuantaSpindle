package quantaspindle

import (
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"testing"
)

func newTestHistogram(t *testing.T, lower, upper float64, bins int, losses ...float64) *LossHistogram {
	t.Helper()
	h, err := NewLossHistogram(lower, upper, bins)
	if err != nil {
		t.Fatal(err)
	}
	for _, loss := range losses {
		if err := h.Add(loss); err != nil {
			t.Fatal(err)
		}
	}
	return h
}

func checkHistogramWidths(t *testing.T, r HistogramRiskSummary) {
	t.Helper()
	if !finite(r.BinWidth) || r.BinWidth <= 0 || !finite(r.RoundingAllowance) || r.RoundingAllowance < 0 {
		t.Fatalf("invalid interval-width metadata: %+v", r)
	}
	for i, interval := range []RiskInterval{r.VaR, r.ExpectedShortfall} {
		if !finite(interval.Lower) || !finite(interval.Upper) || interval.Lower > interval.Upper {
			t.Fatalf("invalid interval %+v", interval)
		}
		width := new(big.Rat).Sub(new(big.Rat).SetFloat64(interval.Upper), new(big.Rat).SetFloat64(interval.Lower))
		limit := new(big.Rat).SetFloat64(r.BinWidth)
		if i == 1 {
			limit.Add(limit, new(big.Rat).SetFloat64(r.RoundingAllowance))
		}
		if width.Cmp(limit) > 0 {
			t.Fatalf("interval %+v width exceeds bin width %g plus allowance %g", interval, r.BinWidth, r.RoundingAllowance)
		}
	}
}

func checkHistogramContains(t *testing.T, r HistogramRiskSummary, losses []float64) {
	t.Helper()
	x := append([]float64(nil), losses...)
	sort.Float64s(x)
	// Independent rank calculation for the small samples in these tests.
	rank := max(1, int(math.Ceil(r.Confidence*float64(len(x))-1e-12)))
	tail := max(1, int(math.Ceil((1-r.Confidence)*float64(len(x))-1e-12)))
	if r.Count != uint64(len(x)) || r.TailCount != uint64(tail) || x[rank-1] < r.VaR.Lower || x[rank-1] > r.VaR.Upper {
		t.Fatalf("wrong rank/count for %v: %+v", losses, r)
	}
	mean := new(big.Rat)
	for _, loss := range x[len(x)-tail:] {
		mean.Add(mean, new(big.Rat).SetFloat64(loss))
	}
	mean.Quo(mean, big.NewRat(int64(tail), 1))
	if new(big.Rat).SetFloat64(r.ExpectedShortfall.Lower).Cmp(mean) > 0 || new(big.Rat).SetFloat64(r.ExpectedShortfall.Upper).Cmp(mean) < 0 {
		t.Fatalf("ES interval %+v excludes rational mean %s for %v", r.ExpectedShortfall, mean, losses)
	}
	checkHistogramWidths(t, r)
}

func TestLossHistogramExhaustiveSixLossVectors(t *testing.T) {
	// Independently enumerate every six-loss vector over {-2,0,3}: 3^6=729.
	// Nine confidence levels exercise tails including fractional tail counts.
	alphabet := []float64{-2, 0, 3}
	for vector := 0; vector < 729; vector++ {
		losses := make([]float64, 6)
		code := vector
		for i := range losses {
			losses[i] = alphabet[code%3]
			code /= 3
		}
		for _, bins := range []int{1, 2, 5, 7} {
			h := newTestHistogram(t, -2, 3, bins, losses...)
			for confidence := 1; confidence <= 9; confidence++ {
				alpha := float64(confidence) / 10
				r, err := h.Risk(alpha)
				if err != nil {
					t.Fatal(err)
				}
				exact, err := RiskFromLosses(losses, alpha)
				if err != nil {
					t.Fatal(err)
				}
				if exact.VaR < r.VaR.Lower || exact.VaR > r.VaR.Upper || exact.ExpectedShortfall < r.ExpectedShortfall.Lower || exact.ExpectedShortfall > r.ExpectedShortfall.Upper || uint64(exact.TailCount) != r.TailCount {
					t.Fatalf("vector %d bins %d confidence %g: exact %+v outside %+v", vector, bins, alpha, exact, r)
				}
				checkHistogramContains(t, r, losses)
			}
		}
	}
}

func TestLossHistogramBoundariesAndTies(t *testing.T) {
	h := newTestHistogram(t, -2, 3, 5)
	edges := h.Edges()
	if !reflect.DeepEqual(edges, []float64{-2, -1, 0, 1, 2, 3}) {
		t.Fatal("unexpected edges", edges)
	}
	edges[0] = 100
	if h.Edges()[0] != -2 {
		t.Fatal("Edges exposes mutable storage")
	}
	losses := []float64{-2, -1, -1, math.Nextafter(0, -1), 0, 0, 1, 2, 3, 3}
	for _, loss := range losses {
		if err := h.Add(loss); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(h.counts, []uint64{1, 3, 2, 1, 3}) {
		t.Fatal("wrong half-open bins or endpoint assignment", h.counts)
	}
	for _, alpha := range []float64{.1, .3, .5, .7, .9, math.Nextafter(.7, 0), math.Nextafter(.7, 1)} {
		r, err := h.Risk(alpha)
		if err != nil {
			t.Fatal(err)
		}
		checkHistogramContains(t, r, losses)
		exact, err := RiskFromLosses(losses, alpha)
		if err != nil || uint64(exact.TailCount) != r.TailCount || exact.VaR < r.VaR.Lower || exact.VaR > r.VaR.Upper {
			t.Fatalf("boundary rank mismatch: %+v %+v %v", exact, r, err)
		}
	}
	for _, values := range [][]float64{{-.2, -.2, -.1}, {-2, -2, -2}, {3, 3, 3}, {0}} {
		h := newTestHistogram(t, -2, 3, 5, values...)
		r, err := h.Risk(.5)
		if err != nil {
			t.Fatal(err)
		}
		checkHistogramContains(t, r, values)
	}
}

func TestLossHistogramExtremeFiniteIntervals(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	next := math.Nextafter(1, 2)
	for _, tc := range []struct {
		lower, upper float64
		bins         int
		losses       []float64
	}{
		{-math.MaxFloat64, math.MaxFloat64, 2, []float64{-math.MaxFloat64, 0, math.MaxFloat64}},
		{-math.MaxFloat64, math.MaxFloat64, 3, []float64{-math.MaxFloat64, math.MaxFloat64, math.MaxFloat64}},
		{math.Nextafter(math.MaxFloat64, 0), math.MaxFloat64, 1, []float64{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64}},
		{-8 * tiny, 8 * tiny, 8, []float64{-7 * tiny, 0, 7 * tiny}},
		{1, next, 1, []float64{1, next, next}},
		{-1e308, 1e308, 4, []float64{-1e308, 3, 1e308}},
	} {
		t.Run(fmt.Sprintf("%g/%g/%d", tc.lower, tc.upper, tc.bins), func(t *testing.T) {
			h := newTestHistogram(t, tc.lower, tc.upper, tc.bins, tc.losses...)
			edges := h.Edges()
			// Compare every rounded edge against independent exact interpolation.
			for i, edge := range edges {
				fraction := big.NewRat(int64(i), int64(tc.bins))
				left := new(big.Rat).Mul(new(big.Rat).Sub(big.NewRat(1, 1), fraction), new(big.Rat).SetFloat64(tc.lower))
				right := new(big.Rat).Mul(fraction, new(big.Rat).SetFloat64(tc.upper))
				want, _ := new(big.Rat).Add(left, right).Float64()
				if edge != want || (i > 0 && edge <= edges[i-1]) {
					t.Fatalf("edge %d = %g, want %g", i, edge, want)
				}
			}
			for _, confidence := range []float64{.1, .5, .95} {
				r, err := h.Risk(confidence)
				if err != nil {
					t.Fatal(err)
				}
				checkHistogramContains(t, r, tc.losses)
			}
		})
	}
}

func TestLossHistogramRejectsInvalidInputsAtomically(t *testing.T) {
	for _, tc := range []struct {
		lower, upper float64
		bins         int
	}{
		{0, 1, 0}, {0, 1, -1}, {0, 1, 1_000_001}, {0, 1, int(^uint(0) >> 1)},
		{1, 1, 1}, {2, 1, 1}, {math.NaN(), 1, 1}, {0, math.Inf(1), 1},
		{math.Inf(-1), 0, 1}, {0, math.NaN(), 1},
		{-math.MaxFloat64, math.MaxFloat64, 1},
		{1, math.Nextafter(1, 2), 2}, {0, math.SmallestNonzeroFloat64, 2},
	} {
		if _, err := NewLossHistogram(tc.lower, tc.upper, tc.bins); err == nil {
			t.Fatalf("accepted unsafe configuration %+v", tc)
		}
	}
	for _, h := range []*LossHistogram{nil, {}} {
		if h.Add(0) == nil || h.Merge(newTestHistogram(t, -1, 1, 2)) == nil {
			t.Fatal("uninitialized histogram accepted mutation")
		}
		if _, err := h.Risk(.95); err == nil {
			t.Fatal("uninitialized histogram accepted query")
		}
	}
	h := newTestHistogram(t, -1, 1, 2)
	if _, err := h.Risk(.5); err == nil {
		t.Fatal("empty histogram accepted query")
	}
	if err := h.Add(0); err != nil {
		t.Fatal(err)
	}
	before := append([]uint64(nil), h.counts...)
	for _, loss := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Nextafter(-1, math.Inf(-1)), math.Nextafter(1, math.Inf(1))} {
		if h.Add(loss) == nil || h.Count() != 1 || !reflect.DeepEqual(h.counts, before) {
			t.Fatal("invalid loss accepted or state changed", loss)
		}
	}
	for _, alpha := range []float64{0, 1, -1, math.NaN(), math.Inf(1)} {
		if _, err := h.Risk(alpha); err == nil {
			t.Fatal("invalid confidence accepted", alpha)
		}
	}
	for _, other := range []*LossHistogram{nil, {}, newTestHistogram(t, -2, 1, 2), newTestHistogram(t, -1, 1, 3)} {
		if h.Merge(other) == nil || h.Count() != 1 || !reflect.DeepEqual(h.counts, before) {
			t.Fatal("incompatible merge accepted or state changed")
		}
	}
}

func TestLossHistogramMergeOrderAndCountOverflow(t *testing.T) {
	losses := []float64{-2, 0, 3, 3, -2, 0, .1, -1, 2}
	full := newTestHistogram(t, -2, 3, 7, losses...)
	parts := []*LossHistogram{
		newTestHistogram(t, -2, 3, 7, losses[:3]...),
		newTestHistogram(t, -2, 3, 7, losses[3:6]...),
		newTestHistogram(t, -2, 3, 7, losses[6:]...),
	}
	want, _ := full.Risk(.7)
	for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		merged := newTestHistogram(t, -2, 3, 7)
		for _, i := range order {
			if err := merged.Merge(parts[i]); err != nil {
				t.Fatal(err)
			}
		}
		got, _ := merged.Risk(.7)
		if got != want || !reflect.DeepEqual(merged.counts, full.counts) || parts[0].Count() != 3 {
			t.Fatalf("merge order %v changed results: %+v want %+v", order, got, want)
		}
	}
	// Reach MaxUint64 using public operations, without allocating observations.
	one := newTestHistogram(t, -1, 1, 2, 0)
	maximum := newTestHistogram(t, -1, 1, 2, 0)
	for bit := 1; bit < 64; bit++ {
		if err := maximum.Merge(maximum); err != nil {
			t.Fatal(err)
		}
		if err := maximum.Merge(one); err != nil {
			t.Fatal(err)
		}
	}
	if maximum.Count() != math.MaxUint64 {
		t.Fatal("wrong maximum count", maximum.Count())
	}
	before := append([]uint64(nil), maximum.counts...)
	for _, operation := range []func() error{func() error { return maximum.Add(0) }, func() error { return maximum.Merge(one) }, func() error { return maximum.Merge(maximum) }} {
		if operation() == nil || maximum.Count() != math.MaxUint64 || !reflect.DeepEqual(maximum.counts, before) {
			t.Fatal("count overflow accepted or mutation was not atomic")
		}
	}
	for _, alpha := range []float64{math.SmallestNonzeroFloat64, .5, math.Nextafter(1, 0)} {
		r, err := maximum.Risk(alpha)
		if err != nil || r.TailCount < 1 || r.TailCount > r.Count {
			t.Fatalf("large-count rank failed: %+v %v", r, err)
		}
		checkHistogramWidths(t, r)
	}
}

func TestLossHistogramLargeStreamBoundedStorage(t *testing.T) {
	h := newTestHistogram(t, -2, 3, 128)
	edges, counts := &h.edges[0], &h.counts[0]
	for i := 0; i < 1_000_000; i++ {
		if err := h.Add(float64(i%6) - 2); err != nil {
			t.Fatal(err)
		}
	}
	allocations := testing.AllocsPerRun(10, func() {
		for i := 0; i < 1000; i++ {
			if err := h.Add(0); err != nil {
				panic(err)
			}
		}
	})
	if allocations != 0 || len(h.edges) != 129 || cap(h.edges) != 129 || len(h.counts) != 128 || cap(h.counts) != 128 || edges != &h.edges[0] || counts != &h.counts[0] || h.Count() < 1_000_000 {
		t.Fatalf("storage grew with stream: count=%d allocations=%g", h.Count(), allocations)
	}
	r, err := h.Risk(.95)
	if err != nil || r.Count != h.Count() || r.VaR.Lower > 3 || r.VaR.Upper < 3 || r.ExpectedShortfall.Lower > 3 || r.ExpectedShortfall.Upper < 3 {
		t.Fatalf("large-stream risk: %+v %v", r, err)
	}
	checkHistogramWidths(t, r)
}
