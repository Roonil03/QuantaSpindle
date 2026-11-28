package quantaspindle

import (
	"fmt"
	"math"
	"math/big"
	"sort"
)

// RiskInterval is a closed interval of finite signed losses.
type RiskInterval struct {
	Lower float64
	Upper float64
}

// HistogramRiskSummary bounds empirical tail risk without retaining losses.
// VaR uses the same nearest rank, tail count and boundary tolerance as
// RiskFromLosses. ExpectedShortfall encloses the arithmetic mean of the largest
// TailCount losses. BinWidth is an upward-rounded maximum represented bin width.
// VaR's width is at most BinWidth. ExpectedShortfall's width is at most
// BinWidth+RoundingAllowance, interpreted in exact arithmetic if that addition
// would overflow. RoundingAllowance covers outward float64 rounding only.
type HistogramRiskSummary struct {
	Count             uint64
	Confidence        float64
	TailCount         uint64
	VaR               RiskInterval
	ExpectedShortfall RiskInterval
	BinWidth          float64
	RoundingAllowance float64
}

// LossHistogram is an opt-in fixed-memory accumulator for finite signed losses.
// Storage is O(bins), Add takes O(log bins) and allocates no memory, and Risk and
// Merge take O(bins). It retains counts, not observations or per-bin sums.
// Construct it with NewLossHistogram; its zero value is invalid. Do not copy it
// after construction or call its methods concurrently without synchronization.
type LossHistogram struct {
	edges  []float64
	counts []uint64
	count  uint64
	width  float64
}

// NewLossHistogram partitions [lower, upper] into bins (1..1,000,000).
// Interior edge i is lower+(upper-lower)*i/bins, evaluated exactly and rounded
// once to nearest float64, ties to even. Rounded edges must be strictly increasing
// and every represented bin width must fit a finite float64; otherwise the
// configuration is rejected. This permits signed ranges whose total width
// overflows float64 when the individual bins remain representable.
// Bins are [edge[i], edge[i+1]), except the last includes upper. Both declared
// endpoints are accepted; an interior edge belongs to the bin on its right.
func NewLossHistogram(lower, upper float64, bins int) (*LossHistogram, error) {
	if !finite(lower) || !finite(upper) || lower >= upper || bins < 1 || bins > 1_000_000 {
		return nil, fmt.Errorf("histogram requires finite lower < upper and 1..1000000 bins")
	}
	// Canonical zeros make independently constructed grids identical.
	if lower == 0 {
		lower = 0
	}
	if upper == 0 {
		upper = 0
	}
	h := &LossHistogram{edges: make([]float64, bins+1), counts: make([]uint64, bins)}
	h.edges[0], h.edges[bins] = lower, upper
	lo, hi := new(big.Rat).SetFloat64(lower), new(big.Rat).SetFloat64(upper)
	span := new(big.Rat).Sub(hi, lo)
	var position, edge, width big.Rat
	for i := 1; i <= bins; i++ {
		if i < bins {
			position.SetFrac64(int64(i), int64(bins))
			edge.Mul(span, &position)
			edge.Add(&edge, lo)
			h.edges[i], _ = edge.Float64()
		}
		if !finite(h.edges[i]) || h.edges[i] <= h.edges[i-1] {
			return nil, fmt.Errorf("histogram edges collapse at bin %d", i-1)
		}
		width.Sub(new(big.Rat).SetFloat64(h.edges[i]), new(big.Rat).SetFloat64(h.edges[i-1]))
		w := histogramRound(&width, true)
		if !finite(w) {
			return nil, fmt.Errorf("histogram bin %d width is not representable", i-1)
		}
		h.width = math.Max(h.width, w)
	}
	return h, nil
}

// Count returns the number of accepted losses (zero for a nil receiver).
func (h *LossHistogram) Count() uint64 {
	if h == nil {
		return 0
	}
	return h.count
}

// Edges returns an independently owned copy of the represented bin edges.
func (h *LossHistogram) Edges() []float64 {
	if h == nil {
		return nil
	}
	return append([]float64(nil), h.edges...)
}

func (h *LossHistogram) valid() bool {
	return h != nil && len(h.counts) > 0 && len(h.edges) == len(h.counts)+1
}

// Add accepts a finite loss within the inclusive declared bounds. Invalid or
// out-of-range inputs and uint64 count overflow return errors without mutation.
func (h *LossHistogram) Add(loss float64) error {
	if !h.valid() {
		return fmt.Errorf("histogram is not initialized")
	}
	if !finite(loss) || loss < h.edges[0] || loss > h.edges[len(h.edges)-1] {
		return fmt.Errorf("histogram loss is not finite or outside declared bounds")
	}
	if h.count == math.MaxUint64 {
		return fmt.Errorf("histogram count overflow")
	}
	i := sort.Search(len(h.counts), func(i int) bool { return loss < h.edges[i+1] })
	if i == len(h.counts) {
		i-- // The inclusive upper endpoint belongs to the last bin.
	}
	h.counts[i]++
	h.count++
	return nil
}

// Merge adds all counts from another histogram with identical represented
// edges. It does not mutate other; self-merge doubles counts. Integer addition
// makes results independent of merge order. Invalid/incompatible inputs and
// uint64 count overflow return errors without changing either histogram.
func (h *LossHistogram) Merge(other *LossHistogram) error {
	if !h.valid() || !other.valid() || len(h.edges) != len(other.edges) {
		return fmt.Errorf("histograms are not initialized or have incompatible edges")
	}
	for i, edge := range h.edges {
		if edge != other.edges[i] {
			return fmt.Errorf("histograms have incompatible edges")
		}
	}
	if math.MaxUint64-h.count < other.count {
		return fmt.Errorf("histogram count overflow")
	}
	for i := range h.counts {
		h.counts[i] += other.counts[i]
	}
	h.count += other.count
	return nil
}

// Risk returns conservative closed intervals for the same empirical VaR/ES
// definitions as RiskFromLosses. Empty histograms and nonfinite confidences or
// confidences outside (0,1) are rejected. Exact rational weighted bin endpoints
// avoid overflowing intermediate sums. ES endpoints round outward and then
// expand by one finite float64 step; no observation is silently clamped.
func (h *LossHistogram) Risk(confidence float64) (HistogramRiskSummary, error) {
	var r HistogramRiskSummary
	if !h.valid() || h.count == 0 || !finite(confidence) || confidence <= 0 || confidence >= 1 {
		return r, fmt.Errorf("histogram risk requires observations and confidence strictly between zero and one")
	}
	r.Count, r.Confidence, r.BinWidth = h.count, confidence, h.width
	rank := histogramRank(confidence, h.count)
	r.TailCount = histogramRank(1-confidence, h.count)
	var cumulative uint64
	for i, count := range h.counts {
		cumulative += count
		if cumulative >= rank {
			r.VaR = RiskInterval{Lower: h.edges[i], Upper: h.edges[i+1]}
			break
		}
	}
	var lower, upper, term, weight big.Rat
	remaining := r.TailCount
	for i := len(h.counts) - 1; i >= 0 && remaining > 0; i-- {
		take := min(remaining, h.counts[i])
		if take == 0 {
			continue
		}
		weight.SetInt(new(big.Int).SetUint64(take))
		term.SetFloat64(h.edges[i])
		term.Mul(&term, &weight)
		lower.Add(&lower, &term)
		term.SetFloat64(h.edges[i+1])
		term.Mul(&term, &weight)
		upper.Add(&upper, &term)
		remaining -= take
	}
	weight.SetInt(new(big.Int).SetUint64(r.TailCount))
	lower.Quo(&lower, &weight)
	upper.Quo(&upper, &weight)
	r.ExpectedShortfall.Lower = histogramExpand(histogramRound(&lower, false), false)
	r.ExpectedShortfall.Upper = histogramExpand(histogramRound(&upper, true), true)
	if !finite(r.ExpectedShortfall.Lower) || !finite(r.ExpectedShortfall.Upper) {
		return HistogramRiskSummary{}, fmt.Errorf("histogram risk interval is not representable")
	}
	// Report the actual outward displacement, rounded upward. Computing it
	// exactly also handles spacing changes at powers of two and subnormals.
	var allowance big.Rat
	allowance.SetFloat64(r.ExpectedShortfall.Lower)
	allowance.Sub(&lower, &allowance)
	term.SetFloat64(r.ExpectedShortfall.Upper)
	term.Sub(&term, &upper)
	allowance.Add(&allowance, &term)
	r.RoundingAllowance = histogramRound(&allowance, true)
	return r, nil
}

func histogramRank(probability float64, count uint64) uint64 {
	x := probability * float64(count)
	nearest := math.Round(x)
	if math.Abs(x-nearest) <= 1e-12*math.Max(1, math.Abs(x)) {
		x = nearest
	} else {
		x = math.Ceil(x)
	}
	if x <= 1 {
		return 1
	}
	// float64(MaxUint64) rounds to 2^64. Bound before converting, including
	// the case where 1-confidence rounds to 1 for very small confidences.
	if x >= float64(count) {
		return count
	}
	return uint64(x)
}

func histogramRound(value *big.Rat, up bool) float64 {
	x, exact := value.Float64()
	if !exact && finite(x) {
		comparison := new(big.Rat).SetFloat64(x).Cmp(value)
		if (up && comparison < 0) || (!up && comparison > 0) {
			direction := math.Inf(-1)
			if up {
				direction = math.Inf(1)
			}
			x = math.Nextafter(x, direction)
		}
	}
	return x
}

func histogramExpand(x float64, up bool) float64 {
	direction := math.Inf(-1)
	if up {
		direction = math.Inf(1)
	}
	if next := math.Nextafter(x, direction); finite(next) {
		return next
	}
	return x
}
