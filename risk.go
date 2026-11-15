package quantaspindle

import (
	"fmt"
	"math"
	"sort"
)

// RiskSummary describes caller-retained losses. VaR uses the empirical nearest
// rank ceil(confidence*N); ExpectedShortfall averages the largest
// ceil((1-confidence)*N) losses, with a minimum of one observation. Counts within
// 1e-12 relative tolerance of an integer are rounded to avoid binary boundary
// artifacts. Negative risk values are preserved for all-gain samples. These
// finite-sample estimators are not regulatory capital calculations.
type RiskSummary struct {
	Count             int
	Confidence        float64
	MeanLoss          float64
	StdDevLoss        float64
	VaR               float64
	ExpectedShortfall float64
	TailCount         int
}

// RiskFromReturns computes exact empirical risk with loss = -return for a unit
// initial portfolio. It copies and sorts caller-retained returns using O(N)
// additional memory and O(N log N) time. It never retains observations itself.
func RiskFromReturns(returns []float64, confidence float64) (RiskSummary, error) {
	losses := make([]float64, len(returns))
	for i, r := range returns {
		losses[i] = -r
	}
	return RiskFromLosses(losses, confidence)
}

// RiskFromLosses computes exact empirical risk from losses, copying its input.
// Positive loss means a loss; negative loss means a gain. Standard deviation uses
// N-1 and is zero for a singleton. Empty or nonfinite inputs are rejected.
func RiskFromLosses(losses []float64, confidence float64) (RiskSummary, error) {
	r := RiskSummary{Count: len(losses), Confidence: confidence}
	if len(losses) == 0 || !finite(confidence) || confidence <= 0 || confidence >= 1 {
		return r, fmt.Errorf("risk requires observations and confidence strictly between zero and one")
	}
	x := append([]float64(nil), losses...)
	m2 := 0.0
	for i, v := range x {
		if !finite(v) {
			return r, fmt.Errorf("loss %d is not finite", i)
		}
		d := v - r.MeanLoss
		r.MeanLoss += d / float64(i+1)
		m2 += d * (v - r.MeanLoss)
	}
	if len(x) > 1 {
		r.StdDevLoss = math.Sqrt(math.Max(m2, 0) / float64(len(x)-1))
	}
	if !finite(r.MeanLoss) || !finite(r.StdDevLoss) {
		return r, fmt.Errorf("risk moments overflow")
	}
	sort.Float64s(x)
	r.VaR = x[max(1, stableCeil(confidence*float64(len(x))))-1]
	r.TailCount = max(1, stableCeil((1-confidence)*float64(len(x))))
	for i := 0; i < r.TailCount; i++ {
		v := x[len(x)-1-i]
		r.ExpectedShortfall += (v - r.ExpectedShortfall) / float64(i+1)
	}
	if !finite(r.ExpectedShortfall) {
		return r, fmt.Errorf("expected shortfall overflow")
	}
	return r, nil
}

func stableCeil(x float64) int {
	nearest := math.Round(x)
	if math.Abs(x-nearest) <= 1e-12*math.Max(1, math.Abs(x)) {
		return int(nearest)
	}
	return int(math.Ceil(x))
}

// PortfolioLoss returns sum(holdings[i]*(initial[i]-final[i])). All prices must
// be positive, finite and in a consistent currency; holdings may be short. There
// is no financing, dividend, rebalancing or cash-account convention implicit here.
func PortfolioLoss(initial, final, holdings []float64) (float64, error) {
	if len(initial) == 0 || len(final) != len(initial) || len(holdings) != len(initial) {
		return 0, fmt.Errorf("portfolio dimensions mismatch")
	}
	loss := 0.0
	for i, p := range initial {
		if !finite(p) || p <= 0 || !finite(final[i]) || final[i] <= 0 || !finite(holdings[i]) {
			return 0, fmt.Errorf("invalid portfolio input %d", i)
		}
		loss += holdings[i] * (p - final[i])
	}
	if !finite(loss) {
		return 0, fmt.Errorf("portfolio loss overflow")
	}
	return loss, nil
}
