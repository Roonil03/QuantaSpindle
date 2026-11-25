package quantaspindle

import (
	"fmt"
	"math"
	"math/big"
	"math/bits"
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
	maxLoss := 0.0
	for i, v := range x {
		if !finite(v) {
			return r, fmt.Errorf("loss %d is not finite", i)
		}
		maxLoss = math.Max(maxLoss, math.Abs(v))
	}
	r.MeanLoss = riskMean(x, maxLoss)
	if len(x) > 1 {
		r.StdDevLoss = riskStdDev(x, maxLoss)
	}
	if !finite(r.MeanLoss) || !finite(r.StdDevLoss) {
		return r, fmt.Errorf("risk moments overflow")
	}
	sort.Float64s(x)
	r.VaR = x[max(1, stableCeil(confidence*float64(len(x))))-1]
	r.TailCount = max(1, stableCeil((1-confidence)*float64(len(x))))
	r.ExpectedShortfall = riskMean(x[len(x)-r.TailCount:], maxLoss)
	if !finite(r.ExpectedShortfall) {
		return r, fmt.Errorf("expected shortfall overflow")
	}
	return r, nil
}

func riskMean(x []float64, maxLoss float64) float64 {
	sum, correction := 0.0, 0.0
	for _, v := range x {
		sum, correction = riskAdd(sum, correction, v)
		if !finite(sum) || !finite(correction) {
			return riskScaledMean(x, maxLoss)
		}
	}
	if total := sum + correction; finite(total) {
		return total / float64(len(x))
	}
	return riskScaledMean(x, maxLoss)
}

func riskScaledMean(x []float64, maxLoss float64) float64 {
	// Scaling by the count, rather than the largest observation, leaves small
	// residuals intact when large opposite-sign losses cancel. An extra bit of
	// headroom keeps intermediate sums away from overflow even after rounding.
	n := float64(len(x))
	_, exponent := math.Frexp(n)
	exponent++
	sum, correction := 0.0, 0.0
	residual, residualCorrection := 0.0, 0.0
	for _, v := range x {
		scaled := math.Ldexp(v, -exponent)
		sum, correction = riskAdd(sum, correction, scaled)
		// Count scaling can round subnormal terms. Retain the discarded bits
		// in their original units so constant subnormal samples stay nonzero.
		if math.Abs(scaled) < 0x1p-1022 {
			remainder := v - math.Ldexp(scaled, exponent)
			residual, residualCorrection = riskAdd(residual, residualCorrection, remainder)
		}
	}
	// Recombine discarded subnormal bits before dividing, so separately
	// rounded pieces cannot change a subnormal mean by a whole representable
	// step. If restoring the sum overflows, its mean is too large for these
	// tiny residual bits to affect it; normalize and divide before rescaling.
	total := sum + correction
	mean := 0.0
	if restored := math.Ldexp(total, exponent); finite(restored) {
		restored, restoredCorrection := riskAdd(restored, residualCorrection, residual)
		mean = (restored + restoredCorrection) / n
	} else {
		fraction, sumExponent := math.Frexp(total)
		mean = math.Ldexp(fraction/n, exponent+sumExponent)
	}
	// A mean is bounded by the largest magnitude observation. Rounding at
	// MaxFloat64 must not turn a representable mean into an infinity.
	return math.Max(-maxLoss, math.Min(maxLoss, mean))
}

func riskAdd(sum, correction, v float64) (float64, float64) {
	next := sum + v
	if math.Abs(sum) >= math.Abs(v) {
		correction += (sum - next) + v
	} else {
		correction += (v - next) + sum
	}
	return next, correction
}

func riskStdDev(x []float64, maxLoss float64) float64 {
	// Power-of-two scaling preserves nearby floats' spacing and brings even
	// subnormal losses into a range where squared deviations are representable.
	// Center about an observation before averaging to retain small spreads on
	// large offsets, including a mean halfway between adjacent floats.
	_, exponent := math.Frexp(maxLoss)
	anchor := math.Ldexp(x[0], -exponent)
	sum, correction := 0.0, 0.0
	for _, v := range x {
		y := (math.Ldexp(v, -exponent) - anchor) - correction
		next := sum + y
		correction = (next - sum) - y
		sum = next
	}
	mean := sum / float64(len(x))
	sum, correction = 0, 0
	for _, v := range x {
		d := (math.Ldexp(v, -exponent) - anchor) - mean
		y := d*d - correction
		next := sum + y
		correction = (next - sum) - y
		sum = next
	}
	return math.Ldexp(math.Sqrt(sum/float64(len(x)-1)), exponent)
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
	loss, correction := 0.0, 0.0
	overflow := false
	for i, p := range initial {
		if !finite(p) || p <= 0 || !finite(final[i]) || final[i] <= 0 || !finite(holdings[i]) {
			return 0, fmt.Errorf("invalid portfolio input %d", i)
		}
		term := holdings[i] * (p - final[i])
		if !finite(term) {
			return 0, fmt.Errorf("portfolio loss term %d overflow", i)
		}
		if !overflow {
			loss, correction = riskAdd(loss, correction, term)
			overflow = !finite(loss) || !finite(correction)
		}
	}
	loss += correction
	if overflow || !finite(loss) {
		// A float64 spans 2098 binary places, including subnormals. This
		// precision holds an exact sum of any slice of finite float64 terms,
		// including the extra bits needed for its element count. Only the rare
		// overflowing running sum pays for multiprecision arithmetic.
		precision := uint(2098 + bits.Len(uint(len(initial))))
		var sum, term big.Float
		sum.SetPrec(precision)
		term.SetPrec(precision)
		for i, p := range initial {
			term.SetFloat64(holdings[i] * (p - final[i]))
			sum.Add(&sum, &term)
		}
		loss, _ = sum.Float64()
	}
	if !finite(loss) {
		return 0, fmt.Errorf("portfolio loss overflow")
	}
	return loss, nil
}
