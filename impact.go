package quantaspindle

import (
	"fmt"
	"math"
)

// SquareRootImpact is an optional, independent execution overlay. Volatility is
// dimensionless daily volatility; DailyVolume and quantity use the same units.
// Temporary and Permanent are nonnegative dimensionless coefficients multiplying
// volatility*sqrt(abs(quantity)/DailyVolume). HalfSpread is a relative price
// fraction. This is a stylized square-root model, not an optimal execution solver.
type SquareRootImpact struct {
	Volatility  float64
	DailyVolume float64
	Temporary   float64
	Permanent   float64
	HalfSpread  float64
}

// Execution describes one signed trade. Positive quantity buys and consumes
// cash; negative quantity sells and receives cash. Cost is adverse cash cost
// relative to quantity*mid, including spread and temporary impact only. Cost is
// computed from the modeled relative adjustment before prices are rounded.
// Impact fractions and prices are model-dependent estimates. ParticipationRate
// is abs(quantity)/DailyVolume, a fraction of daily liquidity, not a measured
// intraday execution participation rate. TemporaryImpact and PermanentImpact are
// unsigned relative price fractions; the trade sign determines their direction.
type Execution struct {
	Quantity          float64
	ExecutionPrice    float64
	PostTradeMid      float64
	CashChange        float64
	Cost              float64
	DailyVolume       float64
	HalfSpread        float64
	ParticipationRate float64
	TemporaryImpact   float64
	PermanentImpact   float64
}

// Validate checks finite nonnegative coefficients and strictly positive volume.
func (m SquareRootImpact) Validate() error {
	if !finite(m.DailyVolume) || m.DailyVolume <= 0 {
		return fmt.Errorf("daily volume must be positive and finite")
	}
	for _, v := range []float64{m.Volatility, m.Temporary, m.Permanent, m.HalfSpread} {
		if !finite(v) || v < 0 {
			return fmt.Errorf("impact coefficients must be nonnegative and finite")
		}
	}
	return nil
}

// Execute applies signed temporary execution cost and persistent mid-price
// displacement separately. It does not mutate GBM paths or holdings. A zero
// quantity is a no-op, and zero coefficients reproduce the supplied mid price.
func (m SquareRootImpact) Execute(mid, quantity float64) (Execution, error) {
	e := Execution{Quantity: quantity, DailyVolume: m.DailyVolume, HalfSpread: m.HalfSpread}
	if err := m.Validate(); err != nil {
		return e, err
	}
	if !finite(mid) || mid <= 0 || !finite(quantity) {
		return e, fmt.Errorf("mid must be positive and quantity finite")
	}
	if quantity == 0 {
		e.ExecutionPrice = mid
		e.PostTradeMid = mid
		return e, nil
	}
	sign := math.Copysign(1, quantity)
	e.ParticipationRate = math.Abs(quantity) / m.DailyVolume
	e.TemporaryImpact = squareRootImpactFraction(m.Temporary, m.Volatility, math.Abs(quantity), m.DailyVolume)
	e.PermanentImpact = squareRootImpactFraction(m.Permanent, m.Volatility, math.Abs(quantity), m.DailyVolume)
	e.ExecutionPrice = mid * (1 + sign*(m.HalfSpread+e.TemporaryImpact))
	e.PostTradeMid = mid * (1 + sign*e.PermanentImpact)
	e.CashChange = -quantity * e.ExecutionPrice
	e.Cost = impactCost(math.Abs(quantity), mid, m.HalfSpread+e.TemporaryImpact)
	if !finite(e.ParticipationRate) || !finite(e.TemporaryImpact) || !finite(e.PermanentImpact) || !finite(e.ExecutionPrice) || e.ExecutionPrice <= 0 || !finite(e.PostTradeMid) || e.PostTradeMid <= 0 || !finite(e.CashChange) || !finite(e.Cost) {
		return Execution{}, fmt.Errorf("impact produces invalid price or cash value")
	}
	return e, nil
}

// squareRootImpactFraction keeps the ratio and product exponents separate so
// intermediate underflow or overflow cannot erase a representable impact.
// All arguments are finite and nonnegative, and quantity and volume are positive.
func squareRootImpactFraction(coefficient, volatility, quantity, volume float64) float64 {
	if coefficient == 0 || volatility == 0 {
		return 0
	}
	q, qe := math.Frexp(quantity)
	v, ve := math.Frexp(volume)
	exponent := qe - ve
	if exponent%2 != 0 {
		q *= 2
		exponent--
	}
	c, ce := math.Frexp(coefficient)
	s, se := math.Frexp(volatility)
	return math.Ldexp(c*s*math.Sqrt(q/v), ce+se+exponent/2)
}

// impactCost multiplies finite nonnegative factors without prematurely rounding
// an intermediate notional or price adjustment out of float64's range.
func impactCost(quantity, mid, adjustment float64) float64 {
	if adjustment == 0 {
		return 0
	}
	q, qe := math.Frexp(quantity)
	m, me := math.Frexp(mid)
	a, ae := math.Frexp(adjustment)
	return math.Ldexp(q*m*a, qe+me+ae)
}
