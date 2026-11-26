package quantaspindle

import (
	"math"
	"testing"
)

func TestSquareRootImpactRepresentableFractions(t *testing.T) {
	tests := []struct {
		name                                               string
		model                                              SquareRootImpact
		mid, quantity, participation, temporary, permanent float64
	}{
		{"ratio underflow buy", SquareRootImpact{Volatility: 1, DailyVolume: 1e200, Temporary: 1, Permanent: 1}, 100, 1e-200, 0, 1e-200, 1e-200},
		{"ratio underflow sell", SquareRootImpact{Volatility: 1, DailyVolume: 1e200, Temporary: 1, Permanent: 1}, 100, -1e-200, 0, 1e-200, 1e-200},
		{"temporary intermediate overflow", SquareRootImpact{Volatility: 1e200, DailyVolume: 1, Temporary: 1e-150}, 1e-200, 1e300, 1e300, 1e200, 0},
		{"permanent intermediate overflow", SquareRootImpact{Volatility: 1e200, DailyVolume: 1, Permanent: 1e-150}, 1e-200, 1e300, 1e300, 0, 1e200},
		{"intermediate underflow", SquareRootImpact{Volatility: 1e-200, DailyVolume: 1, Temporary: 1e200, Permanent: 1e200}, 100, 1e-300, 1e-300, 1e-150, 1e-150},
		{"zero coefficients", SquareRootImpact{Volatility: 1e200, DailyVolume: 1}, 1e-200, 1e300, 1e300, 0, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.model.Execute(tt.mid, tt.quantity)
			if err != nil {
				t.Fatal(err)
			}
			impactRelativeClose(t, "participation", got.ParticipationRate, tt.participation)
			impactRelativeClose(t, "temporary", got.TemporaryImpact, tt.temporary)
			impactRelativeClose(t, "permanent", got.PermanentImpact, tt.permanent)
			if tt.name == "temporary intermediate overflow" {
				impactRelativeClose(t, "execution price", got.ExecutionPrice, 1)
				impactRelativeClose(t, "cash change", got.CashChange, -1e300)
				impactRelativeClose(t, "cost", got.Cost, 1e300)
			}
		})
	}
}

func TestSquareRootImpactCostBeforePriceRounding(t *testing.T) {
	tests := []struct {
		name                string
		model               SquareRootImpact
		mid, quantity, cost float64
	}{
		{"spread buy", SquareRootImpact{DailyVolume: 1, HalfSpread: 1e-17}, 1, 1e16, .1},
		{"spread sell", SquareRootImpact{DailyVolume: 1, HalfSpread: 1e-17}, 1, -1e16, .1},
		{"temporary buy", SquareRootImpact{Volatility: 1, DailyVolume: 1e16, Temporary: 1e-17}, 1, 1e16, .1},
		{"temporary sell", SquareRootImpact{Volatility: 1, DailyVolume: 1e16, Temporary: 1e-17}, 1, -1e16, .1},
		{"tiny price adjustment", SquareRootImpact{DailyVolume: 1, HalfSpread: 1e-200}, 1e-200, 1e300, 1e-100},
		{"tiny notional", SquareRootImpact{DailyVolume: 1, HalfSpread: 1e200}, 1e-200, 1e-200, 1e-200},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.model.Execute(tt.mid, tt.quantity)
			if err != nil {
				t.Fatal(err)
			}
			impactRelativeClose(t, "cost", got.Cost, tt.cost)
			if tt.mid == 1 && got.ExecutionPrice != 1 {
				t.Fatalf("execution price = %g, want rounded mid 1", got.ExecutionPrice)
			}
			if math.Signbit(got.CashChange) == math.Signbit(tt.quantity) {
				t.Fatalf("cash change %g must oppose quantity %g", got.CashChange, tt.quantity)
			}
		})
	}
}

func TestSquareRootImpactOverflowingAdjustment(t *testing.T) {
	m := SquareRootImpact{Volatility: 1, DailyVolume: 1, Temporary: 1e308, HalfSpread: 1e308}
	for _, mid := range []float64{1e-308, math.SmallestNonzeroFloat64} {
		got, err := m.Execute(mid, 1)
		if err != nil {
			t.Fatal(err)
		}
		want := 2 * (mid * 1e308)
		impactRelativeClose(t, "execution price", got.ExecutionPrice, want)
		impactRelativeClose(t, "cash change", got.CashChange, -want)
		impactRelativeClose(t, "cost", got.Cost, want)
		if got.TemporaryImpact != 1e308 || got.PostTradeMid != mid {
			t.Fatalf("unexpected impact or midpoint: %+v", got)
		}
	}
	// The scale-aware sum must still reject a nonpositive sale price or a
	// required output that really exceeds float64.
	for _, tc := range []struct{ mid, quantity float64 }{
		{1e-308, -1}, {1, 1}, {1e-308, 1e308},
	} {
		if _, err := m.Execute(tc.mid, tc.quantity); err == nil {
			t.Fatalf("accepted unrepresentable execution for %+v", tc)
		}
	}
}

func TestSquareRootImpactMidpointDisplacementRounding(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		mid, quantity, impact float64
		want                  float64
	}{
		{"reported buy", math.Nextafter(2, 0), 1, 1e-16, 2},
		{"sell", 2, -1, 1e-16, math.Nextafter(2, 0)},
		{"rounded unchanged buy", 2, 1, 1e-17, 2},
		{"rounded unchanged sell", 2, -1, 1e-17, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := SquareRootImpact{Volatility: 1, DailyVolume: 1, Permanent: tc.impact}
			got, err := m.Execute(tc.mid, tc.quantity)
			if err != nil || got.PostTradeMid != tc.want || got.PermanentImpact != tc.impact {
				t.Fatalf("execution = %+v, %v; want midpoint %.17g", got, err, tc.want)
			}
		})
	}
	for _, impact := range []float64{1, 2} {
		m := SquareRootImpact{Volatility: 1, DailyVolume: 1, Permanent: impact}
		if _, err := m.Execute(2, -1); err == nil {
			t.Fatalf("accepted nonpositive midpoint for impact %g", impact)
		}
	}
	m := SquareRootImpact{Volatility: 1, DailyVolume: 1, Permanent: 1}
	if _, err := m.Execute(math.MaxFloat64, 1); err == nil {
		t.Fatal("accepted overflowing midpoint")
	}
}

func TestSquareRootImpactRejectsUnrepresentableResults(t *testing.T) {
	tests := []struct {
		name          string
		model         SquareRootImpact
		mid, quantity float64
	}{
		{"participation overflow", SquareRootImpact{DailyVolume: 1e-200}, 1, 1e200},
		{"impact overflow", SquareRootImpact{Volatility: 1e200, DailyVolume: 1, Temporary: 1}, 1e-200, 1e300},
		{"cash overflow", SquareRootImpact{DailyVolume: 1}, 1e200, 1e200},
		{"cost overflow", SquareRootImpact{DailyVolume: 1, HalfSpread: math.Nextafter(1, 0)}, 1e109, -1e200},
		{"nonpositive sell price", SquareRootImpact{DailyVolume: 1, HalfSpread: 1}, 1, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.model.Execute(tt.mid, tt.quantity); err == nil {
				t.Fatal("expected an invalid price or cash error")
			}
		})
	}
}

func impactRelativeClose(t *testing.T, field string, got, want float64) {
	t.Helper()
	if !finite(got) || (want == 0 && got != 0) || (want != 0 && math.Abs((got-want)/want) > 2e-15) {
		t.Fatalf("%s = %.17g, want %.17g", field, got, want)
	}
}
