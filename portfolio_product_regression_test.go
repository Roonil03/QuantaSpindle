package quantaspindle

import (
	"math"
	"math/big"
	"reflect"
	"testing"
)

func TestPortfolioLossOverflowingProducts(t *testing.T) {
	for _, tc := range []struct {
		name                     string
		initial, final, holdings []float64
	}{
		{"reported cancellation", []float64{3, 3}, []float64{1, 1}, []float64{1e308, -1e308}},
		{"price gain cancellation", []float64{1, 1}, []float64{3, 3}, []float64{1e308, -1e308}},
		{"finite large residual", []float64{3, 3, 2}, []float64{1, 1, 1}, []float64{1e308, -1e308, 1e308}},
		{"finite small residual", []float64{3, 3, 2}, []float64{1, 1, 1}, []float64{1e308, -1e308, 3}},
		{"subnormal residual", []float64{3, 3, 2}, []float64{1, 1, 1}, []float64{1e308, -1e308, math.SmallestNonzeroFloat64}},
		{"exact price differences", []float64{math.MaxFloat64, math.MaxFloat64, 2}, []float64{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, 1}, []float64{math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Exact rationals provide an independent oracle for the full formula.
			want := new(big.Rat)
			for i, p := range tc.initial {
				term := new(big.Rat).Sub(new(big.Rat).SetFloat64(p), new(big.Rat).SetFloat64(tc.final[i]))
				term.Mul(term, new(big.Rat).SetFloat64(tc.holdings[i]))
				want.Add(want, term)
			}
			wantFloat, _ := want.Float64()
			var permute func(int)
			permute = func(start int) {
				if start < len(tc.initial) {
					for i := start; i < len(tc.initial); i++ {
						for _, values := range [][]float64{tc.initial, tc.final, tc.holdings} {
							values[start], values[i] = values[i], values[start]
						}
						permute(start + 1)
						for _, values := range [][]float64{tc.initial, tc.final, tc.holdings} {
							values[start], values[i] = values[i], values[start]
						}
					}
					return
				}
				before := [][]float64{append([]float64(nil), tc.initial...), append([]float64(nil), tc.final...), append([]float64(nil), tc.holdings...)}
				got, err := PortfolioLoss(tc.initial, tc.final, tc.holdings)
				if err != nil || got != wantFloat {
					t.Fatalf("loss = %.17g, %v; want %.17g for %v", got, err, wantFloat, tc.holdings)
				}
				if !reflect.DeepEqual(before, [][]float64{tc.initial, tc.final, tc.holdings}) {
					t.Fatal("portfolio inputs mutated")
				}
			}
			permute(0)
		})
	}
}

func TestPortfolioLossOverflowingProductsStillValidate(t *testing.T) {
	for _, tc := range []struct{ initial, final, holdings []float64 }{
		{[]float64{3}, []float64{1}, []float64{1e308}},
		{[]float64{3, 3, 3}, []float64{1, 1, 1}, []float64{1e308, -1e308, -1e308}},
		{[]float64{3, 3, math.NaN()}, []float64{1, 1, 1}, []float64{1e308, -1e308, 1}},
		{[]float64{3, 3, 2}, []float64{1, 1, 0}, []float64{1e308, -1e308, 1}},
		{[]float64{3, 3, 2}, []float64{1, 1, 1}, []float64{1e308, -1e308, math.Inf(1)}},
	} {
		if _, err := PortfolioLoss(tc.initial, tc.final, tc.holdings); err == nil {
			t.Fatalf("accepted invalid or overflowing portfolio: %+v", tc)
		}
	}
}
