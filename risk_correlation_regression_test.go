package quantaspindle

import (
	"context"
	"math"
	"math/big"
	"reflect"
	"testing"
)

func TestPortfolioLossFiniteCancellationPermutations(t *testing.T) {
	for _, tc := range []struct {
		name     string
		holdings []float64
		want     float64
	}{
		{"reported overflow", []float64{1e308, 1e308, -1e308}, 1e308},
		{"zero net", []float64{1e308, 1e308, -1e308, -1e308}, 0},
		{"maximum", []float64{math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64}, math.MaxFloat64},
		{"small residual", []float64{1e308, 1e308, 3, -1e308, -1e308}, 3},
		{"subnormal residual", []float64{1e308, 1e308, math.SmallestNonzeroFloat64, -1e308, -1e308}, math.SmallestNonzeroFloat64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			initial, final := make([]float64, len(tc.holdings)), make([]float64, len(tc.holdings))
			for i := range initial {
				initial[i], final[i] = 2, 1
			}
			var permute func(int)
			permute = func(start int) {
				if start < len(tc.holdings) {
					for i := start; i < len(tc.holdings); i++ {
						tc.holdings[start], tc.holdings[i] = tc.holdings[i], tc.holdings[start]
						permute(start + 1)
						tc.holdings[start], tc.holdings[i] = tc.holdings[i], tc.holdings[start]
					}
					return
				}
				before := append([]float64(nil), tc.holdings...)
				got, err := PortfolioLoss(initial, final, tc.holdings)
				if err != nil || got != tc.want {
					t.Fatalf("holdings %v: loss = %.17g, %v; want %.17g", tc.holdings, got, err, tc.want)
				}
				if !reflect.DeepEqual(tc.holdings, before) {
					t.Fatal("holdings mutated")
				}
			}
			permute(0)
		})
	}
}

func TestPortfolioLossStillRejectsOverflowAndInvalidInputs(t *testing.T) {
	for _, tc := range []struct{ initial, final, holdings []float64 }{
		{[]float64{2, 2}, []float64{1, 1}, []float64{1e308, 1e308}},
		{[]float64{3}, []float64{1}, []float64{math.MaxFloat64}},
		{[]float64{2, 2, math.NaN()}, []float64{1, 1, 1}, []float64{1e308, 1e308, -1e308}},
		{[]float64{2}, []float64{0}, []float64{1}},
		{[]float64{2}, []float64{1}, []float64{math.Inf(1)}},
	} {
		if _, err := PortfolioLoss(tc.initial, tc.final, tc.holdings); err == nil {
			t.Fatal("invalid/unrepresentable portfolio accepted", tc)
		}
	}
}

func nearSingularCorrelation() [][]float64 {
	return [][]float64{
		{1, 0.99999863803730449, -0.77687756636285288},
		{0.99999863803730449, 1, -0.77791570514828801},
		{-0.77687756636285288, -0.77791570514828801, 1},
	}
}

// Use exact binary rationals as an independent Sylvester-criterion oracle.
func correlationMinorSigns(c [][]float64) (int, int) {
	a, b, d := new(big.Rat).SetFloat64(c[0][1]), new(big.Rat).SetFloat64(c[0][2]), new(big.Rat).SetFloat64(c[1][2])
	minor := new(big.Rat).Sub(big.NewRat(1, 1), new(big.Rat).Mul(a, a))
	det := new(big.Rat).Mul(a, b)
	det.Mul(det, d).Mul(det, big.NewRat(2, 1)).Add(det, big.NewRat(1, 1))
	for _, x := range []*big.Rat{a, b, d} {
		det.Sub(det, new(big.Rat).Mul(x, x))
	}
	return minor.Sign(), det.Sign()
}

func TestNearlySingularPositiveDefiniteCorrelation(t *testing.T) {
	original := nearSingularCorrelation()
	minor, det := correlationMinorSigns(original)
	if minor != 1 || det != 1 {
		t.Fatalf("test oracle is not positive definite: minor=%d determinant=%d", minor, det)
	}
	for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		c := tinyConfig()
		c.Paths, c.Steps, c.Workers, c.BatchSize = 1, 1, 1, 1
		c.Assets = []Asset{{InitialPrice: 1, Sigma: .2}, {InitialPrice: 1, Sigma: .2}, {InitialPrice: 1, Sigma: .2}}
		c.Correlation = make([][]float64, 3)
		for i := range c.Correlation {
			c.Correlation[i] = make([]float64, 3)
			for j := range c.Correlation[i] {
				c.Correlation[i][j] = original[order[i]][order[j]]
			}
		}
		before := make([][]float64, 3)
		for i := range before {
			before[i] = append([]float64(nil), c.Correlation[i]...)
		}
		if err := c.Validate(); err != nil {
			t.Fatalf("order %v Validate: %v", order, err)
		}
		factor, err := correlationFactor(c.Correlation, 3)
		if err != nil {
			t.Fatal(err)
		}
		for i := range factor {
			for j := range factor {
				sum := 0.0
				for k := 0; k < 3; k++ {
					sum += factor[i][k] * factor[j][k]
				}
				if !finite(sum) || math.Abs(sum-c.Correlation[i][j]) > 2e-15 {
					t.Fatalf("order %v factor reconstruction [%d,%d] = %.17g", order, i, j, sum)
				}
			}
		}
		batch, err := (CPU{}).Batch(context.Background(), c, 0, 1)
		if err != nil || len(batch) != 1 {
			t.Fatalf("order %v CPU.Batch: %v", order, err)
		}
		if err := validateResult(batch[0], 0, c); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, c.Correlation) {
			t.Fatal("correlation mutated")
		}
	}
}

func TestNearlySingularCorrelationRejectsInvalidNeighbors(t *testing.T) {
	for _, asymmetric := range []bool{false, true} {
		c := tinyConfig()
		c.Assets = []Asset{{InitialPrice: 1}, {InitialPrice: 1}, {InitialPrice: 1}}
		c.Correlation = nearSingularCorrelation()
		c.Correlation[1][2] -= 1e-5
		if !asymmetric {
			c.Correlation[2][1] = c.Correlation[1][2]
			_, det := correlationMinorSigns(c.Correlation)
			if det >= 0 {
				t.Fatal("invalid neighbor must have negative determinant")
			}
		}
		if err := c.Validate(); err == nil {
			t.Fatal("invalid neighboring matrix accepted")
		}
		if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err == nil {
			t.Fatal("CPU accepted invalid neighboring matrix")
		}
	}
}

func BenchmarkCorrelationFactor256(b *testing.B) {
	for _, retry := range []bool{false, true} {
		name := "ordinary"
		if retry {
			name = "precise_retry"
		}
		b.Run(name, func(b *testing.B) {
			c := make([][]float64, 256)
			for i := range c {
				c[i] = make([]float64, 256)
				c[i][i] = 1
			}
			if retry {
				for i, row := range nearSingularCorrelation() {
					copy(c[i], row)
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := correlationFactor(c, 256); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestNearlySingularCorrelationWithDuplicateAsset(t *testing.T) {
	c := tinyConfig()
	c.Assets = []Asset{{InitialPrice: 1}, {InitialPrice: 1}, {InitialPrice: 1}, {InitialPrice: 1}}
	original := nearSingularCorrelation()
	order := []int{0, 1, 2, 2}
	c.Correlation = make([][]float64, len(order))
	for i := range order {
		c.Correlation[i] = make([]float64, len(order))
		for j := range order {
			c.Correlation[i][j] = original[order[i]][order[j]]
		}
	}
	if err := c.Validate(); err != nil {
		t.Fatal("singular duplicate asset rejected", err)
	}
	if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkPortfolioLossAccumulation(b *testing.B) {
	for _, tc := range []struct {
		name     string
		holdings []float64
	}{
		{"ordinary", []float64{.1, -.2, .3, .4}},
		{"overflow_retry", []float64{1e308, 1e308, -1e308}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			initial, final := make([]float64, len(tc.holdings)), make([]float64, len(tc.holdings))
			for i := range initial {
				initial[i], final[i] = 2, 1
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := PortfolioLoss(initial, final, tc.holdings); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
