package quantaspindle

import (
	"context"
	"math"
	"math/big"
	"testing"
)

func onePathConfig() Config {
	c := DefaultConfig()
	c.Paths, c.Steps, c.Workers, c.BatchSize = 1, 1, 1, 1
	return c
}

func TestCPUTinyRealizedVolatility(t *testing.T) {
	c := onePathConfig()
	c.Steps = 2
	c.Assets[0].Mu = 0
	c.Assets[0].Sigma = 1e-150
	baseline, err := (CPU{}).Batch(context.Background(), c, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	baseVol := baseline[0].RealizedVolatility[0]
	if baseVol <= 0 {
		t.Fatalf("nonpositive baseline volatility %g", baseVol)
	}
	for _, pair := range [][2]float64{{1e-170, c.DT}, {1e-308, c.DT}, {1e-310, c.DT}, {1e-155, 1e-310}, {0, c.DT}} {
		sigma := pair[0]
		c.Assets[0].Sigma, c.DT = sigma, pair[1]
		got, err := (CPU{}).Batch(context.Background(), c, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
		want := baseVol * (sigma / 1e-150)
		vol := got[0].RealizedVolatility[0]
		tolerance := math.Max(want*1e-11, 32*math.SmallestNonzeroFloat64)
		if !finite(vol) || (sigma > 0 && vol <= 0) || math.Abs(vol-want) > tolerance {
			t.Fatalf("sigma %g: volatility %g, want %g within %g", sigma, vol, want, tolerance)
		}
	}
}

func TestCPUPriceExponentialRange(t *testing.T) {
	for _, tc := range []struct {
		name              string
		initial, mu, want float64
	}{
		{"exponential underflow", 1e300, -750, 1.90168496347500644e-26},
		{"subnormal exponential", 1e300, -740, 4.18873988004804894e-22},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := onePathConfig()
			c.DT, c.StorePaths = 1, true
			c.Assets[0] = Asset{InitialPrice: tc.initial, Mu: tc.mu}
			got, _ := collect(t, c)
			s := got[0]
			if math.Abs(s.FinalPrices[0]/tc.want-1) > 2e-13 {
				t.Fatalf("price %.17g, want %.17g", s.FinalPrices[0], tc.want)
			}
			if tc.mu < 0 && s.Returns[0] != -1 {
				t.Fatalf("return %g, want -1", s.Returns[0])
			}
			if err := validateResult(s, 0, c); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, mu := range []float64{-750, 750} {
		c := onePathConfig()
		c.DT = 1
		c.Assets[0] = Asset{InitialPrice: 1, Mu: mu}
		if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err == nil {
			t.Fatalf("accepted unrepresentable price for mu %g", mu)
		}
	}
}

func TestCPUIntegratedVarianceRange(t *testing.T) {
	c := onePathConfig()
	c.DT = 1
	c.Assets[0].Mu, c.Assets[0].Sigma = 0, 1
	baseline, err := (CPU{}).Batch(context.Background(), c, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	c.Assets[0].Sigma, c.DT = 1e155, 1e-310
	got, err := (CPU{}).Batch(context.Background(), c, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got[0].FinalPrices[0]/baseline[0].FinalPrices[0]-1) > 1e-12 {
		t.Fatalf("scaled variance changed price: got %g want %g", got[0].FinalPrices[0], baseline[0].FinalPrices[0])
	}
	c.DT = 1
	if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err == nil {
		t.Fatal("accepted unrepresentable integrated variance")
	}
}

func TestCPUAdjacentPriceReturn(t *testing.T) {
	for _, direction := range []float64{math.Inf(1), math.Inf(-1)} {
		initial := 1.25
		final := math.Nextafter(initial, direction)
		c := onePathConfig()
		c.DT = 1
		c.Assets[0] = Asset{InitialPrice: initial, Mu: math.Log(final / initial)}
		got, summary := collect(t, c)
		if got[0].FinalPrices[0] != final {
			t.Fatalf("final price %.17g want %.17g", got[0].FinalPrices[0], final)
		}
		// Use exact rationals to avoid the rounded intermediate ratio.
		i, f := new(big.Rat).SetFloat64(initial), new(big.Rat).SetFloat64(final)
		want, _ := new(big.Rat).Quo(new(big.Rat).Sub(f, i), i).Float64()
		if got[0].Returns[0] != want || summary.MeanReturns[0] != want {
			t.Fatalf("scenario/summary returns %g/%g want %g", got[0].Returns[0], summary.MeanReturns[0], want)
		}
	}
}

type nilSessionBackend struct{ CPU }

func (nilSessionBackend) BeginRun(context.Context, Config) (BackendSession, error) {
	return (*nilSession)(nil), nil
}

type nilSession struct{ CPU }

func (*nilSession) Close() error { panic("nil session must not be closed") }

func TestRunRejectsTypedNilBackend(t *testing.T) {
	c := onePathConfig()
	for _, backend := range []Backend{nil, (*CPU)(nil), (*trackingBackend)(nil)} {
		if _, err := Run(context.Background(), c, backend, nil); err == nil || err.Error() != "backend is nil" {
			t.Fatalf("backend %T returned %v", backend, err)
		}
	}
	if _, err := Run(context.Background(), c, nilSessionBackend{}, nil); err == nil || err.Error() != "backend returned nil run session" {
		t.Fatalf("typed nil session returned %v", err)
	}
	for _, backend := range []Backend{CPU{}, &CPU{}} {
		if _, err := Run(context.Background(), c, backend, nil); err != nil {
			t.Fatal(err)
		}
	}
}

// referencePrice uses high-precision Taylor expansion and repeated squaring,
// independent of the float64 logarithm and ln(2) reduction used by the engine.
func referencePrice(price, lr float64) float64 {
	const precision = 256
	number := func(x float64) *big.Float { return new(big.Float).SetPrec(precision).SetFloat64(x) }
	x := number(lr)
	x.Quo(x, number(4096))
	sum, term := number(1), number(1)
	for n := 1; n <= 128; n++ {
		term.Mul(term, x)
		term.Quo(term, number(float64(n)))
		sum.Add(sum, term)
	}
	for n := 0; n < 12; n++ {
		sum.Mul(sum, sum)
	}
	sum.Mul(sum, number(price))
	value, _ := sum.Float64()
	return value
}

func TestPriceAfterLogReturnRange(t *testing.T) {
	for _, tc := range []struct {
		name        string
		initial, lr float64
	}{
		{"smallest underflow", math.SmallestNonzeroFloat64, -1},
		{"smallest stays positive", math.SmallestNonzeroFloat64, -0.5},
		{"smallest just above half", math.SmallestNonzeroFloat64, -math.Ln2},
		{"smallest below half", math.SmallestNonzeroFloat64, math.Nextafter(-math.Ln2, math.Inf(-1))},
		{"odd subnormal above half", 3 * math.SmallestNonzeroFloat64, -math.Ln2},
		{"odd subnormal below half", 3 * math.SmallestNonzeroFloat64, math.Nextafter(-math.Ln2, math.Inf(-1))},
		{"upper subnormal tiny positive", math.Nextafter(0x1p-1022, 0), 5e-17},
		{"upper subnormal tiny negative", math.Nextafter(0x1p-1022, 0), -5e-17},
		{"upper subnormal positive below half", math.Nextafter(0x1p-1022, 0), 1e-16},
		{"upper subnormal negative below half", math.Nextafter(0x1p-1022, 0), -1e-16},
		{"upper subnormal positive above half", math.Nextafter(0x1p-1022, 0), 1.2e-16},
		{"upper subnormal negative above half", math.Nextafter(0x1p-1022, 0), -1.2e-16},
		{"smallest normal shrinks below half", 0x1p-1022, -1e-16},
		{"smallest normal shrinks above half", 0x1p-1022, -1.2e-16},
		{"smallest grows", math.SmallestNonzeroFloat64, 1},
		{"smallest recovers", math.SmallestNonzeroFloat64, 700},
		{"smallest exponential overflow recovers", math.SmallestNonzeroFloat64, 1450},
		{"smallest true overflow", math.SmallestNonzeroFloat64, 1455},
		{"subnormal recovers", 0x1p-1030, 750},
		{"normal recovers", 1e-300, 750},
		{"exponential underflow recovers", 1e300, -750},
		{"ordinary overflow boundary below", 1e300, math.Nextafter(math.Log(math.MaxFloat64/1e300), math.Inf(-1))},
		{"ordinary overflow boundary", 1e300, math.Log(math.MaxFloat64 / 1e300)},
		{"ordinary overflow boundary above", 1e300, math.Nextafter(math.Log(math.MaxFloat64/1e300), math.Inf(1))},
		{"largest rounds finite", math.MaxFloat64, 5e-17},
		{"largest overflows", math.MaxFloat64, 1e-16},
		{"largest shrinks", math.MaxFloat64, -1e-16},
		{"largest subnormal result", math.MaxFloat64, -1453},
		{"largest underflow", math.MaxFloat64, -1454},
		{"huge positive return", math.SmallestNonzeroFloat64, math.MaxFloat64},
		{"huge negative return", math.MaxFloat64, -math.MaxFloat64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want float64
			switch tc.lr {
			case math.MaxFloat64:
				want = math.Inf(1)
			case -math.MaxFloat64:
				want = 0
			default:
				want = referencePrice(tc.initial, tc.lr)
			}
			got := priceAfterLogReturn(tc.initial, tc.lr)
			if want <= 0x1p-1022 || math.IsInf(want, 1) || want == math.MaxFloat64 {
				if got != want {
					t.Fatalf("price %.17g, want %.17g", got, want)
				}
			} else if !finite(got) || math.Abs(got/want-1) > 1e-15 {
				t.Fatalf("price %.17g, want %.17g", got, want)
			}
		})
	}
}

func TestCPUPriceRangeLimits(t *testing.T) {
	for _, tc := range []struct {
		name        string
		initial, lr float64
	}{
		{"smallest underflow", math.SmallestNonzeroFloat64, -1},
		{"smallest stays positive", math.SmallestNonzeroFloat64, -0.5},
		{"smallest just above half", math.SmallestNonzeroFloat64, -math.Ln2},
		{"smallest below half", math.SmallestNonzeroFloat64, math.Nextafter(-math.Ln2, math.Inf(-1))},
		{"odd subnormal above half", 3 * math.SmallestNonzeroFloat64, -math.Ln2},
		{"odd subnormal below half", 3 * math.SmallestNonzeroFloat64, math.Nextafter(-math.Ln2, math.Inf(-1))},
		{"upper subnormal tiny positive", math.Nextafter(0x1p-1022, 0), 5e-17},
		{"upper subnormal tiny negative", math.Nextafter(0x1p-1022, 0), -5e-17},
		{"upper subnormal positive below half", math.Nextafter(0x1p-1022, 0), 1e-16},
		{"upper subnormal negative below half", math.Nextafter(0x1p-1022, 0), -1e-16},
		{"upper subnormal positive above half", math.Nextafter(0x1p-1022, 0), 1.2e-16},
		{"upper subnormal negative above half", math.Nextafter(0x1p-1022, 0), -1.2e-16},
		{"smallest normal shrinks below half", 0x1p-1022, -1e-16},
		{"smallest normal shrinks above half", 0x1p-1022, -1.2e-16},
		{"smallest grows", math.SmallestNonzeroFloat64, 1},
		{"smallest recovers", math.SmallestNonzeroFloat64, 700},
		{"ordinary overflow boundary below", 1e300, math.Nextafter(math.Log(math.MaxFloat64/1e300), math.Inf(-1))},
		{"ordinary overflow boundary", 1e300, math.Log(math.MaxFloat64 / 1e300)},
		{"ordinary overflow boundary above", 1e300, math.Nextafter(math.Log(math.MaxFloat64/1e300), math.Inf(1))},
		{"largest rounds finite", math.MaxFloat64, 5e-17},
		{"largest overflows", math.MaxFloat64, 1e-16},
		{"largest underflow", math.MaxFloat64, -1454},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := onePathConfig()
			c.DT, c.StorePaths = 1, true
			c.Assets[0] = Asset{InitialPrice: tc.initial, Mu: tc.lr}
			want := referencePrice(tc.initial, tc.lr)
			got, err := (CPU{}).Batch(context.Background(), c, 0, 1)
			if want == 0 || !finite(want) {
				if err == nil {
					t.Fatalf("accepted unrepresentable price: %+v", got[0])
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			price := got[0].FinalPrices[0]
			if want <= 0x1p-1022 || want == math.MaxFloat64 {
				if price != want {
					t.Fatalf("price %.17g, want %.17g", price, want)
				}
			} else if !finite(price) || math.Abs(price/want-1) > 1e-15 {
				t.Fatalf("price %.17g, want %.17g", price, want)
			}
			if err := validateResult(got[0], 0, c); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCPUSubnormalPriceRecoversAfterExponentialOverflow(t *testing.T) {
	c := onePathConfig()
	c.DT, c.Steps, c.StorePaths = 1, 2, true
	c.Assets[0] = Asset{InitialPrice: 1}
	c.Regimes = []Regime{{Mu: []float64{-744}, Sigma: []float64{0}}, {Mu: []float64{750}, Sigma: []float64{0}}}
	c.Transition = [][]float64{{0, 1}, {0, 1}}
	got, _ := collect(t, c)
	s := got[0]
	first := referencePrice(1, -744)
	want := referencePrice(first, 750)
	if s.Prices[0][1] != first || math.Abs(s.FinalPrices[0]/want-1) > 1e-15 {
		t.Fatalf("prices %v, want [1, %.17g, %.17g]", s.Prices[0], first, want)
	}
	if err := validateResult(s, 0, c); err != nil {
		t.Fatal(err)
	}
}
