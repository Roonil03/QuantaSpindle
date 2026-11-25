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
