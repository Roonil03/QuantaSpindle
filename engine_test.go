package quantaspindle

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync/atomic"
	"testing"
)

func tinyConfig() Config {
	c := DefaultConfig()
	c.Paths = 19
	c.Steps = 12
	c.Workers = 2
	c.BatchSize = 4
	return c
}
func closeTo(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %.15g want %.15g tolerance %.3g", got, want, tol)
	}
}

func TestValidation(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Config)
	}{
		{"zero paths", func(c *Config) { c.Paths = 0 }}, {"negative steps", func(c *Config) { c.Steps = -1 }}, {"huge steps", func(c *Config) { c.Steps = 1_000_001 }},
		{"nan time", func(c *Config) { c.DT = math.NaN() }}, {"infinite time", func(c *Config) { c.DT = math.Inf(1) }}, {"zero workers", func(c *Config) { c.Workers = 0 }}, {"huge workers", func(c *Config) { c.Workers = 1025 }},
		{"zero batch", func(c *Config) { c.BatchSize = 0 }}, {"huge batch", func(c *Config) { c.BatchSize = 16385 }}, {"no assets", func(c *Config) { c.Assets = nil }}, {"price zero", func(c *Config) { c.Assets[0].InitialPrice = 0 }},
		{"sigma negative", func(c *Config) { c.Assets[0].Sigma = -1 }}, {"mu nan", func(c *Config) { c.Assets[0].Mu = math.NaN() }},
		{"bad diagonal", func(c *Config) { c.Correlation = [][]float64{{.9}} }}, {"ragged correlation", func(c *Config) { c.Correlation = [][]float64{{}} }},
		{"transition without regimes", func(c *Config) { c.Transition = [][]float64{{1}} }}, {"state without regimes", func(c *Config) { c.InitialRegime = 1 }},
		{"regime dimensions", func(c *Config) { c.Regimes = []Regime{{Mu: []float64{0}, Sigma: nil}}; c.Transition = [][]float64{{1}} }},
		{"transition probability", func(c *Config) {
			c.Regimes = []Regime{{Mu: []float64{0}, Sigma: []float64{.2}}}
			c.Transition = [][]float64{{.9}}
		}},
		{"huge stored wave", func(c *Config) { c.Paths = 1000; c.Steps = 1_000_000; c.BatchSize = 1000; c.StorePaths = true }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tinyConfig()
			tt.change(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("expected validation error")
			}
			if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err == nil {
				t.Fatal("batch accepted invalid config")
			}
		})
	}
	c := tinyConfig()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		first uint64
		count int
	}{{0, 0}, {19, 1}, {18, 2}, {math.MaxUint64, 1}, {0, 16385}} {
		if _, err := (CPU{}).Batch(context.Background(), c, tc.first, tc.count); err == nil {
			t.Fatal("accepted invalid batch", tc)
		}
	}
}

func collect(t *testing.T, c Config) ([]Scenario, RunSummary) {
	t.Helper()
	var out []Scenario
	s, err := Run(context.Background(), c, CPU{}, func(x Scenario) error { out = append(out, x); return nil })
	if err != nil {
		t.Fatal(err)
	}
	return out, s
}

func TestReproducibleAcrossSchedulingAndBatches(t *testing.T) {
	c := tinyConfig()
	c.StorePaths = true
	c.Regimes = []Regime{{Name: "calm", Mu: []float64{.03}, Sigma: []float64{.1}}, {Name: "stress", Mu: []float64{-.1}, Sigma: []float64{.4}}}
	c.Transition = [][]float64{{.9, .1}, {.2, .8}}
	a, sa := collect(t, c)
	for _, setting := range [][2]int{{1, 1}, {3, 5}, {64, 19}, {2, 7}} {
		c.Workers = setting[0]
		c.BatchSize = setting[1]
		b, sb := collect(t, c)
		if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(sa, sb) {
			t.Fatalf("results changed for workers=%d batch=%d", c.Workers, c.BatchSize)
		}
	}
	c.Seed++
	b, _ := collect(t, c)
	if reflect.DeepEqual(a, b) {
		t.Fatal("seed did not change results")
	}
	for i, s := range a {
		if s.ID != uint64(i) || s.Seed != ScenarioSeed(c.Seed-1, uint64(i)) {
			t.Fatal("scenario identity incorrect")
		}
	}
}

func TestZeroVolatilityAndDrawdown(t *testing.T) {
	c := tinyConfig()
	c.Paths = 1
	c.Steps = 3
	c.DT = 1
	c.Assets[0].Mu = -.2
	c.Assets[0].Sigma = 0
	c.StorePaths = true
	a, _ := collect(t, c)
	s := a[0]
	closeTo(t, s.FinalPrices[0], 100*math.Exp(-.6), 1e-12)
	closeTo(t, s.Returns[0], math.Exp(-.6)-1, 1e-14)
	closeTo(t, s.MaxDrawdown[0], 1-math.Exp(-.6), 1e-14)
	closeTo(t, s.RealizedVolatility[0], 0, 1e-14)
	if len(s.Prices[0]) != 4 || s.Prices[0][0] != 100 {
		t.Fatal("path missing initial value")
	}
	c.StorePaths = false
	a, _ = collect(t, c)
	if a[0].Prices != nil || a[0].Regimes != nil {
		t.Fatal("paths stored by default")
	}
}

func TestFourRegimeTiming(t *testing.T) {
	c := tinyConfig()
	c.Paths = 1
	c.Steps = 4
	c.DT = 1
	c.StorePaths = true
	c.Regimes = []Regime{{Mu: []float64{.1}, Sigma: []float64{0}}, {Mu: []float64{.2}, Sigma: []float64{0}}, {Mu: []float64{.3}, Sigma: []float64{0}}, {Mu: []float64{.4}, Sigma: []float64{0}}}
	c.Transition = [][]float64{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}}
	a, _ := collect(t, c)
	s := a[0]
	if !reflect.DeepEqual(s.Regimes, []int{0, 1, 2, 3, 0}) || s.FinalRegime != 0 {
		t.Fatal("wrong regime timing", s.Regimes)
	}
	closeTo(t, s.FinalPrices[0], 100*math.E, 1e-12)
	closeTo(t, s.RealizedVolatility[0], math.Sqrt(1.0/60), 1e-14)
	c.InitialRegime = 2
	c.Transition = [][]float64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
	a, _ = collect(t, c)
	closeTo(t, a[0].FinalPrices[0], 100*math.Exp(1.2), 1e-12)
}

func TestCorrelationPSDAndValidation(t *testing.T) {
	c := tinyConfig()
	c.Assets = append(c.Assets, c.Assets[0])
	c.Correlation = [][]float64{{1, 1}, {1, 1}}
	a, _ := collect(t, c)
	for _, s := range a {
		if s.FinalPrices[0] != s.FinalPrices[1] {
			t.Fatal("perfect correlation lost")
		}
	}
	c.Correlation = [][]float64{{1, -1}, {-1, 1}}
	if err := c.Validate(); err != nil {
		t.Fatal("singular negative correlation rejected", err)
	}
	c.Correlation = [][]float64{{1, .5}, {.4, 1}}
	if err := c.Validate(); err == nil {
		t.Fatal("asymmetry accepted")
	}
	c.Assets = append(c.Assets, c.Assets[0])
	c.Correlation = [][]float64{{1, .9, .9}, {.9, 1, -.9}, {.9, -.9, 1}}
	if err := c.Validate(); err == nil {
		t.Fatal("indefinite correlation accepted")
	}
}

func TestEmpiricalInnovationCorrelation(t *testing.T) {
	c := tinyConfig()
	c.Paths = 16000
	c.Steps = 1
	c.DT = 1
	c.BatchSize = 128
	c.Assets = []Asset{{InitialPrice: 1, Mu: .5, Sigma: 1}, {InitialPrice: 1, Mu: .5, Sigma: 1}}
	c.Correlation = [][]float64{{1, .65}, {.65, 1}}
	n, sx, sy, sxx, syy, sxy := 0., 0., 0., 0., 0., 0.
	_, err := Run(context.Background(), c, CPU{}, func(s Scenario) error {
		x, y := math.Log(s.FinalPrices[0]), math.Log(s.FinalPrices[1])
		n++
		sx += x
		sy += y
		sxx += x * x
		syy += y * y
		sxy += x * y
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	r := (sxy - sx*sy/n) / math.Sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
	// At rho=.65, asymptotic SE is (1-rho^2)/sqrt(N) ~= .00457;
	// .025 is over five standard errors with this fixed seed.
	closeTo(t, r, .65, .025)
}

func TestAnalyticalGBMMoments(t *testing.T) {
	c := tinyConfig()
	c.Paths = 30000
	c.Steps = 16
	c.DT = 1.0 / 16
	c.BatchSize = 128
	c.Assets[0] = Asset{InitialPrice: 100, Mu: .07, Sigma: .3}
	n, mean, m2 := 0., 0., 0.
	_, err := Run(context.Background(), c, CPU{}, func(s Scenario) error {
		n++
		d := s.FinalPrices[0] - mean
		mean += d / n
		m2 += d * (s.FinalPrices[0] - mean)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := func(k float64) float64 { return math.Pow(100, k) * math.Exp(k*.07+.5*k*(k-1)*.09) }
	wantMean := raw(1)
	wantVariance := raw(2) - wantMean*wantMean
	mu4 := raw(4) - 4*wantMean*raw(3) + 6*wantMean*wantMean*raw(2) - 3*math.Pow(wantMean, 4)
	// Predeclared six-standard-error tolerances for sample mean and variance,
	// using the exact lognormal central fourth moment (fixed seed, no tuning).
	closeTo(t, mean, wantMean, 6*math.Sqrt(wantVariance/n))
	closeTo(t, m2/(n-1), wantVariance, 6*math.Sqrt((mu4-wantVariance*wantVariance)/n))
}

func TestCancellationSinkFailureAndOwnership(t *testing.T) {
	c := tinyConfig()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s, err := Run(ctx, c, CPU{}, nil)
	if !errors.Is(err, context.Canceled) || s.Scenarios != 0 {
		t.Fatal("pre-cancellation ignored", s, err)
	}
	sentinel := errors.New("sink stopped")
	s, err = Run(context.Background(), c, CPU{}, func(s Scenario) error {
		if s.ID == 3 {
			return sentinel
		}
		return nil
	})
	if !errors.Is(err, sentinel) || s.Scenarios != 3 {
		t.Fatal("wrong sink failure prefix", s, err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	s, err = Run(ctx, c, CPU{}, func(s Scenario) error { calls++; cancel(); return nil })
	if !errors.Is(err, context.Canceled) || calls != 1 || s.Scenarios != 1 {
		t.Fatal("cancellation during sink ignored", s, err)
	}
	_, want := collect(t, c)
	s, err = Run(context.Background(), c, CPU{}, func(s Scenario) error {
		s.Returns[0] = math.NaN()
		s.RealizedVolatility[0] = math.NaN()
		s.MaxDrawdown[0] = math.NaN()
		return nil
	})
	if err != nil || !reflect.DeepEqual(s, want) {
		t.Fatal("sink mutation corrupted summary", s, err)
	}
}

type trackingBackend struct {
	active atomic.Int32
	peak   atomic.Int32
}

func (b *trackingBackend) Name() string { return "tracking" }
func (b *trackingBackend) Batch(ctx context.Context, c Config, first uint64, count int) ([]Scenario, error) {
	n := b.active.Add(1)
	defer b.active.Add(-1)
	for {
		old := b.peak.Load()
		if n <= old || b.peak.CompareAndSwap(old, n) {
			break
		}
	}
	return (CPU{}).Batch(ctx, c, first, count)
}

func TestWorkerBoundAndOverflow(t *testing.T) {
	c := tinyConfig()
	b := &trackingBackend{}
	if _, err := Run(context.Background(), c, b, nil); err != nil {
		t.Fatal(err)
	}
	if b.peak.Load() > int32(c.Workers) || b.active.Load() != 0 {
		t.Fatal("worker bound violated")
	}
	c.Assets[0].Mu = 1e308
	if _, err := (CPU{}).Batch(context.Background(), c, 0, 1); err == nil {
		t.Fatal("numeric overflow accepted")
	}
	if _, err := Run(context.Background(), tinyConfig(), nil, nil); err == nil {
		t.Fatal("nil backend accepted")
	}
	if _, err := Run(nil, tinyConfig(), CPU{}, nil); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := (CPU{}).Batch(nil, tinyConfig(), 0, 1); err == nil {
		t.Fatal("nil context accepted")
	}
	if err := DefaultRegimeConfig().Validate(); err != nil {
		t.Fatal("regime defaults invalid", err)
	}
}
