package quantaspindle

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"testing"
)

type precisionBackend func(context.Context, Config, uint64, int) ([]Scenario, error)

func (precisionBackend) Name() string { return "precision-fixture" }
func (b precisionBackend) Batch(ctx context.Context, c Config, first uint64, count int) ([]Scenario, error) {
	return b(ctx, c, first, count)
}

func precisionScenarios(values []Scenario) precisionBackend {
	return func(_ context.Context, c Config, first uint64, count int) ([]Scenario, error) {
		out := make([]Scenario, count)
		for i := range out {
			s := values[int(first)+i]
			s.ID, s.Seed = first+uint64(i), ScenarioSeed(c.Seed, first+uint64(i))
			s.FinalPrices = append([]float64(nil), s.FinalPrices...)
			s.Returns = append([]float64(nil), s.Returns...)
			s.RealizedVolatility = append([]float64(nil), s.RealizedVolatility...)
			s.MaxDrawdown = append([]float64(nil), s.MaxDrawdown...)
			out[i] = s
		}
		return out, nil
	}
}

func exactPrecisionMean(values []float64) float64 {
	sum := new(big.Rat)
	for _, value := range values {
		sum.Add(sum, new(big.Rat).SetFloat64(value))
	}
	sum.Quo(sum, big.NewRat(int64(len(values)), 1))
	mean, _ := sum.Float64()
	return mean
}

func checkPrecisionSummary(t *testing.T, summary RunSummary, values []Scenario) {
	t.Helper()
	if summary.Scenarios != uint64(len(values)) || summary.Backend != "precision-fixture" {
		t.Fatalf("unexpected summary identity: %+v", summary)
	}
	for asset := range values[0].Returns {
		for metric, means := range [][]float64{summary.MeanReturns, summary.MeanRealizedVolatility, summary.MeanMaxDrawdown} {
			observations := make([]float64, len(values))
			for i, s := range values {
				observations[i] = [][]float64{s.Returns, s.RealizedVolatility, s.MaxDrawdown}[metric][asset]
			}
			want := exactPrecisionMean(observations)
			got := means[asset]
			// Allow one rounding step for normal means; subnormal rounding
			// and the all-zero result must match the rational reference exactly.
			tolerance := 0.0
			if math.Abs(want) >= 0x1p-1022 {
				tolerance = math.Abs(want - math.Nextafter(want, 0))
			}
			if !finite(got) || math.Abs(got-want) > tolerance {
				t.Fatalf("asset %d metric %d mean %.17g, want %.17g", asset, metric, got, want)
			}
		}
	}
}

func cancellationScenarios() []Scenario {
	returns := []float64{.5, 1.7763568394002506e-16, -.5, .25}
	prices := []float64{1.875, 1.2500000000000002, .625, 1.5625}
	values := make([]Scenario, len(returns))
	for i, r := range returns {
		values[i] = Scenario{FinalPrices: []float64{prices[i], prices[i]}, Returns: []float64{r, r}, RealizedVolatility: []float64{math.Abs(r), 0}, MaxDrawdown: []float64{math.Max(-r, 0), 0}}
	}
	return values
}

func TestRunMeanCancellationPermutations(t *testing.T) {
	values := cancellationScenarios()[:3]
	c := onePathConfig()
	c.Paths = len(values)
	c.Assets = []Asset{{InitialPrice: 1.25}, {InitialPrice: 1.25}}
	for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		for _, setting := range [][2]int{{1, 1}, {2, 2}, {3, 1}} {
			t.Run(fmt.Sprint(order, setting), func(t *testing.T) {
				ordered := []Scenario{values[order[0]], values[order[1]], values[order[2]]}
				c.Workers, c.BatchSize = setting[0], setting[1]
				summary, err := Run(context.Background(), c, precisionScenarios(ordered), nil)
				if err != nil {
					t.Fatal(err)
				}
				checkPrecisionSummary(t, summary, ordered)
			})
		}
	}
}

func TestRunMeanOverflowAndSubnormalRounding(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	for _, observations := range [][]float64{
		{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64},
		{math.MaxFloat64, math.MaxFloat64, 0},
		{-1, math.MaxFloat64, -1},
		{tiny, tiny, tiny}, {tiny, tiny, 2 * tiny}, {tiny, 2 * tiny, 2 * tiny},
		{0, 0, 0},
	} {
		t.Run(fmt.Sprint(observations), func(t *testing.T) {
			values := make([]Scenario, len(observations))
			for i, value := range observations {
				values[i] = Scenario{FinalPrices: []float64{1}, Returns: []float64{value}, RealizedVolatility: []float64{math.Abs(value)}, MaxDrawdown: []float64{math.Min(math.Abs(value), 1)}}
			}
			c := onePathConfig()
			c.Paths = len(values)
			summary, err := Run(context.Background(), c, precisionScenarios(values), nil)
			if err != nil {
				t.Fatal(err)
			}
			checkPrecisionSummary(t, summary, values)
		})
	}
}

func TestRunCompensatedMeanPartialSummaries(t *testing.T) {
	values := cancellationScenarios()
	for _, stop := range []string{"sink", "cancellation", "backend", "invalid result"} {
		for prefix := 1; prefix <= 3; prefix++ {
			t.Run(fmt.Sprintf("%s/%d", stop, prefix), func(t *testing.T) {
				c := onePathConfig()
				c.Paths = len(values)
				c.Assets = []Asset{{InitialPrice: 1.25}, {InitialPrice: 1.25}}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				sentinel := errors.New("stop precision run")
				backend := precisionScenarios(values)
				wrapped := precisionBackend(func(ctx context.Context, c Config, first uint64, count int) ([]Scenario, error) {
					if stop == "backend" && first == uint64(prefix) {
						return nil, sentinel
					}
					out, err := backend.Batch(ctx, c, first, count)
					if stop == "invalid result" && first == uint64(prefix) {
						out[0].Returns[0] = math.NaN()
					}
					return out, err
				})
				delivered := 0
				summary, err := Run(ctx, c, wrapped, func(s Scenario) error {
					if stop == "sink" && s.ID == uint64(prefix) {
						return sentinel
					}
					delivered++
					if stop == "cancellation" && delivered == prefix {
						cancel()
					}
					// The accumulator must retain values copied before ownership
					// passes to the sink, including on partial-run exits.
					s.Returns[0], s.RealizedVolatility[0], s.MaxDrawdown[0] = math.NaN(), math.NaN(), math.NaN()
					return nil
				})
				if err == nil || delivered != prefix {
					t.Fatalf("delivered %d, want %d; error %v", delivered, prefix, err)
				}
				if (stop == "sink" || stop == "backend") && !errors.Is(err, sentinel) {
					t.Fatalf("lost original failure: %v", err)
				}
				if stop == "cancellation" && !errors.Is(err, context.Canceled) {
					t.Fatalf("lost cancellation: %v", err)
				}
				checkPrecisionSummary(t, summary, values[:prefix])
			})
		}
	}
}

func TestCPUPriceTinyDrift(t *testing.T) {
	for _, initial := range []float64{math.Nextafter(2, 0), 1.25, math.Nextafter(2, math.Inf(1)), 0x1p-500, 0x1p500} {
		for _, drift := range []float64{-1e-9, -1.2e-16, -1e-16, -5e-17, 0, 5e-17, 1e-16, 1.2e-16, 1e-9} {
			t.Run(fmt.Sprintf("%.17g/%g", initial, drift), func(t *testing.T) {
				c := onePathConfig()
				c.DT, c.StorePaths = 1, true
				c.Assets[0] = Asset{InitialPrice: initial, Mu: drift}
				got, summary := collect(t, c)
				want := referencePrice(initial, drift)
				if got[0].FinalPrices[0] != want || got[0].Prices[0][1] != want {
					t.Fatalf("price %.17g, want %.17g", got[0].FinalPrices[0], want)
				}
				wantReturn := (want - initial) / initial
				if got[0].Returns[0] != wantReturn || summary.MeanReturns[0] != wantReturn {
					t.Fatalf("scenario/summary returns %g/%g, want %g", got[0].Returns[0], summary.MeanReturns[0], wantReturn)
				}
			})
		}
	}
}
