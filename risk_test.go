package quantaspindle

import (
	"math"
	"reflect"
	"testing"
)

func TestRiskConventions(t *testing.T) {
	x := []float64{4, -1, 3, 0, 2, 1, 5, 6, 7, 8}
	before := append([]float64(nil), x...)
	r, err := RiskFromLosses(x, .7)
	if err != nil {
		t.Fatal(err)
	}
	if r.VaR != 5 || r.ExpectedShortfall != 7 || r.TailCount != 3 {
		t.Fatal("wrong nearest-rank or discrete tail convention", r)
	}
	if !reflect.DeepEqual(x, before) {
		t.Fatal("mutated caller data")
	}
	r, err = RiskFromReturns([]float64{.1, .2, .3}, .95)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, r.VaR, -.1, 1e-15)
	closeTo(t, r.ExpectedShortfall, -.1, 1e-15)
	r, err = RiskFromLosses([]float64{2}, .95)
	if err != nil || r.StdDevLoss != 0 || r.VaR != 2 || r.ExpectedShortfall != 2 {
		t.Fatal("singleton", r, err)
	}
	for _, x := range [][]float64{nil, {math.NaN()}, {math.Inf(1)}} {
		if _, err := RiskFromLosses(x, .95); err == nil {
			t.Fatal("invalid sample accepted")
		}
	}
	for _, alpha := range []float64{0, 1, -1, math.NaN(), math.Inf(1)} {
		if _, err := RiskFromLosses([]float64{1}, alpha); err == nil {
			t.Fatal("invalid confidence accepted")
		}
	}
	loss, err := PortfolioLoss([]float64{100, 50}, []float64{90, 60}, []float64{2, -1})
	if err != nil || loss != 30 {
		t.Fatal("portfolio loss", loss, err)
	}
	if _, err := PortfolioLoss([]float64{1}, nil, nil); err == nil {
		t.Fatal("portfolio dimensions accepted")
	}
}

func TestRiskStandardDeviationAcrossScales(t *testing.T) {
	next := math.Nextafter(1, 2)
	large := 1e150
	nextLarge := math.Nextafter(large, math.Inf(1))
	tests := []struct {
		name   string
		losses []float64
		want   float64
	}{
		{"tiny symmetric", []float64{-1e-170, 1e-170}, math.Sqrt2 * 1e-170},
		{"tiny asymmetric", []float64{1e-170, 2e-170, 3e-170}, 1e-170},
		{"tiny offset", []float64{1e-170, math.Nextafter(1e-170, 1)}, (math.Nextafter(1e-170, 1) - 1e-170) / math.Sqrt2},
		{"adjacent", []float64{1, next}, (next - 1) / math.Sqrt2},
		{"large offset", []float64{large, nextLarge}, (nextLarge - large) / math.Sqrt2},
		{"large symmetric", []float64{-1e170, 1e170}, math.Sqrt2 * 1e170},
		{"subnormal", []float64{-1e-310, 1e-310}, math.Sqrt2 * 1e-310},
		{"smallest subnormal", []float64{0, math.SmallestNonzeroFloat64}, math.SmallestNonzeroFloat64},
		{"constant", []float64{1e-170, 1e-170, 1e-170}, 0},
		{"zero", []float64{0, 0}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, reverse := range []bool{false, true} {
				losses := append([]float64(nil), tt.losses...)
				if reverse {
					for i, j := 0, len(losses)-1; i < j; i, j = i+1, j-1 {
						losses[i], losses[j] = losses[j], losses[i]
					}
				}
				r, err := RiskFromLosses(losses, .95)
				if err != nil {
					t.Fatal(err)
				}
				if tt.want == 0 {
					if r.StdDevLoss != 0 {
						t.Fatalf("StdDevLoss = %g, want zero", r.StdDevLoss)
					}
				} else if !finite(r.StdDevLoss) || math.Abs(r.StdDevLoss-tt.want)/tt.want > 4e-15 {
					t.Fatalf("StdDevLoss = %.17g, want %.17g (reverse=%v)", r.StdDevLoss, tt.want, reverse)
				}
				returns, err := RiskFromReturns(losses, .95)
				if err != nil || returns.StdDevLoss != r.StdDevLoss {
					t.Fatalf("RiskFromReturns stddev = %g, want %g; error = %v", returns.StdDevLoss, r.StdDevLoss, err)
				}
			}
		})
	}
}

func TestRiskMomentsNearFloatLimit(t *testing.T) {
	tests := []struct {
		name                  string
		losses                []float64
		confidence            float64
		mean, stddev, varLoss float64
		shortfall             float64
		tailCount             int
	}{
		{"opposite signs", []float64{-1e308, 1e308}, .5, 0, math.Sqrt2 * 1e308, -1e308, 1e308, 1},
		{"opposite signs entire tail", []float64{-1e308, 1e308}, .25, 0, math.Sqrt2 * 1e308, -1e308, 0, 2},
		{"positive mean", []float64{-1e308, 1e308, 1e308}, .1, 1e308 / 3, 2 / math.Sqrt(3) * 1e308, -1e308, 1e308 / 3, 3},
		{"negative mean", []float64{-1e308, -1e308, 1e308}, .1, -1e308 / 3, 2 / math.Sqrt(3) * 1e308, -1e308, -1e308 / 3, 3},
		{"cancellation residual", []float64{-1e308, 3, 1e308}, .1, 1, 1e308, -1e308, 1, 3},
		{"mixed-sign tail", []float64{-1e308, -1e308, 1e308, 1e308}, .25, 0, math.Sqrt(4.0/3) * 1e308, -1e308, 1e308 / 3, 3},
		{"maximum constant", []float64{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64}, .25, math.MaxFloat64, 0, math.MaxFloat64, math.MaxFloat64, 3},
		{"minimum constant", []float64{-math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64}, .25, -math.MaxFloat64, 0, -math.MaxFloat64, -math.MaxFloat64, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Every ordering must succeed: the running mean must not overflow
			// merely because opposite-sign observations arrive consecutively.
			var checkPermutations func([]float64, int)
			checkPermutations = func(losses []float64, start int) {
				if start < len(losses) {
					for i := start; i < len(losses); i++ {
						losses[start], losses[i] = losses[i], losses[start]
						checkPermutations(losses, start+1)
						losses[start], losses[i] = losses[i], losses[start]
					}
					return
				}
				r, err := RiskFromLosses(losses, tt.confidence)
				if err != nil {
					t.Fatalf("losses %v: %v", losses, err)
				}
				for _, field := range []struct {
					name      string
					got, want float64
				}{
					{"mean", r.MeanLoss, tt.mean},
					{"standard deviation", r.StdDevLoss, tt.stddev},
					{"VaR", r.VaR, tt.varLoss},
					{"expected shortfall", r.ExpectedShortfall, tt.shortfall},
				} {
					if !finite(field.got) || (field.want == 0 && field.got != 0) || (field.want != 0 && math.Abs((field.got-field.want)/field.want) > 4e-15) {
						t.Errorf("losses %v: %s = %.17g, want %.17g", losses, field.name, field.got, field.want)
					}
				}
				if r.Count != len(losses) || r.TailCount != tt.tailCount || r.Confidence != tt.confidence {
					t.Errorf("wrong risk counts/confidence: %+v", r)
				}
				returns := make([]float64, len(losses))
				for i, loss := range losses {
					returns[i] = -loss
				}
				fromReturns, err := RiskFromReturns(returns, tt.confidence)
				if err != nil || fromReturns != r {
					t.Errorf("RiskFromReturns = %+v, %v; want %+v", fromReturns, err, r)
				}
			}
			checkPermutations(append([]float64(nil), tt.losses...), 0)
		})
	}
	if _, err := RiskFromLosses([]float64{-math.MaxFloat64, math.MaxFloat64}, .5); err == nil {
		t.Fatal("unrepresentable standard deviation accepted")
	}
}

func TestRiskSubnormalMeanRounding(t *testing.T) {
	for _, losses := range [][]float64{
		{7 * math.SmallestNonzeroFloat64, 0, 0},
		{0, 7 * math.SmallestNonzeroFloat64, 0},
		{0, 0, 7 * math.SmallestNonzeroFloat64},
		{-7 * math.SmallestNonzeroFloat64, 0, 0},
		{0, -7 * math.SmallestNonzeroFloat64, 0},
		{0, 0, -7 * math.SmallestNonzeroFloat64},
	} {
		r, err := RiskFromLosses(losses, .1)
		if err != nil {
			t.Fatal(err)
		}
		want := 2 * math.SmallestNonzeroFloat64
		if losses[0] < 0 || losses[1] < 0 || losses[2] < 0 {
			want = -want
		}
		if r.MeanLoss != want || r.ExpectedShortfall != want {
			t.Errorf("losses %v: mean/shortfall = %g/%g, want %g", losses, r.MeanLoss, r.ExpectedShortfall, want)
		}
	}
}

func TestRiskOrdinaryMeansAndShortfalls(t *testing.T) {
	for _, tt := range []struct {
		losses          []float64
		mean, shortfall float64
	}{
		{[]float64{1, 2, 3, 4}, 2.5, 3.5},
		{[]float64{-4, -3, -2, -1}, -2.5, -1.5},
		{[]float64{-4, -2, 1, 3}, -.5, 2},
		{[]float64{.1, .2, .3, .4}, .25, .35},
		{[]float64{math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64}, math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
	} {
		r, err := RiskFromLosses(tt.losses, .5)
		if err != nil {
			t.Fatal(err)
		}
		if !finite(r.MeanLoss) || !finite(r.ExpectedShortfall) || math.Abs(r.MeanLoss-tt.mean) > 4e-15*math.Abs(tt.mean) || math.Abs(r.ExpectedShortfall-tt.shortfall) > 4e-15*math.Abs(tt.shortfall) {
			t.Errorf("losses %v: mean/shortfall = %.17g/%.17g, want %.17g/%.17g", tt.losses, r.MeanLoss, r.ExpectedShortfall, tt.mean, tt.shortfall)
		}
	}
}

func BenchmarkRiskFromLossesMoments(b *testing.B) {
	for _, size := range []int{16, 1024} {
		losses := make([]float64, size)
		for i := range losses {
			losses[i] = float64((i*37)%101-50) / 100
		}
		name := "16"
		if size == 1024 {
			name = "1024"
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, err := RiskFromLosses(losses, .95); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestSquareRootImpact(t *testing.T) {
	m := SquareRootImpact{Volatility: .02, DailyVolume: 10000, Temporary: .5, Permanent: .25, HalfSpread: .001}
	b, err := m.Execute(100, 100)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, b.ExecutionPrice, 100.2, 1e-12)
	closeTo(t, b.PostTradeMid, 100.05, 1e-12)
	closeTo(t, b.CashChange, -10020, 1e-10)
	closeTo(t, b.Cost, 20, 1e-10)
	if b.DailyVolume != 10000 || b.HalfSpread != .001 {
		t.Fatal("missing liquidity/spread output", b)
	}
	closeTo(t, b.ParticipationRate, .01, 1e-15)
	closeTo(t, b.TemporaryImpact, .001, 1e-15)
	closeTo(t, b.PermanentImpact, .0005, 1e-15)
	s, err := m.Execute(100, -100)
	if err != nil {
		t.Fatal(err)
	}
	closeTo(t, s.ExecutionPrice, 99.8, 1e-12)
	closeTo(t, s.PostTradeMid, 99.95, 1e-12)
	closeTo(t, s.CashChange, 9980, 1e-10)
	closeTo(t, s.Cost, 20, 1e-10)
	z, err := (SquareRootImpact{DailyVolume: 1}).Execute(100, -100)
	if err != nil || z.ExecutionPrice != 100 || z.PostTradeMid != 100 || z.Cost != 0 {
		t.Fatal("zero coefficient baseline", z, err)
	}
	z, err = m.Execute(100, 0)
	if err != nil || z.ExecutionPrice != 100 || z.CashChange != 0 {
		t.Fatal("zero trade", z, err)
	}
	for _, bad := range []SquareRootImpact{{}, {DailyVolume: 1, Temporary: -1}, {DailyVolume: 1, Volatility: math.NaN()}, {DailyVolume: 1, HalfSpread: math.Inf(1)}} {
		if _, err := bad.Execute(100, 1); err == nil {
			t.Fatal("invalid impact accepted")
		}
	}
	if _, err := (SquareRootImpact{DailyVolume: 1, HalfSpread: 2}).Execute(100, -1); err == nil {
		t.Fatal("negative execution price accepted")
	}
}
