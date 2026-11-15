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
