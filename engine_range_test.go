package quantaspindle

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestRunRangeReplaysAbsoluteIDsAcrossScheduling(t *testing.T) {
	c := DefaultRegimeConfig()
	c.Paths, c.Steps, c.Workers, c.BatchSize = 8, 12, 2, 3
	c.StorePaths = true
	full, fullSummary := collect(t, c)
	for _, setting := range [][2]int{{1, 1}, {2, 2}, {3, 5}, {8, 8}} {
		c.Workers, c.BatchSize = setting[0], setting[1]
		var joined []Scenario
		var summaries []RunSummary
		for _, bounds := range [][2]uint64{{0, 3}, {3, 5}} {
			summary, err := RunRange(context.Background(), c, CPU{}, bounds[0], bounds[1], func(s Scenario) error {
				joined = append(joined, s)
				return nil
			})
			if err != nil || summary.Scenarios != bounds[1] {
				t.Fatalf("range %v: summary=%+v err=%v", bounds, summary, err)
			}
			summaries = append(summaries, summary)
		}
		if !reflect.DeepEqual(joined, full) {
			t.Fatalf("ranges changed scenarios with workers=%d batch=%d", c.Workers, c.BatchSize)
		}
		for _, metrics := range [][3][]float64{
			{summaries[0].MeanReturns, summaries[1].MeanReturns, fullSummary.MeanReturns},
			{summaries[0].MeanRealizedVolatility, summaries[1].MeanRealizedVolatility, fullSummary.MeanRealizedVolatility},
			{summaries[0].MeanMaxDrawdown, summaries[1].MeanMaxDrawdown, fullSummary.MeanMaxDrawdown},
		} {
			for i := range metrics[2] {
				weighted := (3*metrics[0][i] + 5*metrics[1][i]) / 8
				closeTo(t, weighted, metrics[2][i], 1e-15)
			}
		}
	}
}

func TestRangeValidationRejectsInvalidAndOverflowingBounds(t *testing.T) {
	c := tinyConfig()
	c.Paths = 8
	for _, bounds := range [][2]uint64{{0, 0}, {8, 1}, {9, 1}, {7, 2}, {math.MaxUint64, 1}, {1, math.MaxUint64}, {math.MaxUint64, math.MaxUint64}} {
		if err := c.ValidateRange(bounds[0], bounds[1]); err == nil {
			t.Fatalf("accepted range %v", bounds)
		}
		called := false
		summary, err := RunRange(context.Background(), c, CPU{}, bounds[0], bounds[1], func(Scenario) error { called = true; return nil })
		if err == nil || called || summary.Scenarios != 0 {
			t.Fatalf("invalid range ran: %v, summary=%+v err=%v", bounds, summary, err)
		}
	}
	c.Correlation = [][]float64{{0.5}}
	if err := c.ValidateRange(3, 1); err == nil {
		t.Fatal("range skipped model validation")
	}
	if _, err := RunRange(context.Background(), c, CPU{}, 3, 1, nil); err == nil {
		t.Fatal("run range skipped correlation validation")
	}
}

func TestRangeResourceLimitUsesSelectedCount(t *testing.T) {
	c := DefaultConfig()
	c.Paths, c.Steps, c.Workers, c.BatchSize = 1_000_000, 10_000, 1024, 64
	c.StorePaths = true
	if err := c.Validate(); err == nil {
		t.Fatal("full run should exceed wave storage limit")
	}
	if err := c.ValidateRange(999999, 1); err != nil {
		t.Fatalf("small range assessed using total paths: %v", err)
	}
	var delivered []uint64
	summary, err := RunRange(context.Background(), c, CPU{}, 999999, 1, func(s Scenario) error {
		delivered = append(delivered, s.ID)
		return nil
	})
	if err != nil || summary.Scenarios != 1 || !reflect.DeepEqual(delivered, []uint64{999999}) {
		t.Fatalf("small range failed: summary=%+v ids=%v err=%v", summary, delivered, err)
	}
	if err := c.ValidateRange(0, 4000); err == nil || !strings.Contains(err.Error(), "512 MiB") {
		t.Fatalf("large range escaped wave limit: %v", err)
	}
	if _, err := RunRange(context.Background(), c, CPU{}, 0, 4000, nil); err == nil {
		t.Fatal("oversized range ran")
	}
}

func TestRunRangeCancellationAndSinkFailureReturnPrefix(t *testing.T) {
	c := tinyConfig()
	for _, cancelRun := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		sentinel := errors.New("sink stopped")
		var ids []uint64
		summary, err := RunRange(ctx, c, CPU{}, 4, 5, func(s Scenario) error {
			if !cancelRun && s.ID == 6 {
				return sentinel
			}
			ids = append(ids, s.ID)
			if cancelRun && len(ids) == 2 {
				cancel()
			}
			return nil
		})
		cancel()
		wantErr := sentinel
		if cancelRun {
			wantErr = context.Canceled
		}
		if !errors.Is(err, wantErr) || summary.Scenarios != 2 || !reflect.DeepEqual(ids, []uint64{4, 5}) {
			t.Fatalf("range prefix: ids=%v summary=%+v err=%v", ids, summary, err)
		}
	}
}

func TestRangeWaveStorageCountsPartialFinalBatch(t *testing.T) {
	c := DefaultConfig()
	c.Paths, c.Steps, c.BatchSize, c.Workers = 1000, 400000, 64, 2
	c.StorePaths = true
	if err := c.ValidateRange(100, 65); err != nil {
		t.Fatalf("65 stored scenarios fit the wave limit: %v", err)
	}
	if err := c.ValidateRange(100, 84); err == nil || !strings.Contains(err.Error(), "512 MiB") {
		t.Fatalf("84 stored scenarios exceed the wave limit: %v", err)
	}
	if err := c.Validate(); err == nil {
		t.Fatal("ordinary full-run storage validation changed")
	}
}
