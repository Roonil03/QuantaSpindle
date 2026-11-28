package quantaspindle

import (
	"context"
	"fmt"
)

// PreparedCPU generates arbitrary batches using one validated configuration and
// correlation factor. It owns a snapshot of all model inputs; changes to the
// original Config after Prepare returns do not affect it. Batch calls may run
// concurrently and return independently owned results. No Close is required.
// A nil or zero PreparedCPU is unusable; construct one with CPU.Prepare.
//
// PreparedCPU is a direct-batch API bound to its configuration, rather than a
// Backend accepting a new Config per call. Run already reuses its CPU factor.
type PreparedCPU struct {
	config Config
	factor [][]float64
}

// Prepare validates c and computes its correlation factor once. The caller must
// not mutate c or its slices until Prepare returns. The same model and configured
// wave storage checks as CPU.Batch apply. The context covers preparation only;
// each subsequent Batch observes its own context. Include this setup cost when
// measuring a complete sequence of prepared calls.
func (CPU) Prepare(ctx context.Context, c Config) (*PreparedCPU, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}
	if err := c.validateDimensions(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	factor, err := correlationFactor(c.Correlation, len(c.Assets))
	if err != nil {
		return nil, err
	}
	// The correlation matrix is replaced by its owned factor, not retained.
	c.Correlation = nil
	c.Assets = append([]Asset(nil), c.Assets...)
	c.Regimes = append([]Regime(nil), c.Regimes...)
	for i := range c.Regimes {
		c.Regimes[i].Mu = append([]float64(nil), c.Regimes[i].Mu...)
		c.Regimes[i].Sigma = append([]float64(nil), c.Regimes[i].Sigma...)
	}
	c.Transition = append([][]float64(nil), c.Transition...)
	for i := range c.Transition {
		c.Transition[i] = append([]float64(nil), c.Transition[i]...)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &PreparedCPU{config: c, factor: factor}, nil
}

// Batch generates count consecutive IDs beginning at first within the snapshot's
// [0, Paths) ID space. Calls may arrive in any order, overlap, or repeat IDs, and
// match CPU.Batch for the prepared configuration. Count is limited to 16384 and
// each call's estimated storage (including the shared factor) to 512 MiB.
// Callers bound aggregate memory by limiting concurrent calls and retained
// results. Cancellation or a numerical error returns no partial batch and does
// not invalidate the prepared object or interrupt other calls.
func (p *PreparedCPU) Batch(ctx context.Context, first uint64, count int) ([]Scenario, error) {
	if p == nil || len(p.factor) == 0 {
		return nil, fmt.Errorf("CPU must be prepared before batching")
	}
	if err := validateCPUBatch(ctx, p.config, first, count); err != nil {
		return nil, err
	}
	return cpuBatch(ctx, p.config, p.factor, first, count)
}

func validateCPUBatch(ctx context.Context, c Config, first uint64, count int) error {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if count < 1 || count > 16384 || first >= uint64(c.Paths) || uint64(count) > uint64(c.Paths)-first {
		return fmt.Errorf("batch IDs/count outside configured paths or batch limit")
	}
	if uint64(count)*scenarioBytes(c)+factorBytes(len(c.Assets)) > maxWaveBytes {
		return fmt.Errorf("batch exceeds 512 MiB storage limit")
	}
	return nil
}
