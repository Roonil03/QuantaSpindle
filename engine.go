package quantaspindle

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
)

// Scenario contains independently owned results. Prices, when requested, are
// asset-major, including time zero. Regimes contains the initial state and each
// post-interval state; FinalRegime is that last state even without stored paths.
type Scenario struct {
	ID                 uint64
	Seed               uint64
	FinalPrices        []float64
	Returns            []float64
	RealizedVolatility []float64
	MaxDrawdown        []float64
	Prices             [][]float64
	Regimes            []int
	FinalRegime        int
}

// Backend generates count consecutive scenario IDs beginning at first. Results
// must be ordered, complete, finite and own their slices. Implementations must
// respect cancellation and must support concurrent Batch calls. Batch must not
// retain or mutate Config; callers must not mutate Config during a run.
type Backend interface {
	Name() string
	Batch(context.Context, Config, uint64, int) ([]Scenario, error)
}

// BackendSession owns resources for one Run. Close must release them even after
// cancellation. Batch retains the Backend concurrency and ownership contract.
type BackendSession interface {
	Backend
	Close() error
}

// RunBackend optionally prepares a backend once for all batches in a Run.
// Separate runs must receive independently owned sessions.
type RunBackend interface {
	BeginRun(context.Context, Config) (BackendSession, error)
}

// CPU is the portable, deterministic SplitMix64-v1/Box-Muller backend.
type CPU struct{}

// Name identifies the CPU backend and its replay algorithm version.
func (CPU) Name() string { return "cpu-splitmix64-v1-boxmuller" }

// Batch generates a bounded consecutive batch without launching goroutines.
func (CPU) Batch(ctx context.Context, c Config, first uint64, count int) ([]Scenario, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}
	if err := c.validateDimensions(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if count < 1 || count > 16384 || first >= uint64(c.Paths) || uint64(count) > uint64(c.Paths)-first {
		return nil, fmt.Errorf("batch IDs/count outside configured paths or batch limit")
	}
	if uint64(count)*scenarioBytes(c)+factorBytes(len(c.Assets)) > maxWaveBytes {
		return nil, fmt.Errorf("batch exceeds 512 MiB storage limit")
	}
	l, err := correlationFactor(c.Correlation, len(c.Assets))
	if err != nil {
		return nil, err
	}
	return cpuBatch(ctx, c, l, first, count)
}

// cpuBatch uses the immutable factor prepared once for this Run or Batch.
func cpuBatch(ctx context.Context, c Config, l [][]float64, first uint64, count int) ([]Scenario, error) {
	out := make([]Scenario, count)
	for i := range out {
		s, e := simulate(ctx, c, l, first+uint64(i))
		if e != nil {
			return nil, e
		}
		out[i] = s
	}
	return out, nil
}

func simulate(ctx context.Context, c Config, l [][]float64, id uint64) (Scenario, error) {
	n := len(c.Assets)
	s := Scenario{ID: id, Seed: ScenarioSeed(c.Seed, id), FinalPrices: make([]float64, n), Returns: make([]float64, n), RealizedVolatility: make([]float64, n), MaxDrawdown: make([]float64, n)}
	prices := s.FinalPrices
	peaks := make([]float64, n)
	means := make([]float64, n)
	m2 := make([]float64, n)
	eps := make([]float64, n)
	for i, a := range c.Assets {
		prices[i] = a.InitialPrice
		peaks[i] = a.InitialPrice
	}
	if c.StorePaths {
		s.Prices = make([][]float64, n)
		for i := range s.Prices {
			s.Prices[i] = make([]float64, c.Steps+1)
			s.Prices[i][0] = prices[i]
		}
		s.Regimes = make([]int, c.Steps+1)
		s.Regimes[0] = c.InitialRegime
	}
	rng := random{state: s.Seed}
	chain := random{state: mix(s.Seed ^ 0xd1b54a32d192ed03)}
	state := c.InitialRegime
	sqrtDT := math.Sqrt(c.DT)
	for step := 0; step < c.Steps; step++ {
		if step%64 == 0 {
			if err := ctx.Err(); err != nil {
				return Scenario{}, err
			}
		}
		for i := range eps {
			eps[i] = rng.normal()
		}
		for i, a := range c.Assets {
			mu, sigma := a.Mu, a.Sigma
			if len(c.Regimes) > 0 {
				mu = c.Regimes[state].Mu[i]
				sigma = c.Regimes[state].Sigma[i]
			}
			z := 0.0
			for j := 0; j <= i; j++ {
				z += l[i][j] * eps[j]
			}
			lr := (mu-0.5*sigma*sigma)*c.DT + sigma*sqrtDT*z
			p := prices[i] * math.Exp(lr)
			if !finite(lr) || !finite(p) || p <= 0 {
				return Scenario{}, fmt.Errorf("scenario %d step %d asset %d numerical overflow/underflow", id, step, i)
			}
			prices[i] = p
			delta := lr - means[i]
			means[i] += delta / float64(step+1)
			m2[i] += delta * (lr - means[i])
			if p > peaks[i] {
				peaks[i] = p
			}
			dd := (peaks[i] - p) / peaks[i]
			if dd > s.MaxDrawdown[i] {
				s.MaxDrawdown[i] = dd
			}
			if c.StorePaths {
				s.Prices[i][step+1] = p
			}
		}
		if len(c.Regimes) > 0 {
			u := chain.uniform()
			sum := 0.0
			next := len(c.Regimes) - 1
			for j, p := range c.Transition[state] {
				sum += p
				if u < sum {
					next = j
					break
				}
			}
			state = next
		}
		if c.StorePaths {
			s.Regimes[step+1] = state
		}
	}
	s.FinalRegime = state
	for i, a := range c.Assets {
		s.Returns[i] = prices[i]/a.InitialPrice - 1
		if c.Steps > 1 {
			s.RealizedVolatility[i] = math.Sqrt(math.Max(m2[i], 0) / float64(c.Steps-1) / c.DT)
		}
		if !finite(s.Returns[i]) || !finite(s.RealizedVolatility[i]) {
			return Scenario{}, fmt.Errorf("scenario %d asset %d nonfinite summary", id, i)
		}
	}
	return s, nil
}

// RunSummary contains means across scenarios delivered successfully to the sink.
// It retains no per-scenario observations. Exact tail risk requires explicit
// caller retention and RiskFromReturns or RiskFromLosses.
type RunSummary struct {
	Scenarios              uint64
	MeanReturns            []float64
	MeanRealizedVolatility []float64
	MeanMaxDrawdown        []float64
	Backend                string
}

// Run generates bounded waves of concurrent batches and delivers scenarios to
// sink serially in ascending ID order. A nil sink discards individual results.
// On error it returns only the successfully delivered prefix summary. The input
// Config and its slices must remain immutable until Run returns.
func Run(ctx context.Context, c Config, backend Backend, sink func(Scenario) error) (summary RunSummary, runErr error) {
	if ctx == nil {
		return summary, fmt.Errorf("context is nil")
	}
	if err := c.validateDimensions(); err != nil {
		return summary, err
	}
	if backend == nil {
		return summary, fmt.Errorf("backend is nil")
	}
	factor, err := correlationFactor(c.Correlation, len(c.Assets))
	if err != nil {
		return summary, err
	}
	if err := ctx.Err(); err != nil {
		return summary, err
	}
	if scoped, ok := backend.(RunBackend); ok {
		session, err := scoped.BeginRun(ctx, c)
		if err != nil {
			return summary, err
		}
		if session == nil {
			return summary, fmt.Errorf("backend returned nil run session")
		}
		defer func() { runErr = errors.Join(runErr, session.Close()) }()
		backend = session
	}
	// Only the built-in CPU backend can consume this factor directly. Custom
	// backends keep their public Batch contract and manage their own resources.
	_, cpuValue := backend.(CPU)
	_, cpuPointer := backend.(*CPU)
	summary.Backend = backend.Name()
	summary.MeanReturns = make([]float64, len(c.Assets))
	summary.MeanRealizedVolatility = make([]float64, len(c.Assets))
	summary.MeanMaxDrawdown = make([]float64, len(c.Assets))
	metrics := make([]float64, 3*len(c.Assets))
	for first := 0; first < c.Paths; {
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		nb := min(c.Workers, (c.Paths-first+c.BatchSize-1)/c.BatchSize)
		results := make([][]Scenario, nb)
		errs := make([]error, nb)
		var wg sync.WaitGroup
		waveCtx, cancel := context.WithCancel(ctx)
		for b := 0; b < nb; b++ {
			start := first + b*c.BatchSize
			count := min(c.BatchSize, c.Paths-start)
			wg.Add(1)
			go func(b, start, count int) {
				defer wg.Done()
				if cpuValue || cpuPointer {
					results[b], errs[b] = cpuBatch(waveCtx, c, factor, uint64(start), count)
				} else {
					results[b], errs[b] = backend.Batch(waveCtx, c, uint64(start), count)
				}
				if errs[b] != nil {
					cancel()
				}
			}(b, start, count)
		}
		wg.Wait()
		cancel()
		// Prefer a real backend failure to cancellation induced by another batch.
		for _, err := range errs {
			if err != nil && !errors.Is(err, context.Canceled) {
				return summary, err
			}
		}
		for _, err := range errs {
			if err != nil {
				return summary, err
			}
		}
		for b, batch := range results {
			expected := min(c.BatchSize, c.Paths-(first+b*c.BatchSize))
			if len(batch) != expected {
				return summary, fmt.Errorf("backend returned %d scenarios, expected %d", len(batch), expected)
			}
			for j, s := range batch {
				if err := ctx.Err(); err != nil {
					return summary, err
				}
				if err := validateResult(s, uint64(first+b*c.BatchSize+j), c); err != nil {
					return summary, err
				}
				// Preserve metrics before transferring slice ownership to the sink.
				n := len(c.Assets)
				copy(metrics[:n], s.Returns)
				copy(metrics[n:2*n], s.RealizedVolatility)
				copy(metrics[2*n:], s.MaxDrawdown)
				if sink != nil {
					if err := sink(s); err != nil {
						return summary, fmt.Errorf("sink scenario %d: %w", s.ID, err)
					}
				}
				summary.Scenarios++
				k := float64(summary.Scenarios)
				for i := range summary.MeanReturns {
					summary.MeanReturns[i] += (metrics[i] - summary.MeanReturns[i]) / k
					summary.MeanRealizedVolatility[i] += (metrics[n+i] - summary.MeanRealizedVolatility[i]) / k
					summary.MeanMaxDrawdown[i] += (metrics[2*n+i] - summary.MeanMaxDrawdown[i]) / k
				}
			}
		}
		first += nb * c.BatchSize
	}
	return summary, nil
}

func validateResult(s Scenario, id uint64, c Config) error {
	n := len(c.Assets)
	if s.ID != id || len(s.FinalPrices) != n || len(s.Returns) != n || len(s.RealizedVolatility) != n || len(s.MaxDrawdown) != n {
		return fmt.Errorf("backend returned malformed scenario %d", id)
	}
	for i := 0; i < n; i++ {
		if !finite(s.FinalPrices[i]) || s.FinalPrices[i] <= 0 || !finite(s.Returns[i]) || s.Returns[i] < -1 || !finite(s.RealizedVolatility[i]) || s.RealizedVolatility[i] < 0 || !finite(s.MaxDrawdown[i]) || s.MaxDrawdown[i] < 0 || s.MaxDrawdown[i] > 1 {
			return fmt.Errorf("backend returned invalid scenario %d asset %d", id, i)
		}
		if c.StorePaths {
			if len(s.Prices) != n || len(s.Prices[i]) != c.Steps+1 || s.Prices[i][0] != c.Assets[i].InitialPrice || s.Prices[i][c.Steps] != s.FinalPrices[i] {
				return fmt.Errorf("backend returned malformed price path for scenario %d asset %d", id, i)
			}
			for _, price := range s.Prices[i] {
				if !finite(price) || price <= 0 {
					return fmt.Errorf("backend returned invalid price path for scenario %d asset %d", id, i)
				}
			}
		}
	}
	states := max(1, len(c.Regimes))
	if s.FinalRegime < 0 || s.FinalRegime >= states {
		return fmt.Errorf("backend returned invalid final regime for scenario %d", id)
	}
	if c.StorePaths {
		if len(s.Regimes) != c.Steps+1 || s.Regimes[0] != c.InitialRegime || s.Regimes[c.Steps] != s.FinalRegime {
			return fmt.Errorf("backend returned malformed regime path for scenario %d", id)
		}
		for _, state := range s.Regimes {
			if state < 0 || state >= states {
				return fmt.Errorf("backend returned invalid regime path for scenario %d", id)
			}
		}
	} else if len(s.Prices) != 0 || len(s.Regimes) != 0 {
		return fmt.Errorf("backend returned unrequested paths for scenario %d", id)
	}
	return nil
}
