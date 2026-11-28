package quantaspindle

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
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

// RangeRunBackend optionally prepares a session for a selected RunRange.
// Config.Paths retains the total ID space; startID and count describe the work
// this session must validate and serve. RunRange prefers this over BeginRun.
type RangeRunBackend interface {
	BeginRunRange(context.Context, Config, uint64, uint64) (BackendSession, error)
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
	if err := validateCPUBatch(ctx, c, first, count); err != nil {
		return nil, err
	}
	l, err := correlationFactor(c.Correlation, len(c.Assets))
	if err != nil {
		return nil, err
	}
	return cpuBatch(ctx, c, l, first, count)
}

// cpuBatch uses the immutable factor prepared for a Run, Batch, or PreparedCPU.
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
	moments := make([]scaledMoments, n)
	eps := make([]float64, n)
	for i, a := range c.Assets {
		prices[i] = a.InitialPrice
		peaks[i] = a.InitialPrice
		moments[i].inverse = 1
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
			// Preserve ordinary arithmetic, but form integrated variance from
			// scaled sigma when squaring sigma loses its range.
			if !finite(lr) || (sigma != 0 && sigma*sigma < 0x1p-1022) {
				vol := sigma * sqrtDT
				lr = mu*c.DT - (0.5*vol)*vol + vol*z
			}
			if !finite(lr) {
				return Scenario{}, fmt.Errorf("scenario %d step %d asset %d numerical overflow/underflow", id, step, i)
			}
			factor := math.Exp(lr)
			p := prices[i] * factor
			// Tiny log returns need Expm1 and FMA to retain the increment
			// before rounding the price. Also recover exponential range and
			// avoid extra rounding near the final price's range limits.
			if math.Abs(lr) < 0x1p-26 || factor < 0x1p-1022 || !finite(factor) || p <= 0x1p-1022 || p >= 0x1p1023 {
				p = priceAfterLogReturn(prices[i], lr)
			}
			if !finite(p) || p <= 0 {
				return Scenario{}, fmt.Errorf("scenario %d step %d asset %d numerical overflow/underflow", id, step, i)
			}
			prices[i] = p
			moments[i].add(lr, step+1)
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
		s.Returns[i] = (prices[i] - a.InitialPrice) / a.InitialPrice
		if c.Steps > 1 {
			s.RealizedVolatility[i] = moments[i].annualizedStdDev(c.Steps, sqrtDT)
		}
		if !finite(s.Returns[i]) || !finite(s.RealizedVolatility[i]) {
			return Scenario{}, fmt.Errorf("scenario %d asset %d nonfinite summary", id, i)
		}
	}
	return s, nil
}

// priceAfterLogReturn multiplies a positive finite price by exp(lr), for finite
// lr, by keeping the price and exponential powers of two separate.
func priceAfterLogReturn(price, lr float64) float64 {
	// Beyond these bounds even the smallest/largest price cannot recover.
	// Bound the conversion to an integer exponent as well.
	if lr > 2048 {
		return math.Inf(1)
	}
	if lr < -2048 {
		return 0
	}
	const (
		ln2Hi = 6.93147180369123816490e-01
		ln2Lo = 1.90821492927058770002e-10
	)
	fraction, exponent := math.Frexp(price)
	power := math.Round(lr * math.Log2E)
	remainder := (lr - power*ln2Hi) - power*ln2Lo
	// Expm1 and FMA retain small changes near the overflow boundary.
	change := math.Expm1(remainder)
	scaled := math.FMA(fraction, change, fraction)
	exponent += int(power)
	price = math.Ldexp(scaled, exponent)
	if price <= 0x1p-1022 {
		// A rounded mantissa may land exactly halfway between subnormal
		// outputs. Only at such a midpoint, use its residual to recover
		// the rounding direction before the final scaling.
		units := math.Ldexp(scaled, exponent+1074)
		if units-math.Floor(units) == 0.5 {
			residual := math.FMA(fraction, change, fraction-scaled)
			if residual != 0 {
				scaled = math.Nextafter(scaled, math.Copysign(math.Inf(1), residual))
				price = math.Ldexp(scaled, exponent)
			}
		}
	}
	return price
}

// scaledMoments maintains Welford moments in power-of-two normalized units.
// Squared deviations remain representable even for subnormal log returns.
// One accumulator per asset keeps memory independent of the number of steps.
type scaledMoments struct {
	mean, m2, inverse float64
	exponent          int
}

func (m *scaledMoments) add(x float64, count int) {
	// Cache the power-of-two multiplier to keep rescaling off the hot path.
	scaled := x * m.inverse
	if m.exponent < -1023 {
		// The inverse scale of tiny returns cannot itself fit in float64.
		scaled = math.Ldexp(x, -m.exponent)
	}
	magnitude := math.Abs(scaled)
	// Normalize the first nonzero tiny value before squaring deviations.
	if magnitude > 1 || (x != 0 && magnitude < 0x1p-128 && m.mean == 0 && m.m2 == 0) {
		_, exponent := math.Frexp(math.Abs(x))
		shift := m.exponent - exponent
		m.mean = math.Ldexp(m.mean, shift)
		m.m2 = math.Ldexp(m.m2, 2*shift)
		m.exponent = exponent
		m.inverse = math.Ldexp(1, -exponent)
		scaled = math.Ldexp(x, -exponent)
	}
	delta := scaled - m.mean
	m.mean += delta / float64(count)
	m.m2 += delta * (scaled - m.mean)
}

func (m scaledMoments) annualizedStdDev(count int, sqrtDT float64) float64 {
	fraction, exponent := math.Frexp(sqrtDT)
	// Divide and rescale together so annualization cannot lose a tiny
	// standard deviation before division by a tiny time step.
	return math.Ldexp(math.Sqrt(math.Max(m.m2, 0)/float64(count-1))/fraction, m.exponent-exponent)
}

func nilBackend(backend Backend) bool {
	if backend == nil {
		return true
	}
	value := reflect.ValueOf(backend)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	}
	return false
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

// runMean accumulates a compensated sum in power-of-two scaled units.
// Scaling by the selected scenario count bounds every partial sum, even when
// unscaled observations would overflow. Residuals retain bits lost by scaling
// subnormal values. Each accumulator uses constant memory throughout a run.
type runMean struct {
	sum, correction, residual, residualCorrection float64
	maxMagnitude                                  float64
	exponent                                      int
}

func (m *runMean) add(value float64) {
	scaled := math.Ldexp(value, -m.exponent)
	m.sum, m.correction = riskAdd(m.sum, m.correction, scaled)
	if math.Abs(scaled) < 0x1p-1022 {
		remainder := value - math.Ldexp(scaled, m.exponent)
		m.residual, m.residualCorrection = riskAdd(m.residual, m.residualCorrection, remainder)
	}
	m.maxMagnitude = math.Max(m.maxMagnitude, math.Abs(value))
}

func (m runMean) mean(count uint64) float64 {
	total := m.sum + m.correction
	n := float64(count)
	var mean float64
	if restored := math.Ldexp(total, m.exponent); finite(restored) {
		restored, correction := riskAdd(restored, m.residualCorrection, m.residual)
		mean = (restored + correction) / n
	} else {
		fraction, exponent := math.Frexp(total)
		mean = math.Ldexp(fraction/n, exponent+m.exponent)
	}
	// A finite sample's mean cannot exceed its largest magnitude value.
	return math.Max(-m.maxMagnitude, math.Min(m.maxMagnitude, mean))
}

// Run generates bounded waves of concurrent batches and delivers scenarios to
// sink serially in ascending ID order. A nil sink discards individual results.
// On error it returns only the successfully delivered prefix summary. The input
// Config and its slices must remain immutable until Run returns.
func Run(ctx context.Context, c Config, backend Backend, sink func(Scenario) error) (summary RunSummary, runErr error) {
	return runRange(ctx, c, backend, 0, uint64(c.Paths), false, sink, nil)
}

// RunRange delivers count scenarios starting at startID, retaining their IDs
// and seeds in the total Config.Paths ID space. Its summary covers only the
// successfully delivered range prefix. Working storage is checked against the
// selected count. Backends receive the original Config and absolute batch IDs;
// custom backends may impose additional validation or resource requirements.
func RunRange(ctx context.Context, c Config, backend Backend, startID, count uint64, sink func(Scenario) error) (RunSummary, error) {
	return runRange(ctx, c, backend, startID, count, true, sink, nil)
}

func runRange(ctx context.Context, c Config, backend Backend, startID, count uint64, selected bool, sink func(Scenario) error, state *RunSummaryState) (summary RunSummary, runErr error) {
	if ctx == nil {
		return summary, fmt.Errorf("context is nil")
	}
	if err := c.validateDimensionsForRange(startID, count, selected); err != nil {
		return summary, err
	}
	if nilBackend(backend) {
		return summary, fmt.Errorf("backend is nil")
	}
	factor, err := correlationFactor(c.Correlation, len(c.Assets))
	if err != nil {
		return summary, err
	}
	if err := ctx.Err(); err != nil {
		return summary, err
	}
	var begin func() (BackendSession, error)
	if scoped, ok := backend.(RangeRunBackend); selected && ok {
		begin = func() (BackendSession, error) { return scoped.BeginRunRange(ctx, c, startID, count) }
	} else if scoped, ok := backend.(RunBackend); ok {
		begin = func() (BackendSession, error) { return scoped.BeginRun(ctx, c) }
	}
	if begin != nil {
		session, err := begin()
		if err != nil {
			return summary, err
		}
		if nilBackend(session) {
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
	var means []runMean
	if state == nil {
		means = make([]runMean, len(metrics))
		_, exponent := math.Frexp(float64(count))
		for i := range means {
			// One extra bit leaves headroom for accumulation rounding.
			means[i].exponent = exponent + 1
		}
	} else {
		// All shards share the total ID-space scale, rather than their
		// individual counts, so compensated components merge without rescaling.
		if !validSummaryIdentity(summary.Backend) {
			return summary, fmt.Errorf("summary state session backend name must have 1..4096 valid UTF-8 bytes")
		}
		state.backend = summary.Backend
		means = state.means
	}
	defer func() {
		if summary.Scenarios == 0 {
			return
		}
		n := len(c.Assets)
		for i := range summary.MeanReturns {
			summary.MeanReturns[i] = means[i].mean(summary.Scenarios)
			summary.MeanRealizedVolatility[i] = means[n+i].mean(summary.Scenarios)
			summary.MeanMaxDrawdown[i] = means[2*n+i].mean(summary.Scenarios)
		}
	}()
	for offset := 0; offset < int(count); {
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		nb := min(c.Workers, (int(count)-offset+c.BatchSize-1)/c.BatchSize)
		results := make([][]Scenario, nb)
		errs := make([]error, nb)
		var wg sync.WaitGroup
		waveCtx, cancel := context.WithCancel(ctx)
		for b := 0; b < nb; b++ {
			start := startID + uint64(offset+b*c.BatchSize)
			batchCount := min(c.BatchSize, int(count)-offset-b*c.BatchSize)
			wg.Add(1)
			go func(b int, start uint64, count int) {
				defer wg.Done()
				if cpuValue || cpuPointer {
					results[b], errs[b] = cpuBatch(waveCtx, c, factor, start, count)
				} else {
					results[b], errs[b] = backend.Batch(waveCtx, c, start, count)
				}
				if errs[b] != nil {
					cancel()
				}
			}(b, start, batchCount)
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
			expected := min(c.BatchSize, int(count)-offset-b*c.BatchSize)
			if len(batch) != expected {
				return summary, fmt.Errorf("backend returned %d scenarios, expected %d", len(batch), expected)
			}
			for j, s := range batch {
				if err := ctx.Err(); err != nil {
					return summary, err
				}
				if err := validateResult(s, startID+uint64(offset+b*c.BatchSize+j), c); err != nil {
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
				for i, value := range metrics {
					means[i].add(value)
				}
				if state != nil {
					state.count = summary.Scenarios
					if len(state.ranges) == 0 {
						state.ranges = append(state.ranges, ScenarioRange{StartID: startID})
					}
					state.ranges[0].Count = summary.Scenarios
				}
			}
		}
		offset += nb * c.BatchSize
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
