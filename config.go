package quantaspindle

import (
	"fmt"
	"math"
	"math/big"
	"runtime"
)

// Asset supplies the initial price and baseline GBM parameters of one asset.
type Asset struct {
	Name         string
	InitialPrice float64
	Mu           float64
	Sigma        float64
}

// Regime supplies annual drift and volatility vectors in Assets order.
type Regime struct {
	Name  string
	Mu    []float64
	Sigma []float64
}

// Config defines a simulation. Empty Regimes uses Asset parameters; empty
// Correlation uses independent innovations. Empty Transition is allowed only
// without regimes. StorePaths stores asset-major prices and interval states.
// Resource limits are 256 assets, four regimes, 1,000,000 steps, 1,000,000,000
// scenarios, 1,024 workers, 16,384 scenarios per batch, and an estimated 512 MiB
// per Run wave including the shared CPU correlation factor. Caller retention,
// input data, custom backend allocations and runtime overhead are excluded.
type Config struct {
	Paths         int
	Steps         int
	DT            float64
	Seed          uint64
	Workers       int
	BatchSize     int
	Assets        []Asset
	Regimes       []Regime
	Transition    [][]float64
	InitialRegime int
	Correlation   [][]float64
	StorePaths    bool
}

// DefaultConfig returns an explicit singleasset, one-year baseline. The caller
// should set drift and volatility appropriate to their modeling assumptions.
func DefaultConfig() Config {
	w := runtime.GOMAXPROCS(0)
	if w > 1024 {
		w = 1024
	}
	return Config{Paths: 1000, Steps: 252, DT: 1.0 / 252, Seed: 1, Workers: w, BatchSize: 64,
		Assets: []Asset{{Name: "asset", InitialPrice: 100, Mu: 0.05, Sigma: 0.2}}}
}

// DefaultRegimeConfig returns illustrative bull, bear, sideways and stress
// regimes for the default asset. Parameters and per-step transition probabilities
// are uncalibrated examples, not estimates or investment advice. The initial
// regime is bull; callers can replace all parameters and transitions.
func DefaultRegimeConfig() Config {
	c := DefaultConfig()
	c.Regimes = []Regime{
		{Name: "bull", Mu: []float64{.12}, Sigma: []float64{.16}},
		{Name: "bear", Mu: []float64{-.08}, Sigma: []float64{.24}},
		{Name: "sideways", Mu: []float64{.01}, Sigma: []float64{.12}},
		{Name: "stress", Mu: []float64{-.30}, Sigma: []float64{.50}},
	}
	c.Transition = [][]float64{{.97, .01, .015, .005}, {.01, .965, .02, .005}, {.02, .02, .955, .005}, {.02, .04, .04, .9}}
	return c
}

const maxWaveBytes uint64 = 512 << 20

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// Validate checks dimensions, finite values, probabilities, positive semidefinite
// correlation and bounded working storage. No simulation workers are launched.
func (c Config) Validate() error {
	if err := c.validateDimensions(); err != nil {
		return err
	}
	_, err := correlationFactor(c.Correlation, len(c.Assets))
	return err
}

func (c Config) validateDimensions() error {
	if c.Paths < 1 || c.Paths > 1_000_000_000 {
		return fmt.Errorf("paths must be in [1,1000000000]")
	}
	if c.Steps < 1 || c.Steps > 1_000_000 {
		return fmt.Errorf("steps must be in [1,1000000]")
	}
	if !finite(c.DT) || c.DT <= 0 || !finite(c.DT*float64(c.Steps)) {
		return fmt.Errorf("DT and total horizon must be positive and finite")
	}
	if c.Workers < 1 || c.Workers > 1024 {
		return fmt.Errorf("workers must be in [1,1024]")
	}
	if c.BatchSize < 1 || c.BatchSize > 16384 {
		return fmt.Errorf("batch size must be in [1,16384]")
	}
	n := len(c.Assets)
	if n < 1 || n > 256 {
		return fmt.Errorf("asset count must be in [1,256]")
	}
	for i, a := range c.Assets {
		if !finite(a.InitialPrice) || a.InitialPrice <= 0 || !finite(a.Mu) || !finite(a.Sigma) || a.Sigma < 0 {
			return fmt.Errorf("asset %d has invalid price, drift or volatility", i)
		}
	}
	if len(c.Regimes) > 4 {
		return fmt.Errorf("at most four regimes are supported")
	}
	if len(c.Regimes) == 0 {
		if len(c.Transition) != 0 || c.InitialRegime != 0 {
			return fmt.Errorf("transition and initial regime require regimes")
		}
	} else {
		r := len(c.Regimes)
		if c.InitialRegime < 0 || c.InitialRegime >= r || len(c.Transition) != r {
			return fmt.Errorf("invalid initial regime or transition dimensions")
		}
		for i, g := range c.Regimes {
			if len(g.Mu) != n || len(g.Sigma) != n || len(c.Transition[i]) != r {
				return fmt.Errorf("regime %d dimensions mismatch", i)
			}
			for j := 0; j < n; j++ {
				if !finite(g.Mu[j]) || !finite(g.Sigma[j]) || g.Sigma[j] < 0 {
					return fmt.Errorf("regime %d asset %d parameters invalid", i, j)
				}
			}
			sum := 0.0
			for _, p := range c.Transition[i] {
				if !finite(p) || p < 0 || p > 1 {
					return fmt.Errorf("transition %d has invalid probability", i)
				}
				sum += p
			}
			if math.Abs(sum-1) > 1e-12 {
				return fmt.Errorf("transition row %d must sum to one (tolerance 1e-12)", i)
			}
		}
	}
	w := min(c.Workers, (c.Paths+c.BatchSize-1)/c.BatchSize)
	if uint64(w)*uint64(min(c.BatchSize, c.Paths))*scenarioBytes(c)+factorBytes(n) > maxWaveBytes {
		return fmt.Errorf("configured wave exceeds 512 MiB storage limit; reduce workers/batch size or disable paths")
	}
	return nil
}

func factorBytes(n int) uint64 { return uint64(n)*uint64(n)*8 + uint64(n)*24 + 24 }

func scenarioBytes(c Config) uint64 {
	// Conservative slice/header/state estimate; products are bounded by validation.
	b := uint64(256 + len(c.Assets)*160)
	if c.StorePaths {
		b += uint64(c.Steps+1) * uint64(len(c.Assets)*8+8)
	}
	return b
}

func correlationFactor(c [][]float64, n int) ([][]float64, error) {
	l := make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}
	if len(c) == 0 {
		for i := range l {
			l[i][i] = 1
		}
		return l, nil
	}
	if len(c) != n {
		return nil, fmt.Errorf("correlation dimensions mismatch")
	}
	for i := range c {
		if len(c[i]) != n {
			return nil, fmt.Errorf("correlation row %d dimensions mismatch", i)
		}
	}
	const tol = 1e-12
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := c[i][j]
			if !finite(v) || v < -1 || v > 1 || math.Abs(v-c[j][i]) > tol {
				return nil, fmt.Errorf("correlation must be finite, symmetric and in [-1,1]")
			}
		}
		if math.Abs(c[i][i]-1) > tol {
			return nil, fmt.Errorf("correlation diagonal must be one")
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := c[i][j]
			for k := 0; k < j; k++ {
				s -= l[i][k] * l[j][k]
			}
			if i == j {
				if s < -tol {
					return correlationFactorPrecise(c, l, tol)
				}
				l[i][j] = math.Sqrt(math.Max(s, 0))
			} else if l[j][j] > 0 {
				l[i][j] = s / l[j][j]
			} else if math.Abs(s) > tol {
				return correlationFactorPrecise(c, l, tol)
			}
		}
	}
	return l, nil
}

// correlationFactorPrecise retries a failed Cholesky decomposition with enough
// precision to avoid amplification of float64 rounding at nearly zero pivots.
// Ordinary matrices keep the allocation and arithmetic costs of the float64
// path. The retry remains O(n^3) time and O(n^2) space and uses the same PSD
// tolerance; it does not relax the acceptance threshold to hide cancellation.
func correlationFactorPrecise(c, l [][]float64, tol float64) ([][]float64, error) {
	n := len(c)
	const precision = 256
	factor := make([]big.Float, n*n)
	for i := range factor {
		factor[i].SetPrec(precision)
	}
	var s, product, magnitude, tolerance, negativeTolerance big.Float
	for _, v := range []*big.Float{&s, &product, &magnitude, &tolerance, &negativeTolerance} {
		v.SetPrec(precision)
	}
	tolerance.SetFloat64(tol)
	negativeTolerance.Neg(&tolerance)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s.SetFloat64(c[i][j])
			for k := 0; k < j; k++ {
				product.Mul(&factor[i*n+k], &factor[j*n+k])
				s.Sub(&s, &product)
			}
			entry := &factor[i*n+j]
			if i == j {
				if s.Cmp(&negativeTolerance) < 0 {
					return nil, fmt.Errorf("correlation is not positive semidefinite")
				}
				if s.Sign() < 0 {
					s.SetInt64(0)
				}
				entry.Sqrt(&s)
			} else if factor[j*n+j].Sign() > 0 {
				entry.Quo(&s, &factor[j*n+j])
			} else if magnitude.Abs(&s).Cmp(&tolerance) > 0 {
				return nil, fmt.Errorf("correlation is not positive semidefinite")
			}
			l[i][j], _ = entry.Float64()
		}
	}
	return l, nil
}
