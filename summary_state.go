package quantaspindle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"unicode/utf8"
)

// ScenarioRange identifies the half-open absolute ID interval
// [StartID, StartID+Count). State provenance contains accepted scenarios only.
type ScenarioRange struct {
	StartID uint64
	Count   uint64
}

// RunSummaryState retains compensated pre-mean sums for range aggregation.
// Use RunRangeWithState or NewRunSummaryState to construct it, Merge to combine
// disjoint accepted ranges, and Summary to finalize independently owned means.
// MarshalBinary/UnmarshalBinary provide versioned transport between workers.
// Storage is O(assets + disjoint accepted ranges); no scenarios are retained.
// Do not copy a state or access it concurrently without synchronization.
type RunSummaryState struct {
	scope       string
	backend     string
	fingerprint [32]byte
	paths       uint64
	assets      int
	count       uint64
	means       []runMean
	ranges      []ScenarioRange
}

// NewRunSummaryState creates an empty merge identity for a model and backend.
// Scope must be a nonempty caller-supplied identity for the configuration and
// replay domain (including the toolchain). The configuration fingerprint also
// checks ordered assets and model fields, excluding Workers and BatchSize.
// Scope/backend identifiers and asset/regime names must be valid UTF-8.
// The backend name must match the actual sessions producing the other states;
// cross-backend merging is rejected. Model validation uses a one-scenario range.
func NewRunSummaryState(c Config, backend, scope string) (*RunSummaryState, error) {
	if err := c.ValidateRange(0, 1); err != nil {
		return nil, err
	}
	return newRunSummaryState(c, backend, scope)
}

func newRunSummaryState(c Config, backend, scope string) (*RunSummaryState, error) {
	if !validSummaryIdentity(scope) || !validSummaryIdentity(backend) {
		return nil, fmt.Errorf("summary state requires nonempty UTF-8 scope and backend names of at most 4096 bytes")
	}
	for _, asset := range c.Assets {
		if !utf8.ValidString(asset.Name) {
			return nil, fmt.Errorf("summary state asset names must be valid UTF-8")
		}
	}
	for _, regime := range c.Regimes {
		if !utf8.ValidString(regime.Name) {
			return nil, fmt.Errorf("summary state regime names must be valid UTF-8")
		}
	}
	model := c
	model.Workers, model.BatchSize = 0, 0
	encoded, err := json.Marshal(model)
	if err != nil {
		return nil, fmt.Errorf("summary state configuration: %w", err)
	}
	s := &RunSummaryState{scope: scope, backend: backend, fingerprint: sha256.Sum256(encoded), paths: uint64(c.Paths), assets: len(c.Assets)}
	s.means = make([]runMean, 3*s.assets)
	_, exponent := math.Frexp(float64(s.paths))
	for i := range s.means {
		s.means[i].exponent = exponent + 1
	}
	return s, nil
}

// RunRangeWithState has RunRange's ordered sink and delivered-prefix contract
// and additionally returns its opaque mergeable state, even on partial failure.
// Metrics are copied before sink ownership transfer and added only after the
// sink accepts a scenario. The range must be nonempty, as for RunRange; use
// NewRunSummaryState for an empty shard. Invalid dimensions, backend or scope
// may return a nil state. This API does not reconstruct sums from rounded means.
func RunRangeWithState(ctx context.Context, c Config, backend Backend, startID, count uint64, scope string, sink func(Scenario) error) (RunSummary, *RunSummaryState, error) {
	if err := c.validateRangeDimensions(startID, count); err != nil {
		return RunSummary{}, nil, err
	}
	if nilBackend(backend) {
		return RunSummary{}, nil, fmt.Errorf("backend is nil")
	}
	state, err := newRunSummaryState(c, backend.Name(), scope)
	if err != nil {
		return RunSummary{}, nil, err
	}
	summary, err := runRange(ctx, c, backend, startID, count, true, sink, state)
	return summary, state, err
}

// Count returns the accepted scenario count (zero for a nil receiver).
func (s *RunSummaryState) Count() uint64 {
	if s == nil {
		return 0
	}
	return s.count
}

// Ranges returns a copy of the sorted, disjoint accepted ID intervals.
// Adjacent intervals coalesce during merging. Empty states have no intervals.
func (s *RunSummaryState) Ranges() []ScenarioRange {
	if s == nil {
		return nil
	}
	return append([]ScenarioRange(nil), s.ranges...)
}

func (s *RunSummaryState) validate() error {
	if s == nil || s.assets < 1 || s.assets > 256 || len(s.means) != 3*s.assets || s.paths < 1 || s.paths > 1_000_000_000 || s.count > s.paths || !validSummaryIdentity(s.scope) || !validSummaryIdentity(s.backend) || s.fingerprint == [32]byte{} {
		return fmt.Errorf("summary state has invalid identity, count or metric dimensions")
	}
	_, exponent := math.Frexp(float64(s.paths))
	for _, m := range s.means {
		if m.exponent != exponent+1 || !finite(m.sum) || !finite(m.correction) || !finite(m.residual) || !finite(m.residualCorrection) || !finite(m.maxMagnitude) || m.maxMagnitude < 0 || !finite(m.sum+m.correction) || !finite(m.residual+m.residualCorrection) {
			return fmt.Errorf("summary state has invalid compensated metrics")
		}
		if s.count == 0 && (m.sum != 0 || m.correction != 0 || m.residual != 0 || m.residualCorrection != 0 || m.maxMagnitude != 0) {
			return fmt.Errorf("empty summary state has nonzero metrics")
		}
		if m.maxMagnitude == 0 && (m.sum != 0 || m.correction != 0 || m.residual != 0 || m.residualCorrection != 0) {
			return fmt.Errorf("summary state has nonzero metrics with zero observed magnitude")
		}
		if s.count > 0 && !finite(m.mean(s.count)) {
			return fmt.Errorf("summary state has nonfinite finalized metrics")
		}
	}
	var total, end uint64
	for i, r := range s.ranges {
		if r.Count == 0 || r.StartID >= s.paths || r.Count > s.paths-r.StartID || (i > 0 && r.StartID <= end) || r.Count > s.count-total {
			return fmt.Errorf("summary state has invalid or overlapping accepted ranges")
		}
		total += r.Count
		end = r.StartID + r.Count
	}
	if total != s.count {
		return fmt.Errorf("summary state count disagrees with accepted ranges")
	}
	return nil
}

func validSummaryIdentity(value string) bool {
	return len(value) > 0 && len(value) <= 4096 && utf8.ValidString(value)
}

// Summary finalizes this state's means without mutating it. Empty states return
// zero scenarios and dimensioned zero means. Compensated merge order can change
// last-bit rounding; there is no bitwise cross-order or cross-backend guarantee.
func (s *RunSummaryState) Summary() (RunSummary, error) {
	if err := s.validate(); err != nil {
		return RunSummary{}, err
	}
	r := RunSummary{Scenarios: s.count, Backend: s.backend, MeanReturns: make([]float64, s.assets), MeanRealizedVolatility: make([]float64, s.assets), MeanMaxDrawdown: make([]float64, s.assets)}
	if s.count > 0 {
		for i := 0; i < s.assets; i++ {
			r.MeanReturns[i] = s.means[i].mean(s.count)
			r.MeanRealizedVolatility[i] = s.means[s.assets+i].mean(s.count)
			r.MeanMaxDrawdown[i] = s.means[2*s.assets+i].mean(s.count)
		}
	}
	return r, nil
}

// Merge combines compatible states with disjoint accepted IDs. Compatibility
// requires identical scope, backend, model fingerprint and metric dimensions.
// Configuration worker/batch settings may differ. Rejected merges are atomic;
// source state and its slices remain unchanged. Retrying an already accepted
// range, including self-merge of a nonempty state, is an overlap error.
func (s *RunSummaryState) Merge(other *RunSummaryState) error {
	if err := s.validate(); err != nil {
		return err
	}
	if err := other.validate(); err != nil {
		return err
	}
	if s.scope != other.scope || s.backend != other.backend || s.fingerprint != other.fingerprint || s.assets != other.assets || s.paths != other.paths {
		return fmt.Errorf("summary states have incompatible scope, backend, model or ordered assets")
	}
	if other.count > s.paths-s.count {
		return fmt.Errorf("summary state count exceeds the configured ID space")
	}
	ranges, err := mergeSummaryRanges(s.ranges, other.ranges)
	if err != nil {
		return err
	}
	means := append([]runMean(nil), s.means...)
	for i, m := range other.means {
		dst := &means[i]
		dst.sum, dst.correction = riskAdd(dst.sum, dst.correction, m.sum)
		dst.sum, dst.correction = riskAdd(dst.sum, dst.correction, m.correction)
		dst.residual, dst.residualCorrection = riskAdd(dst.residual, dst.residualCorrection, m.residual)
		dst.residual, dst.residualCorrection = riskAdd(dst.residual, dst.residualCorrection, m.residualCorrection)
		dst.maxMagnitude = math.Max(dst.maxMagnitude, m.maxMagnitude)
	}
	candidate := *s
	candidate.count += other.count
	candidate.means, candidate.ranges = means, ranges
	if err := candidate.validate(); err != nil {
		return err
	}
	*s = candidate
	return nil
}

func mergeSummaryRanges(a, b []ScenarioRange) ([]ScenarioRange, error) {
	out := make([]ScenarioRange, 0, len(a)+len(b))
	for len(a) > 0 || len(b) > 0 {
		var r ScenarioRange
		if len(b) == 0 || (len(a) > 0 && a[0].StartID < b[0].StartID) {
			r, a = a[0], a[1:]
		} else {
			r, b = b[0], b[1:]
		}
		if len(out) > 0 {
			last := &out[len(out)-1]
			end := last.StartID + last.Count
			if r.StartID < end {
				return nil, fmt.Errorf("summary states contain overlapping accepted ranges")
			}
			if r.StartID == end {
				last.Count += r.Count
				continue
			}
		}
		out = append(out, r)
	}
	return out, nil
}

// The wire schema is versioned and deliberately separate from private state.
// Treat encoded state as opaque trusted-worker data, not an attestation that
// scenarios were actually computed. Neither a fingerprint nor scope authenticates it.
type summaryStateWire struct {
	Version     int
	Scope       string
	Backend     string
	Fingerprint [32]byte
	Paths       uint64
	Assets      int
	Count       uint64
	Means       []summaryMeanWire
	Ranges      []ScenarioRange
}

type summaryMeanWire struct {
	Sum, Correction, Residual, ResidualCorrection, MaxMagnitude float64
	Exponent                                                    int
}

const maxSummaryStateBytes = 64 << 20

// MarshalBinary encodes a validated state for transport, including all
// compensated components and accepted range provenance. Version 1 uses JSON
// internally; callers should treat the bytes as opaque. Payloads above 64 MiB
// are rejected. Rounded RunSummary values are not an interchangeable encoding.
func (s *RunSummaryState) MarshalBinary() ([]byte, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	w := summaryStateWire{Version: 1, Scope: s.scope, Backend: s.backend, Fingerprint: s.fingerprint, Paths: s.paths, Assets: s.assets, Count: s.count, Ranges: s.ranges, Means: make([]summaryMeanWire, len(s.means))}
	for i, m := range s.means {
		w.Means[i] = summaryMeanWire{m.sum, m.correction, m.residual, m.residualCorrection, m.maxMagnitude, m.exponent}
	}
	data, err := json.Marshal(w)
	if len(data) > maxSummaryStateBytes {
		return nil, fmt.Errorf("summary state transport exceeds 64 MiB")
	}
	return data, err
}

// UnmarshalBinary validates and atomically replaces a state from supported
// canonical encoded bytes. Unknown versions/fields, malformed dimensions or components,
// inconsistent counts/ranges and payloads above 64 MiB are rejected. This does
// not authenticate the producer or prove its declared configuration was used.
func (s *RunSummaryState) UnmarshalBinary(data []byte) error {
	if s == nil || len(data) > maxSummaryStateBytes || !utf8.Valid(data) {
		return fmt.Errorf("summary state receiver is nil, transport exceeds 64 MiB or bytes are not valid UTF-8")
	}
	var w summaryStateWire
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&w); err != nil {
		return fmt.Errorf("summary state transport: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("summary state transport has trailing data")
	}
	if w.Version != 1 || w.Assets < 1 || w.Assets > 256 || len(w.Means) != 3*w.Assets {
		return fmt.Errorf("unsupported summary state version or malformed metric dimensions")
	}
	canonical, err := json.Marshal(w)
	if err != nil || !bytes.Equal(data, canonical) {
		return fmt.Errorf("summary state transport is not canonical version 1 encoding")
	}
	candidate := RunSummaryState{scope: w.Scope, backend: w.Backend, fingerprint: w.Fingerprint, paths: w.Paths, assets: w.Assets, count: w.Count, ranges: w.Ranges, means: make([]runMean, len(w.Means))}
	for i, m := range w.Means {
		candidate.means[i] = runMean{sum: m.Sum, correction: m.Correction, residual: m.Residual, residualCorrection: m.ResidualCorrection, maxMagnitude: m.MaxMagnitude, exponent: m.Exponent}
	}
	if err := candidate.validate(); err != nil {
		return err
	}
	*s = candidate
	return nil
}
