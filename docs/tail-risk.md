# Streamed tail-risk intervals

`LossHistogram` is an opt-in Go API for approximate empirical tail risk when retaining every loss is impractical. `RiskFromLosses` and `RiskFromReturns` keep their existing exact, caller-retained sample behavior. The histogram does not modify `RunSummary` or add CLI fields. Positive loss means a loss and negative loss means a gain, using the same units as the supplied observations.

```go
histogram, err := quantaspindle.NewLossHistogram(-2, 3, 1000)
if err != nil {
    return err
}
// Inside a Run sink, compute a portfolio loss, then:
if err := histogram.Add(loss); err != nil {
    return err
}
// After consuming the stream (or its successfully accepted prefix):
risk, err := histogram.Risk(.95)
```

Choose finite bounds before consuming the stream. Both endpoints are accepted. An observation outside those bounds, NaN or infinity returns an error without changing the histogram. There is no automatic clamping, range expansion, backend selection or loss conversion. If a sink calls `Add` and subsequently returns another error, that observation has already been accepted by the histogram; arrange any other fallible sink work before `Add` when its count must match the run's delivered prefix.

The constructor accepts 1 through 1,000,000 bins. Interior edge `i` is the exact real value `lower + (upper-lower)*i/bins`, rounded once to nearest float64, ties to even. Exact rational arithmetic avoids overflowing the intermediate span or product. The represented edges must be strictly increasing, and each represented bin's width, rounded upward, must be finite. Configurations with collapsed edges or unrepresentable bin widths return errors. For example, `[-MaxFloat64, MaxFloat64]` with two bins is valid, but one bin is rejected. Adjacent floats cannot be subdivided into two bins. `Edges()` returns an independent copy so callers can inspect the actual partition.

Interior bins are half-open `[edge[i], edge[i+1])`; an exact interior boundary belongs to the bin on its right. The last bin includes its upper endpoint. Signed zero has one canonical edge representation. A search over represented edges assigns observations, avoiding ambiguous division near boundaries. Rounded widths need not be identical; `HistogramRiskSummary.BinWidth` is the upward-rounded largest represented width, which may differ from the nominal mathematical span divided by the bin count.

`Risk(confidence)` requires observations and a finite confidence strictly between zero and one. It uses nearest rank `ceil(confidence*N)` for VaR and averages the largest `max(1, ceil((1-confidence)*N))` observations for expected shortfall. As in `RiskFromLosses`, products within relative tolerance `1e-12` of an integer round to that integer before taking the ceiling. Rank products use float64 arithmetic, including conversion of the count; counts above `2^53` therefore have float64 rank granularity. Counts and selected tail counts themselves remain uint64 integers. Products at a rounded count endpoint are bounded by the actual count before integer conversion.

The result contains closed loss intervals, rather than an unlabeled point estimate. VaR lies between the two edges of its rank's bin. Expected shortfall is bounded by replacing the selected largest losses with their bins' lower or upper edges, including only the necessary number from a partially selected tail bin. Weighted endpoint sums and division are evaluated exactly with rational arithmetic. The interval encloses the mathematical empirical expected shortfall, including signed cancellation and values near the finite range limits. It does not use or approximate standard deviation, so it can return finite tail bounds even when the exact API rejects an unrepresentable sample standard deviation.

VaR's interval width is at most `BinWidth`. Expected-shortfall endpoints round outward to float64 and expand by one further representable step when that step is finite. The result reports `RoundingAllowance`, the upward-rounded sum of those endpoints' actual outward displacements from the exact weighted endpoint means. Its interval width is at most `BinWidth + RoundingAllowance`. Interpret width and this inequality in real arithmetic when subtracting endpoints or adding the bound would overflow float64. All returned endpoints and the allowance are finite. No extra statistical confidence guarantee is implied: this is a deterministic discretization bound for the accepted empirical sample, not uncertainty about a population or forecast.

Only bin counts, edges and a total count are retained: storage is O(number of bins), independent of observation count. `Add` takes O(log bins) time and allocates no memory. Querying risk takes O(bins) time plus bounded-size rational arithmetic; the number of bits is bounded by the float64 range and uint64 count range. It does not materialize observations. Exact empirical tail risk still requires retained observations or another exact storage/sorting strategy.

For independently simulated scenario ranges, construct one histogram with the same bounds and bin count per range and call `destination.Merge(source)`. Only identical represented grids can merge. Integer count addition makes all merge orders deterministic, and merging leaves the source unchanged. Self-merge doubles its counts. Total counts up to `MaxUint64` are supported; `Add` and `Merge` reject overflow atomically. Merge does not detect overlapping scenario IDs: the caller must ensure ranges represent the intended sample exactly once. Histograms require external synchronization for concurrent access and must not be copied after construction.

The tests enumerate all 729 six-loss vectors over `{-2,0,3}` at nine confidence levels on four different grids, comparing against both `RiskFromLosses` and independent rational means. They also cover endpoints and ties, extreme and subnormal values, rejected grids and inputs, merge orders, count overflow, and a million-loss stream with unchanged bin storage and zero allocations during additions.
