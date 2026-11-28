# Mergeable range summaries

`RunRangeWithState` is an opt-in Go API for aggregating independent scenario ranges without retaining the scenarios. It delivers through the existing ordered sink and returns `(RunSummary, *RunSummaryState, error)`. Existing `Run`, `RunRange` and `RunSummary` APIs keep their contracts. Their ordinary execution path performs no state serialization, configuration hashing or provenance allocation.

```go
scope := "portfolio-job-42/cpu/go1.27.1/model-v3"
summary, shard, runErr := quantaspindle.RunRangeWithState(
    ctx, config, quantaspindle.CPU{}, startID, count, scope, sink,
)
// Inspect runErr. A non-nil shard still represents its accepted prefix.
payload, err := shard.MarshalBinary()
// Transport payload to the coordinator using the application's own channel.
var received quantaspindle.RunSummaryState
err = received.UnmarshalBinary(payload)
err = overall.Merge(&received)
combined, err := overall.Summary()
```

Initialize `overall` with `NewRunSummaryState(config, backendName, scope)`. The backend name is the actual producing session's `Name()`; for CPU, use `(quantaspindle.CPU{}).Name()`. An empty state is a merge identity with zero count, no ID intervals, and dimensioned zero means. Like `RunRange`, the run API requires a nonempty valid requested range. Pre-cancellation, startup failures and sinks that reject the first scenario can return empty states. Invalid dimensions, backend or scope may return nil; check the error and state before use. Session-close failure still returns all successfully accepted scenarios.

Each accepted scenario contributes its return, realized-volatility and drawdown metrics before those sums are rounded into means. Metrics are copied before the sink takes ownership of scenario slices and added only after the sink returns nil. Sink rejection excludes that scenario; cancellation after a successful sink call includes it. Backend batch failures retain the same delivered-prefix semantics as `RunRange`. The state never reconstructs sums by multiplying rounded per-shard means by counts.

Compatibility requires the same explicit scope, actual backend name, asset dimensions and configuration fingerprint. Scope identifies a single configuration and replay domain and must include any relevant external model/backend versions and toolchain identity. The fingerprint covers the full `Config` representation, including ordered asset names/parameters, seed, total `Paths` ID space, regimes, transition/correlation representation and path-storage choice. **Workers and BatchSize are excluded:** deterministic CPU shards may vary their scheduling and batching. Semantically equivalent but differently represented configurations (such as nil versus empty slices) can still have different fingerprints. Scope and backend names must be nonempty valid UTF-8 of at most 4096 bytes; asset and regime names must also be valid UTF-8, so distinct byte strings cannot collapse through JSON replacement.

States from different backends cannot merge. A backend name is a declared implementation identity, not proof of an implementation or replay behavior. Custom backends must continue honoring their contract. No CPU/CUDA replay equivalence is implied. CPU scenario replay remains scoped to the implementation, backend and toolchain. Floating compensated sums can differ in their last bits under different merge trees; merges guarantee compatible accounting, not bitwise equality across orders. The tests declare a tolerance of eight adjacent-float steps for normal means against an exact rational direct scan, with exact rounding required for tested subnormal/zero cases.

Each state uses compensated sums in common power-of-two units based on the configuration's total `Paths`, independent of shard size. This leaves room for all disjoint accepted IDs and avoids overflowing raw totals. Subnormal scaling residuals and their compensation are preserved separately. Merge adds both primary and correction components, including the subnormal residual components, before division by the combined count. Final means remain bounded by observed magnitudes. The same state can be finalized repeatedly, and the returned summary slices are independently owned.

`Ranges()` returns a copy of the actual accepted half-open absolute ID intervals, not the originally requested intervals. Merging sorts these intervals and coalesces adjacent ones; any accepted-ID overlap is rejected without mutation. Retrying a previously accepted shard is therefore an error, including self-merge of a nonempty state. A failed shard's unaccepted suffix may be retried and merged. A coordinator must still ensure it received every required range and use one scope for exactly one intended ID domain: missing ranges are not inferred as successful. Source state remains unchanged by merging.

Persistent storage is O(asset count + disjoint accepted intervals), plus the fixed configuration fingerprint and bounded scope/backend strings. A single shard retains one interval regardless of scenario count. Adjacent completed shards coalesce to one interval; separated shards retain one interval per gap until gaps fill. Merging takes O(asset count + interval count) time and temporary storage. No `Scenario` records or observation slices are retained. `Summary` allocates only its three per-asset output slices.

`MarshalBinary` and `UnmarshalBinary` are the supported distributed transport. They preserve every compensated component, counts, compatibility metadata and provenance. Version 1 uses canonical JSON internally; treat the result as opaque bytes, not a schema to edit or reconstruct. Decoding rejects unknown versions/fields, noncanonical or invalid UTF-8 encodings, nonfinite components, mismatched metric dimensions/scales and inconsistent/overlapping ranges, and leaves the destination unchanged on failure. Encoded payloads above 64 MiB are rejected. The encoded state and fingerprint do not authenticate a worker or prove that its declared configuration was used; use the application's trusted transport/integrity mechanism. `RunSummary` alone, ordinary JSON marshaling of the opaque Go struct, or manually reconstructed rounded means are not substitutes for this transport.

States require external synchronization for concurrent access and must not be copied after construction. Use their supported transport when an independent copy is needed. The feature does not dispatch jobs, write checkpoints or add CLI output fields; the application owns worker coordination, persistence, retry policy and sink side effects.
