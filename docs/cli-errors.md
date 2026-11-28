# CLI error records

The CLI returns a nonzero exit status on failure and prints diagnostics to stderr. By default, errors before run metadata leave stdout empty. To consume those failures as JSON, opt in from a current checkout:

```sh
go run ./cmd/quantaspindle -json-errors -paths 0
```

Stdout then contains exactly one newline-terminated record:

```json
{"error":"paths must be in [1,1000000000]","type":"error"}
```

This covers errors from flag parsing, loading or decoding a strict configuration file, validating CLI options or model configuration, and automatic backend selection. Stderr diagnostics are still emitted, and the process still fails. Successful simulations retain their metadata/scenario/summary sequence, and successful `-validate-only` checks retain their sole validation record. Help requests remain successful and emit no JSON error.

## Preliminary option scan

The CLI discovers this opt-in before ordinary flag parsing, allowing an unknown or malformed flag to precede it. The scan uses the registered flags to recognize values, with these rules:

- `-json-errors` and `--json-errors` enable the mode. An explicit boolean must use `=`, such as `-json-errors=true` or `--json-errors=false`.
- True values are `1`, `t`, `T`, `TRUE`, `true`, and `True`. False values are `0`, `f`, `F`, `FALSE`, `false`, and `False`. The last valid assignment encountered by the preliminary scan wins, even if ordinary parsing will fail at an earlier argument. An invalid assignment does not change that preliminary state and remains a normal parsing error.
- A registered nonboolean flag consumes its next argument as a value unless its value is attached with `=`. For example, `-config -json-errors` treats `-json-errors` as a file name and does not enable the mode. Similarly, `-paths -json-errors` is an invalid numeric value, not an opt-in. A consumed value of `--` is also data, as in normal Go flag parsing.
- Boolean flags do not consume the next argument. Thus `-json-errors false` enables JSON errors and then fails on the positional argument `false`; use `-json-errors=false` to disable the mode.
- An unconsumed `--`, a positional argument, or a lone `-` ends the scan. Options after that boundary do not change the mode. Unknown flags consume only their own token because they have no registered value type. Only one or two leading dashes are accepted for the opt-in.

For example, `-unknown -json-errors` produces a JSON startup error, while `-- -json-errors` retains the default text-only error behavior.

## Failures after metadata

A failed run emits its existing `type:error` record exactly once, whether or not `-json-errors` is enabled. The record retains `error` and `completed`, and includes:

- `elapsed_ns`: nonnegative wall-clock nanoseconds measured with the same boundaries as a successful summary, including selection, execution, and resource cleanup.
- `selection`: the same automatic-selection decision recorded in metadata, when automatic selection was used.
- `cuda`: the same CUDA statistics available to a successful summary, when a CUDA backend was selected. Successful batch/scenario counts and batch timing totals exclude failed batches. `SessionLifetime` includes the owned process lifetime through cleanup, including failed work and idle time.

CPU errors do not claim CUDA statistics. The completed count is the number of scenarios accepted by the output sink in this invocation, including when `-summary-only` omits their records. A failed run can contain a delivered prefix, and it never emits a successful summary.

For example, this valid configuration overflows numerically during a CPU run, after metadata:

```sh
go run ./cmd/quantaspindle -backend cpu -paths 1 -steps 1 -dt 1e308
```

Its final error record reports `completed:0` and `elapsed_ns`, with no `cuda` statistics. Startup error records emitted by `-json-errors` contain only `type` and `error`; they do not claim that a run began.
