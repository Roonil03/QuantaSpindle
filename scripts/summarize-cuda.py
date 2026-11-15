#!/usr/bin/env python3
"""Summarize qsbench JSONL without external packages; timings are full Runs."""
import collections
import json
import statistics
import sys

groups = collections.defaultdict(list)
for filename in sys.argv[1:]:
    with open(filename, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            config = record["config"]
            key = (config["Paths"], record["mode"], config["Workers"], config["BatchSize"])
            groups[key].append(record)

print("scenarios\tmode\tworkers\tbatch\tn\tmedian_ms\tmin_ms\tmax_ms\tscenarios/s\tkernel_ms\thost_ms\tgo_bytes\tgo_allocs")
for (paths, mode, workers, batch), records in sorted(groups.items()):
    elapsed = [r["elapsed_ns"] / 1e6 for r in records]
    median = statistics.median
    gpu = [r["cuda"] for r in records if "cuda" in r]
    fields = [paths, mode, workers, batch, len(records),
              f"{median(elapsed):.3f}", f"{min(elapsed):.3f}", f"{max(elapsed):.3f}",
              f"{median(r['scenarios_per_second'] for r in records):.0f}",
              f"{median(r['KernelTime'] for r in gpu) / 1e6:.3f}" if gpu else "-",
              f"{median(r['HostTime'] for r in gpu) / 1e6:.3f}" if gpu else "-",
              int(median(r["go_allocated_bytes"] for r in records)),
              int(median(r["go_allocations"] for r in records))]
    print("\t".join(map(str, fields)))
