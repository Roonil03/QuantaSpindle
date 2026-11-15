#!/usr/bin/env python3
"""Summarize distinct experiments; one reusable trial contributes one median."""
import collections
import json
import statistics
import sys

groups = collections.defaultdict(list)
seen = set()
for filename in sys.argv[1:]:
    with open(filename, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            config = json.dumps(record["config"], sort_keys=True)
            key = (filename, record["mode"], config)
            trial = (*key, record["trial"])
            if trial in seen:
                raise ValueError(f"duplicate trial {trial}")
            seen.add(trial)
            groups[key].append(record)

def distribution(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), trials=values)

for (filename, mode, config), records in sorted(groups.items()):
    records.sort(key=lambda record: record["trial"])
    result = dict(file=filename, mode=mode, config=json.loads(config), n=len(records))
    for field in ("elapsed_ns", "amortized_ns", "prepare_ns", "close_ns", "first_run_ns"):
        if field in records[0]:
            result[field.removesuffix("_ns") + "_ms"] = distribution([r[field] / 1e6 for r in records])
    if "ready_run_ns" in records[0]:
        result["ready_ms"] = distribution([statistics.median(r["ready_run_ns"]) / 1e6 for r in records])
    print(json.dumps(result))
