#!/usr/bin/env python3
"""Compare exact replay and routing summaries after (never during) timing."""
import json
import collections
import sys

def read(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source]

baseline, current, alternative = map(read, sys.argv[1:4])
trials = int(sys.argv[4])
paths_expected = set(map(int, sys.argv[5].split(",")))
expected_modes = ({"cuda-v0.1.1"}, {"cpu-default", "cpu", "cuda-cold", "cuda-reuse", "auto-cold", "auto-reuse"}, {"cuda-cold"})
for filename, records, modes in zip(sys.argv[1:4], (baseline, current, alternative), expected_modes):
    if {r["mode"] for r in records} != modes or {r["config"]["Paths"] for r in records} != paths_expected:
        raise ValueError(f"missing or extra mode/path coverage in {filename}")
    grouped = collections.defaultdict(set)
    for record in records:
        key = (record["mode"], json.dumps(record["config"], sort_keys=True))
        grouped[key].add(record["trial"])
    if any(value != set(range(1, trials+1)) for value in grouped.values()):
        raise ValueError(f"incomplete trial group in {filename}")
    expected_groups = len(paths_expected) * (17 if records is current else 1)
    if len(grouped) != expected_groups or len(records) != expected_groups * trials:
        raise ValueError(f"incomplete configuration sweep in {filename}")
for paths in paths_expected:
    sweep = {(r["config"]["Workers"], r["config"]["BatchSize"]) for r in current if r["mode"] == "cpu" and r["config"]["Paths"] == paths}
    if sweep != {(w,b) for w in (1,4,8,16) for b in (64,128,1024)}:
        raise ValueError(f"incomplete CPU sweep at {paths}")
expected = {r["config"]["Paths"]: r["summary"] for r in baseline}
cpu = {r["config"]["Paths"]: r["summary"] for r in current if r["mode"] == "cpu-default"}
phased = any("phase" in r for records in (baseline, current, alternative) for r in records)
seen = set()
for filename, records in zip(sys.argv[1:4], (baseline, current, alternative)):
    for record in records:
        key = (filename, record["mode"], json.dumps(record["config"], sort_keys=True), record["trial"])
        if key in seen:
            raise ValueError(f"duplicate trial: {key}")
        seen.add(key)
        paths = record["config"]["Paths"]
        summary = record["summary"]
        if phased:
            if records is baseline:
                expected_phase, expected_block = "cold-pair", 1 if record["trial"] % 2 else 2
            elif records is alternative:
                expected_phase, expected_block = "tagged-native-alternative", 3
            elif record["mode"] == "cuda-cold":
                expected_phase, expected_block = "cold-pair", 2 if record["trial"] % 2 else 1
            else:
                expected_phase, expected_block = "main", 4
            if record.get("phase") != expected_phase or record.get("block_order") != expected_block:
                raise ValueError(f"incorrect timing phase/order: {key}")
        if "operations" in record:
            operations = record["operations"]
            runs = record["run_elapsed_ns"]
            if len(runs) != operations or min(runs) <= 0 or runs[0] != record["first_run_ns"]:
                raise ValueError(f"inconsistent Run clocks: {key}")
            if record["elapsed_ns"] < sum(runs) + record["prepare_ns"] + record["close_ns"]:
                raise ValueError(f"lifecycle excludes measured work: {key}")
            if record["backend"] != summary["Backend"]:
                raise ValueError(f"incorrect backend label: {key}")
            if record["mode"] == "auto-reuse" and (len(record.get("decisions", [])) != operations or not all(d["Warm"] for d in record["decisions"])):
                raise ValueError(f"missing warm routing decisions: {key}")
        reference = expected[paths] if summary["Backend"] == "cuda" else cpu[paths]
        if summary != reference:
            raise ValueError(f"fixed-seed summary mismatch: {key}")
        if "decision" in record and record["decision"]["Backend"] != summary["Backend"]:
            raise ValueError(f"recorded route differs from summary: {key}")
        for decision in record.get("decisions", []):
            if decision["Backend"] != summary["Backend"]:
                raise ValueError(f"repeated route differs from summary: {key}")
print(f"PASS: {sum(map(len, (baseline, current, alternative)))} unique records; exact tagged GPU replay and CPU-routed summary agreement")
