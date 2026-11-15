import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("summarize-cuda.py")


class SummarizerTest(unittest.TestCase):
    def test_configurations_and_input_files_are_not_pooled(self):
        def record(steps):
            return {"config": {"Paths": 10, "Workers": 1, "BatchSize": 10, "Steps": steps, "Seed": 1},
                    "mode": "cpu", "trial": 1, "elapsed_ns": 1000000,
                    "scenarios_per_second": 10000, "go_allocated_bytes": 1, "go_allocations": 1}
        with tempfile.TemporaryDirectory() as directory:
            first, second = (pathlib.Path(directory) / name for name in ("baseline.jsonl", "measure.jsonl"))
            first.write_text(json.dumps(record(1)) + "\n" + json.dumps(record(2)) + "\n", encoding="utf-8")
            second.write_text(json.dumps(record(1)) + "\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(first), str(second)], capture_output=True, text=True, check=True)
            self.assertEqual(len(result.stdout.splitlines()), 4)
            self.assertIn('"Steps":2', result.stdout)
            second.write_text((json.dumps(record(1)) + "\n") * 2, encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(second)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate trial", result.stderr)


if __name__ == "__main__":
    unittest.main()
