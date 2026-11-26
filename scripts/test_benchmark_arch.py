"""Exercise benchmark build commands without compiling or running CUDA code."""

import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parent
STUB = r'''
import json
import os
import pathlib
import sys

command = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if command == "nvcc":
    if args == ["--version"]:
        print("stub nvcc: no CUDA compilation")
    else:
        with open(os.environ["NVCC_LOG"], "a") as log:
            log.write(json.dumps(args) + "\n")
        pathlib.Path(args[args.index("-o") + 1]).write_text("stub native binary\n")
elif command == "go":
    if args[0] == "version":
        print("stub Go toolchain")
    elif args[0] == "env":
        print("linux\namd64\n0")
    elif args[0] == "build":
        if os.environ.get("STOP_GO_BUILD") == "1":
            sys.exit(73)
        pathlib.Path(args[args.index("-o") + 1]).write_text("stub Go binary\n")
    elif args[0] == "test":
        print("stopping before device execution and timing")
        sys.exit(73)
    else:
        sys.exit("unexpected go invocation")
elif command == "git":
    if "archive" in args:
        sys.stdout.buffer.write(pathlib.Path(os.environ["BASELINE_ARCHIVE"]).read_bytes())
    elif "show" in args:
        print("// synthetic tagged CUDA source")
    elif "rev-parse" in args:
        print("synthetic source identity for command testing")
    elif "diff" not in args:
        sys.exit("unexpected git invocation")
else:
    print("stub " + command + ": no GPU access")
'''


@unittest.skipUnless(sys.platform.startswith("linux"), "benchmark environment checks require Linux")
class BenchmarkArchitectureTest(unittest.TestCase):
    def test_selected_architecture_and_metadata(self):
        for script, phase, builds, status in [
            ("benchmark-cuda.sh", "measure", 2, 73),
            ("benchmark-cuda.sh", "baseline", 1, 73),
            ("benchmark-startup.sh", "measure", 2, 73),
            ("benchmark-startup.sh", "baseline-build", 1, 0),
        ]:
            for selected in (None, "", "sm_80"):
                with self.subTest(script=script, phase=phase, CUDA_ARCH=selected):
                    self.check_script(script, phase, builds, status, selected)

    def check_script(self, script, phase, builds, status, selected):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("engine.go", "cuda/backend.go", "accelerators/cuda/runner.cu", "cmd/qsbench/main.go"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// synthetic source fixture\n")
            (root / "scripts").mkdir()
            shutil.copyfile(SCRIPTS / script, root / "scripts" / script)
            archive = root / "baseline.tar"
            with tarfile.open(archive, "w") as tar:
                content = b"// synthetic tagged CUDA source\n"
                entry = tarfile.TarInfo("accelerators/cuda/runner.cu")
                entry.size = len(content)
                tar.addfile(entry, io.BytesIO(content))
            stubs = root / "stubs"
            stubs.mkdir()
            for command in ("nvcc", "go", "git", "c++", "nvidia-smi"):
                path = stubs / command
                path.write_text(f"#!{sys.executable}\n" + STUB)
                path.chmod(0o755)
            env = os.environ.copy()
            env.pop("CUDA_ARCH", None)
            if selected is not None:
                env["CUDA_ARCH"] = selected
            env.update(PATH=str(stubs) + os.pathsep + env["PATH"], PHASE=phase,
                       OUT="benchmarks/generated/arch-check", NVCC_LOG=str(root / "nvcc.jsonl"),
                       BASELINE_ARCHIVE=str(archive), TRIALS="1", REPEATS="1", PATHS_LIST="1",
                       STOP_GO_BUILD="1" if script == "benchmark-cuda.sh" and phase == "baseline" else "0")
            result = subprocess.run(["sh", "scripts/" + script], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, status, result.stdout + result.stderr)
            commands = [json.loads(line) for line in (root / "nvcc.jsonl").read_text().splitlines()]
            self.assertEqual(len(commands), builds)
            target = selected or "sm_86"
            for args in commands:
                self.assertEqual([arg for arg in args if arg.startswith("-arch=")], ["-arch=" + target])
                for flag in ("-std=c++17", "-O3", "--fmad=false", "-cudart=static"):
                    self.assertIn(flag, args)
            tagged = "runner-v0.1.0.cu" if script == "benchmark-cuda.sh" else "v0.1.1/accelerators/cuda/runner.cu"
            self.assertTrue(any(arg.endswith(tagged) for arg in commands[0]))
            if builds == 2:
                self.assertIn("accelerators/cuda/runner.cu", commands[1])
            metadata = "environment-" + phase + ".txt" if script == "benchmark-cuda.sh" else "environment.txt"
            environment = (root / env["OUT"] / metadata).read_text()
            self.assertEqual([line for line in environment.splitlines() if line.startswith("CUDA_ARCH=")],
                             ["CUDA_ARCH=" + target])
            # The measure-phase stub exits at go test, before benchmarks run.
            self.assertFalse((root / env["OUT"] / "measure.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
