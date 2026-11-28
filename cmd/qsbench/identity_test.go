package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

func TestBuildIdentity(t *testing.T) {
	for _, tc := range []struct {
		name     string
		info     *debug.BuildInfo
		ok       bool
		version  string
		revision string
		modified *bool
	}{
		{"versioned clean", &debug.BuildInfo{Main: debug.Module{Path: "github.com/Roonil03/QuantaSpindle", Version: "v0.1.2"}, Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "abc123"}, {Key: "vcs.modified", Value: "false"}}}, true, "v0.1.2", "abc123", boolPointer(false)},
		{"local modified", &debug.BuildInfo{Main: debug.Module{Path: "github.com/Roonil03/QuantaSpindle", Version: "(devel)"}, Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "def456"}, {Key: "vcs.modified", Value: "true"}}}, true, "(devel)", "def456", boolPointer(true)},
		{"no VCS settings", &debug.BuildInfo{Main: debug.Module{Version: "v0.1.2"}}, true, "v0.1.2", "", nil},
		{"invalid modified setting", &debug.BuildInfo{Settings: []debug.BuildSetting{{Key: "vcs.modified", Value: "unknown"}}}, true, "", "", nil},
		{"unavailable", nil, false, "", "", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := identifyBuild(tc.info, tc.ok)
			if got.Available != tc.ok || got.ModuleVersion != tc.version || got.VCSRevision != tc.revision || !reflect.DeepEqual(got.VCSModified, tc.modified) {
				t.Fatalf("identity = %+v", got)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err := json.Unmarshal(encoded, &fields); err != nil {
				t.Fatal(err)
			}
			_, present := fields["vcs_modified"]
			if present != (tc.modified != nil) {
				t.Fatalf("unknown modified state must not be reported as clean: %s", encoded)
			}
		})
	}
}

func boolPointer(value bool) *bool { return &value }

func TestBuildIdentityEmittedAcrossRecords(t *testing.T) {
	for _, info := range []*debug.BuildInfo{
		{Main: debug.Module{Path: "github.com/Roonil03/QuantaSpindle", Version: "v0.1.2"}, Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "abc123"}, {Key: "vcs.modified", Value: "false"}}},
		{Main: debug.Module{Path: "github.com/Roonil03/QuantaSpindle", Version: "(devel)"}, Settings: []debug.BuildSetting{{Key: "vcs.revision", Value: "def456"}, {Key: "vcs.modified", Value: "true"}}},
		nil,
	} {
		deps := defaultBenchmarkDependencies()
		calls := 0
		deps.build = func() (*debug.BuildInfo, bool) { calls++; return info, info != nil }
		records := collectIdentityRecords(t, deps, "-backend", "cpu", "-paths", "64", "-steps", "64")
		if calls != 1 || len(records) != 2 {
			t.Fatalf("build info not collected once: calls=%d records=%d", calls, len(records))
		}
		for _, record := range records {
			if !reflect.DeepEqual(record.Build, identifyBuild(info, info != nil)) {
				t.Fatalf("wrong emitted build identity: %+v", record.Build)
			}
		}
	}
}

func TestRunnerIdentityUnavailable(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing-runner")
	got := identifyRunner(missing)
	if got.Available || got.Path != "" || got.SHA256 != "" || got.Error == "" {
		t.Fatalf("guessed missing executable identity: %+v", got)
	}
	got = hashRunner(missing, os.Open)
	if got.Available || got.Path != missing || got.SHA256 != "" || got.Error == "" {
		t.Fatalf("guessed unreadable executable digest: %+v", got)
	}
	got = hashRunner(t.TempDir(), os.Open)
	if got.Available || got.SHA256 != "" || got.Error == "" {
		t.Fatalf("guessed unreadable stream digest: %+v", got)
	}
}

// The test binary is a portable mock runner for both process execution modes.
func init() {
	if os.Getenv("QS_BENCH_TEST_RUNNER") == "1" {
		os.Exit(benchmarkRunnerHelper())
	}
}

func benchmarkRunnerHelper() int {
	stream := len(os.Args) == 2 && os.Args[1] == "--stream"
	if stream {
		_, _ = os.Stdout.Write([]byte("QSPNCS01"))
	}
	for {
		var request [72]byte
		if _, err := io.ReadFull(os.Stdin, request[:]); err == io.EOF {
			return 0
		} else if err != nil {
			return 2
		}
		count := int(binary.LittleEndian.Uint64(request[16:]))
		response := make([]byte, 176+32*count)
		copy(response, "QSPNCR01")
		binary.LittleEndian.PutUint32(response[8:], 1)
		binary.LittleEndian.PutUint64(response[16:], uint64(count))
		copy(response[48:], "qsbench-mock-device")
		for i := 0; i < count; i++ {
			binary.LittleEndian.PutUint64(response[176+32*i:], math.Float64bits(100))
		}
		if _, err := os.Stdout.Write(response); err != nil {
			return 3
		}
		if !stream {
			return 0
		}
	}
}

func copyBenchmarkRunner(t *testing.T, directory, name, suffix string) (string, string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(directory, name)
	source, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	// PE and ELF loaders permit trailing bytes. Distinct overlays let the same
	// protocol helper have different actual executable byte digests.
	data = append(data, []byte(suffix)...)
	if err := os.WriteFile(path, data, 0700); err != nil {
		t.Fatal(err)
	}
	return path, fmt.Sprintf("%x", sha256.Sum256(data))
}

type identityRecord struct {
	Mode       string
	Build      buildIdentity
	Runner     *runnerIdentity `json:"cuda_runner"`
	Elapsed    int64           `json:"elapsed_ns"`
	Summary    json.RawMessage
	Statistics json.RawMessage `json:"cuda"`
}

func collectIdentityRecords(t *testing.T, deps benchmarkDependencies, args ...string) []identityRecord {
	t.Helper()
	oldFlags, oldArgs, oldStdout := flag.CommandLine, os.Args, os.Stdout
	defer func() { flag.CommandLine, os.Args, os.Stdout = oldFlags, oldArgs, oldStdout }()
	flag.CommandLine = flag.NewFlagSet("qsbench", flag.ContinueOnError)
	os.Args = append([]string{"qsbench", "-paths", "1", "-batches", "1", "-steps", "1", "-trials", "2"}, args...)
	output, err := os.CreateTemp(t.TempDir(), "records-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	os.Stdout = output
	if err := runWithDependencies(deps); err != nil {
		t.Fatal(err)
	}
	if _, err := output.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	var records []identityRecord
	decoder := json.NewDecoder(output)
	for {
		var record identityRecord
		if err := decoder.Decode(&record); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	return records
}

func TestBenchmarkRunnerIdentitySelectionAndStability(t *testing.T) {
	dir := t.TempDir()
	first, firstDigest := copyBenchmarkRunner(t, dir, "runner-first", "first")
	second, secondDigest := copyBenchmarkRunner(t, dir, "runner-second", "second")
	defaultRunner, defaultDigest := copyBenchmarkRunner(t, dir, "quantaspindle-cuda", "default")
	t.Setenv("QS_BENCH_TEST_RUNNER", "1")
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	for _, tc := range []struct {
		name, primary, legacy string
		args                  []string
		want                  map[string]runnerIdentity
	}{
		{"separate flags", second, second, []string{"-cuda-runner", first, "-cuda-batch-runner", second}, map[string]runnerIdentity{"cuda": {Path: first, SHA256: firstDigest}, "cuda-batch": {Path: second, SHA256: secondDigest}}},
		{"shared explicit PATH name", second, second, []string{"-cuda-runner", "runner-first"}, map[string]runnerIdentity{"cuda": {Path: first, SHA256: firstDigest}, "cuda-batch": {Path: first, SHA256: firstDigest}}},
		{"primary environment", first, second, nil, map[string]runnerIdentity{"cuda": {Path: first, SHA256: firstDigest}, "cuda-batch": {Path: first, SHA256: firstDigest}}},
		{"batch override PATH name", first, first, []string{"-cuda-batch-runner", "runner-second"}, map[string]runnerIdentity{"cuda": {Path: first, SHA256: firstDigest}, "cuda-batch": {Path: second, SHA256: secondDigest}}},
		{"legacy environment PATH name", "", "runner-second", nil, map[string]runnerIdentity{"cuda": {Path: second, SHA256: secondDigest}, "cuda-batch": {Path: second, SHA256: secondDigest}}},
		{"PATH default", "", "", nil, map[string]runnerIdentity{"cuda": {Path: defaultRunner, SHA256: defaultDigest}, "cuda-batch": {Path: defaultRunner, SHA256: defaultDigest}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("QS_CUDA_RUNNER", tc.primary)
			t.Setenv("QUANTASPINDLE_CUDA_RUNNER", tc.legacy)
			deps := defaultBenchmarkDependencies()
			calls := 0
			deps.runner = func(path string) runnerIdentity {
				calls++
				return identifyRunner(path)
			}
			records := collectIdentityRecords(t, deps, append([]string{"-backend", "cuda,cuda-batch"}, tc.args...)...)
			if calls != 2 || len(records) != 4 {
				t.Fatalf("identity collection repeated during trials: calls=%d records=%d", calls, len(records))
			}
			for _, record := range records {
				want := tc.want[record.Mode]
				if record.Runner == nil || !record.Runner.Available || record.Runner.Path != want.Path || record.Runner.SHA256 != want.SHA256 || record.Elapsed <= 0 || len(record.Statistics) == 0 {
					t.Fatalf("unexpected identity record: %+v", record)
				}
			}
		})
	}
	if firstDigest == secondDigest {
		t.Fatal("changing executable bytes did not change digest")
	}
}

func TestIdentityCollectionOutsideTrialClockAndPinnedExecution(t *testing.T) {
	dir := t.TempDir()
	path, _ := copyBenchmarkRunner(t, dir, "trusted-runner", "timing")
	t.Setenv("QS_BENCH_TEST_RUNNER", "1")
	t.Setenv("QS_CUDA_RUNNER", path)
	t.Setenv("QUANTASPINDLE_CUDA_RUNNER", "missing-runner")
	deps := defaultBenchmarkDependencies()
	clock := time.Unix(0, 0)
	deps.now = func() time.Time {
		clock = clock.Add(time.Nanosecond)
		return clock
	}
	deps.build = func() (*debug.BuildInfo, bool) {
		clock = clock.Add(time.Hour)
		return nil, false
	}
	deps.runner = func(selected string) runnerIdentity {
		clock = clock.Add(time.Hour)
		identity := identifyRunner(selected)
		// Execution must use the resolved path, not repeat environment lookup.
		if err := os.Setenv("QS_CUDA_RUNNER", "missing-runner"); err != nil {
			t.Fatal(err)
		}
		return identity
	}
	records := collectIdentityRecords(t, deps, "-backend", "cuda")
	for _, record := range records {
		if record.Elapsed != 1 || record.Build.Available || record.Runner == nil || record.Runner.Path != path {
			t.Fatalf("metadata entered the trial clock or path was not pinned: %+v", record)
		}
	}
}

func TestUnreadableIdentityDoesNotPreventExecution(t *testing.T) {
	t.Setenv("QS_BENCH_TEST_RUNNER", "1")
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	deps := defaultBenchmarkDependencies()
	deps.runner = func(string) runnerIdentity {
		return hashRunner(path, func(string) (*os.File, error) { return nil, errors.New("identity read denied") })
	}
	for _, record := range collectIdentityRecords(t, deps, "-backend", "cuda") {
		if record.Runner == nil || record.Runner.Available || record.Runner.SHA256 != "" || record.Runner.Error != "identity read denied" || record.Runner.Path != path {
			t.Fatalf("unavailable identity was not explicit: %+v", record)
		}
	}
	deps.runner = func(string) runnerIdentity { t.Fatal("CPU mode looked up a CUDA runner"); return runnerIdentity{} }
	for _, record := range collectIdentityRecords(t, deps, "-backend", "cpu", "-paths", "64", "-steps", "64") {
		if record.Runner != nil {
			t.Fatal("CPU record has a runner identity")
		}
	}
}
