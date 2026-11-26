package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/hybrid"
)

// The test executable doubles as a portable, invocation-detecting CUDA runner.
func TestMain(m *testing.M) {
	if marker := os.Getenv("QS_CLI_TEST_RUNNER_MARKER"); marker != "" {
		if err := os.WriteFile(marker, []byte("invoked"), 0600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		os.Exit(1)
	}
	os.Exit(m.Run())
}

func buildCLI(t *testing.T) string {
	t.Helper()
	binary := filepath.Join(t.TempDir(), "quantaspindle")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if output, err := exec.Command("go", "build", "-o", binary, ".").CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v\n%s", err, output)
	}
	return binary
}

func TestCLIProcessExitStatus(t *testing.T) {
	binary := buildCLI(t)
	for _, arg := range []string{"-h", "--help", "-unknown-flag"} {
		t.Run(arg, func(t *testing.T) {
			output, err := exec.Command(binary, arg).CombinedOutput()
			if arg == "-unknown-flag" {
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() == 0 {
					t.Fatalf("unknown flag exit: %v\n%s", err, output)
				}
				return
			}
			if err != nil || !bytes.Contains(output, []byte("Usage of quantaspindle:")) || bytes.Contains(output, []byte("flag: help requested")) {
				t.Fatalf("help failed: %v\n%s", err, output)
			}
		})
	}
}

func decodeCLIRecords(t *testing.T, output []byte) []map[string]json.RawMessage {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(output))
	var records []map[string]json.RawMessage
	for {
		var record map[string]json.RawMessage
		if err := decoder.Decode(&record); err == io.EOF {
			return records
		} else if err != nil {
			t.Fatalf("non-JSON stdout: %v", err)
		}
		records = append(records, record)
	}
}

func TestCLIProgressPreservesResults(t *testing.T) {
	args := []string{"-backend", "cpu", "-paths", "6", "-steps", "2", "-workers", "2", "-batch", "2"}
	for _, summaryOnly := range []bool{false, true} {
		baseArgs := append([]string(nil), args...)
		if summaryOnly {
			baseArgs = append(baseArgs, "-summary-only")
		}
		var baseline, diagnostics bytes.Buffer
		if err := execute(context.Background(), baseArgs, &baseline, &diagnostics); err != nil {
			t.Fatal(err)
		}
		if diagnostics.Len() != 0 {
			t.Fatalf("default progress enabled: %s", &diagnostics)
		}
		wantRecords := decodeCLIRecords(t, baseline.Bytes())
		for _, tc := range []struct {
			interval string
			counts   []int
		}{{"0", nil}, {"2", []int{2, 4, 6}}, {"4", []int{4, 6}}, {"10", []int{6}}} {
			t.Run(fmt.Sprintf("summary=%v/interval=%s", summaryOnly, tc.interval), func(t *testing.T) {
				var output, diagnostic bytes.Buffer
				if err := execute(context.Background(), append(baseArgs, "-progress-every", tc.interval), &output, &diagnostic); err != nil {
					t.Fatal(err)
				}
				var expected strings.Builder
				for _, count := range tc.counts {
					fmt.Fprintf(&expected, "quantaspindle: progress delivered=%d\n", count)
				}
				if diagnostic.String() != expected.String() {
					t.Fatalf("progress %q, want %q", diagnostic.String(), expected.String())
				}
				records := decodeCLIRecords(t, output.Bytes())
				if len(records) != len(wantRecords) {
					t.Fatalf("record count %d, want %d", len(records), len(wantRecords))
				}
				for i, record := range records {
					if !bytes.Equal(record["type"], wantRecords[i]["type"]) || !bytes.Equal(record["scenario"], wantRecords[i]["scenario"]) || !bytes.Equal(record["summary"], wantRecords[i]["summary"]) {
						t.Fatalf("progress changed results at record %d", i)
					}
				}
			})
		}
	}
}

type cancelOnWrite struct {
	bytes.Buffer
	match  []byte
	cancel context.CancelFunc
}

type failOnWrite struct {
	bytes.Buffer
	match []byte
	err   error
}

func (w *failOnWrite) Write(data []byte) (int, error) {
	if bytes.Contains(data, w.match) {
		return 0, w.err
	}
	return w.Buffer.Write(data)
}

func (w *cancelOnWrite) Write(data []byte) (int, error) {
	n, err := w.Buffer.Write(data)
	if bytes.Contains(w.Bytes(), w.match) {
		w.cancel()
	}
	return n, err
}

func TestCLIProgressCancellationPrefix(t *testing.T) {
	for _, summaryOnly := range []bool{false, true} {
		t.Run(fmt.Sprintf("summary=%v", summaryOnly), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var output, diagnostic bytes.Buffer
			var out, diag io.Writer = &output, &diagnostic
			args := []string{"-backend", "cpu", "-paths", "6", "-workers", "1", "-batch", "1", "-progress-every", "2"}
			wantCount := uint64(3)
			wantProgress := "quantaspindle: progress delivered=2\nquantaspindle: progress delivered=3\n"
			if summaryOnly {
				args = append(args, "-summary-only", "-steps", "2")
				w := &cancelOnWrite{match: []byte("delivered=2"), cancel: cancel}
				diag = w
				wantCount = 2
				wantProgress = "quantaspindle: progress delivered=2\n"
			} else {
				// Large records reach the underlying writer during delivery, so
				// cancellation is deterministic and needs no timing assumptions.
				args = append(args, "-store-paths", "-steps", "2000")
				out = &cancelOnWrite{match: []byte(`"type":"scenario","scenario":{"ID":2,`), cancel: cancel}
			}
			err := execute(ctx, args, out, diag)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation: %v", err)
			}
			if w, ok := out.(*cancelOnWrite); ok {
				output = w.Buffer
			}
			if w, ok := diag.(*cancelOnWrite); ok {
				diagnostic = w.Buffer
			}
			if diagnostic.String() != wantProgress {
				t.Fatalf("cancellation progress: %q, want %q", diagnostic.String(), wantProgress)
			}
			records := decodeCLIRecords(t, output.Bytes())
			for _, record := range records {
				if string(record["type"]) == `"summary"` {
					t.Fatal("claimed successful completion")
				}
			}
			last := records[len(records)-1]
			var completed uint64
			if err := json.Unmarshal(last["completed"], &completed); err != nil || string(last["type"]) != `"error"` || completed != wantCount {
				t.Fatalf("incorrect delivered prefix: %s (decode: %v)", output.Bytes(), err)
			}
		})
	}
}

func TestCLIProgressOutputFailurePrefix(t *testing.T) {
	writeErr := errors.New("output disconnected")
	output := &failOnWrite{match: []byte(`"type":"scenario","scenario":{"ID":3,`), err: writeErr}
	var diagnostic bytes.Buffer
	err := execute(context.Background(), []string{"-backend", "cpu", "-paths", "6", "-workers", "1", "-batch", "1", "-progress-every", "2", "-store-paths", "-steps", "2000"}, output, &diagnostic)
	if !errors.Is(err, writeErr) {
		t.Fatalf("output failure: %v", err)
	}
	want := "quantaspindle: progress delivered=2\nquantaspindle: progress delivered=3\n"
	if diagnostic.String() != want || bytes.Contains(output.Bytes(), []byte(`"type":"summary"`)) {
		t.Fatalf("incorrect failure progress: %q", diagnostic.String())
	}
}

func TestCLIValidationResolvedConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"Paths":7,"Steps":3,"Seed":22,"Workers":2,"BatchSize":3,"Assets":[{"InitialPrice":50,"Mu":0.1,"Sigma":0.3}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	args := []string{"-backend", "cpu", "-config", path, "-paths", "5", "-seed", "42", "-workers", "1", "-batch", "2", "-store-paths"}
	var normal, validation, diagnostic bytes.Buffer
	if err := execute(context.Background(), args, &normal, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := execute(context.Background(), append(args, "-validate-only", "-progress-every", "1"), &validation, &diagnostic); err != nil {
		t.Fatal(err)
	}
	runRecords := decodeCLIRecords(t, normal.Bytes())
	validationRecords := decodeCLIRecords(t, validation.Bytes())
	if len(validationRecords) != 1 || string(validationRecords[0]["type"]) != `"validation"` || string(validationRecords[0]["valid"]) != "true" || diagnostic.Len() != 0 {
		t.Fatalf("unexpected validation output: %s; diagnostic: %s", &validation, &diagnostic)
	}
	var runConfig, checkedConfig qs.Config
	if err := json.Unmarshal(runRecords[0]["config"], &runConfig); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(validationRecords[0]["config"], &checkedConfig); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(runConfig, checkedConfig) || checkedConfig.Paths != 5 || checkedConfig.Seed != 42 || checkedConfig.Assets[0].InitialPrice != 50 {
		t.Fatalf("validation changed config: run=%+v checked=%+v", runConfig, checkedConfig)
	}
}

func TestCLIValidationRejectsSameErrors(t *testing.T) {
	binary := buildCLI(t)
	for _, tc := range []struct {
		name, config, message string
		args                  []string
	}{
		{name: "missing file", config: `{}`, args: []string{"-config", filepath.Join(t.TempDir(), "missing.json")}, message: "missing.json"},
		{name: "invalid JSON", config: `{"Paths":`, message: "config:"},
		{name: "unknown field", config: `{"Unknown":1}`, message: `unknown field "Unknown"`},
		{name: "probability row", config: `{"Regimes":[{"Mu":[0],"Sigma":[0.2]}],"Transition":[[0.5]]}`, message: "must sum to one"},
		{name: "invalid override", config: `{}`, args: []string{"-paths", "0"}, message: "paths must be"},
		{name: "invalid backend", config: `{}`, args: []string{"-backend", "fpga"}, message: "unknown backend"},
		{name: "negative timeout", config: `{}`, args: []string{"-timeout", "-1s"}, message: "timeout must be nonnegative"},
		{name: "negative progress", config: `{}`, args: []string{"-progress-every", "-1"}, message: "progress-every must be nonnegative"},
		{name: "invalid flag", config: `{}`, args: []string{"-unknown-flag"}, message: "flag provided but not defined"},
		{name: "positional argument", config: `{}`, args: []string{"extra"}, message: "unexpected positional"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, []byte(tc.config), 0600); err != nil {
				t.Fatal(err)
			}
			args := append([]string{"-config", path}, tc.args...)
			var previous string
			for _, validateOnly := range []bool{false, true} {
				modeArgs := append([]string(nil), args...)
				if validateOnly {
					// Put this before positional arguments so flag parsing sees it.
					modeArgs = append([]string{"-validate-only"}, modeArgs...)
				}
				var output bytes.Buffer
				err := execute(context.Background(), modeArgs, &output, io.Discard)
				if err == nil || !strings.Contains(err.Error(), tc.message) || output.Len() != 0 {
					t.Fatalf("validation=%v: err=%v, output=%s", validateOnly, err, &output)
				}
				if validateOnly && err.Error() != previous {
					t.Fatalf("validation changed error: %v, want %s", err, previous)
				}
				processOutput, processErr := exec.Command(binary, modeArgs...).CombinedOutput()
				var exitErr *exec.ExitError
				if !errors.As(processErr, &exitErr) || exitErr.ExitCode() == 0 || !bytes.Contains(processOutput, []byte(tc.message)) {
					t.Fatalf("process validation=%v: err=%v, output=%s", validateOnly, processErr, processOutput)
				}
				previous = err.Error()
			}
		})
	}
}

func TestCLIValidationDoesNotStartCUDARunner(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "cuda-invoked")
	t.Setenv("QS_CLI_TEST_RUNNER_MARKER", marker)
	runner, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Verify the fake runner really does leave a marker when invoked.
	if err := exec.Command(runner, "--stream").Run(); err == nil {
		t.Fatal("fake runner should fail")
	}
	if err := os.Remove(marker); err != nil {
		t.Fatalf("fake runner did not create marker: %v", err)
	}
	for _, backend := range []string{"auto", "cuda"} {
		var output bytes.Buffer
		args := []string{"-validate-only", "-backend", backend, "-cuda-runner", runner, "-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps)}
		if backend == "cuda" {
			// Configuration validity does not imply CUDA model support.
			args = append(args, "-store-paths")
		}
		if err := execute(context.Background(), args, &output, io.Discard); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("validation invoked CUDA runner: %v", err)
		}
		records := decodeCLIRecords(t, output.Bytes())
		if len(records) != 1 || string(records[0]["type"]) != `"validation"` {
			t.Fatalf("validation generated run records: %s", &output)
		}
	}
}

func TestCLIStreamsOrderedRecords(t *testing.T) {
	var output bytes.Buffer
	err := execute(context.Background(), []string{"-paths", "5", "-steps", "2", "-batch", "2", "-workers", "2"}, &output, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(&output)
	var records []map[string]json.RawMessage
	for decoder.More() {
		var record map[string]json.RawMessage
		if err := decoder.Decode(&record); err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if len(records) != 7 {
		t.Fatalf("got %d records", len(records))
	}
	if string(records[0]["type"]) != `"metadata"` || string(records[6]["type"]) != `"summary"` {
		t.Fatal("missing provenance or final summary")
	}
	for i := 0; i < 5; i++ {
		var s struct{ ID uint64 }
		if err := json.Unmarshal(records[i+1]["scenario"], &s); err != nil {
			t.Fatal(err)
		}
		if s.ID != uint64(i) {
			t.Fatal("out of order", s.ID)
		}
	}
}

func TestCLIExplicitAssetsReplaceDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	data := `{"assets":[{"InitialPrice":50,"Sigma":0},{"InitialPrice":50,"Sigma":0}]}`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := execute(context.Background(), []string{"-config", path, "-paths", "1", "-steps", "1", "-dt", "1"}, &output, io.Discard); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(&output)
	var metadata struct {
		Config qs.Config `json:"config"`
	}
	var record struct {
		Scenario qs.Scenario `json:"scenario"`
	}
	if err := decoder.Decode(&metadata); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Decode(&record); err != nil {
		t.Fatal(err)
	}
	if len(metadata.Config.Assets) != 2 || metadata.Config.Assets[0].Mu != 0 || metadata.Config.Assets[1].Mu != 0 {
		t.Fatalf("unexpected model provenance: %+v", metadata.Config.Assets)
	}
	if len(record.Scenario.FinalPrices) != 2 || record.Scenario.FinalPrices[0] != 50 || record.Scenario.FinalPrices[1] != 50 {
		t.Fatalf("identical assets diverged: %+v", record.Scenario.FinalPrices)
	}
}

func TestCLIRejectsInvalidArguments(t *testing.T) {
	for _, args := range [][]string{{"-backend", "fpga"}, {"-paths", "0"}, {"-timeout", "-1s"}, {"extra"}} {
		if err := execute(context.Background(), args, io.Discard, io.Discard); err == nil {
			t.Fatal("accepted", args)
		}
	}
}

func TestCLIPropagatesCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var output bytes.Buffer
	if err := execute(ctx, []string{"-paths", "4"}, &output, io.Discard); err == nil {
		t.Fatal("accepted canceled context")
	}
	if strings.Contains(output.String(), `"type":"summary"`) {
		t.Fatal("claimed complete on cancellation")
	}
}

func TestCLIRejectsMalformedConfig(t *testing.T) {
	for _, content := range []string{`null`, `{"Unknown":1}`, `{} {}`, `[]`} {
		path := filepath.Join(t.TempDir(), "config.json")
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		if err := execute(context.Background(), []string{"-config", path, "-paths", "1", "-steps", "1"}, io.Discard, io.Discard); err == nil {
			t.Fatal("accepted", content)
		}
	}
}

func TestCLIDefaultAutoReportsActualCPUAndPreservesConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"Workers":2,"BatchSize":3}`), 0600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := execute(context.Background(), []string{"-config", path, "-paths", "5", "-steps", "2", "-cuda-runner", filepath.Join(t.TempDir(), "does-not-exist")}, &output, io.Discard); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(&output)
	var metadata struct {
		Backend   string                           `json:"backend"`
		Requested string                           `json:"requested_backend"`
		Config    qs.Config                        `json:"config"`
		Execution qs.Config                        `json:"execution_config"`
		Selection struct{ Backend, Reason string } `json:"selection"`
	}
	if err := decoder.Decode(&metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Requested != "auto" || metadata.Backend != (qs.CPU{}).Name() || metadata.Selection.Backend != metadata.Backend || metadata.Selection.Reason != "below-threshold" {
		t.Fatalf("misleading provenance: %+v", metadata)
	}
	if metadata.Config.Workers != 2 || metadata.Config.BatchSize != 3 || metadata.Execution.Workers != 2 || metadata.Execution.BatchSize != 3 {
		t.Fatal("explicit execution settings changed")
	}
	var last struct {
		Summary qs.RunSummary `json:"summary"`
	}
	for decoder.More() {
		if err := decoder.Decode(&last); err != nil {
			t.Fatal(err)
		}
	}
	if last.Summary.Backend != metadata.Backend || last.Summary.Scenarios != 5 {
		t.Fatalf("summary disagrees with selection: %+v", last.Summary)
	}
}

func TestCLIExplicitCUDAStillFailsWithoutFallback(t *testing.T) {
	var output bytes.Buffer
	err := execute(context.Background(), []string{"-backend", "cuda", "-paths", "1", "-steps", "1", "-cuda-runner", filepath.Join(t.TempDir(), "missing")}, &output, io.Discard)
	if err == nil || strings.Contains(output.String(), `"type":"scenario"`) || strings.Contains(output.String(), `"type":"summary"`) {
		t.Fatal("explicit CUDA silently fell back", err)
	}
}
