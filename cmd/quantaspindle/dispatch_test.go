package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/hybrid"
)

func dispatchRunner(t *testing.T, mode string) (string, string) {
	t.Helper()
	runner, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	log := filepath.Join(t.TempDir(), "runner.log")
	t.Setenv("QS_CLI_TEST_CUDA_MODE", mode)
	t.Setenv("QS_CLI_TEST_CUDA_LOG", log)
	return runner, log
}

func TestCLIExplicitCUDARangeProcess(t *testing.T) {
	binary := buildCLI(t)
	runner, _ := dispatchRunner(t, "valid")
	base := []string{"-backend", "cuda", "-cuda-runner", runner, "-paths", "8", "-steps", "12", "-seed", "42", "-workers", "2", "-batch", "3"}
	output, diagnostic, err := diagnosticProcess(binary, base)
	if err != nil || diagnostic != "" {
		t.Fatalf("full fake CUDA run: %v %s", err, diagnostic)
	}
	full := scenariosFromRecords(t, decodeCLIRecords(t, output))
	for _, batch := range []string{"1", "2", "5"} {
		var joined []qs.Scenario
		for _, bounds := range [][2]uint64{{0, 3}, {3, 5}} {
			args := append(append([]string(nil), base...), "-start-id", fmt.Sprint(bounds[0]), "-count", fmt.Sprint(bounds[1]), "-batch", batch, "-progress-every", "2")
			output, diagnostic, err := diagnosticProcess(binary, args)
			if err != nil {
				t.Fatalf("CUDA range %v: %v %s", bounds, err, diagnostic)
			}
			records := decodeCLIRecords(t, output)
			checkRangeRecord(t, records[0], bounds[0], bounds[1])
			last := records[len(records)-1]
			checkRangeRecord(t, last, bounds[0], bounds[1])
			var summary qs.RunSummary
			if err := json.Unmarshal(last["summary"], &summary); err != nil || summary.Scenarios != bounds[1] || summary.Backend != "cuda" || string(last["scope"]) != `"range"` {
				t.Fatalf("CUDA range summary: %s (%v)", output, err)
			}
			wantProgress := "quantaspindle: progress delivered=2\n"
			if bounds[1] == 5 {
				wantProgress += "quantaspindle: progress delivered=4\nquantaspindle: progress delivered=5\n"
			} else {
				wantProgress += "quantaspindle: progress delivered=3\n"
			}
			if diagnostic != wantProgress {
				t.Fatalf("CUDA range progress: %q, want %q", diagnostic, wantProgress)
			}
			joined = append(joined, scenariosFromRecords(t, records)...)
		}
		if !reflect.DeepEqual(joined, full) {
			t.Fatalf("CLI CUDA range IDs/seeds/results changed with batch=%s", batch)
		}
	}
	args := append(append([]string(nil), base...), "-start-id", "3", "-count", "5", "-summary-only")
	output, _, err = diagnosticProcess(binary, args)
	if err != nil || len(decodeCLIRecords(t, output)) != 2 {
		t.Fatalf("summary-only CUDA range: %v %s", err, output)
	}
}

func TestCLICUDARangeValidatesBeforeStartup(t *testing.T) {
	runner, log := dispatchRunner(t, "valid")
	base := []string{"-backend", "cuda", "-cuda-runner", runner, "-paths", "8", "-steps", "2", "-start-id", "3", "-count", "2"}
	for _, bounds := range [][]string{{"-start-id", "8"}, {"-count", "0"}, {"-count", "18446744073709551615"}} {
		var output bytes.Buffer
		if err := execute(context.Background(), append(append([]string(nil), base...), bounds...), &output, io.Discard); err == nil || output.Len() != 0 {
			t.Fatalf("invalid bounds passed: %v err=%v output=%s", bounds, err, &output)
		}
	}
	for _, validateOnly := range []bool{false, true} {
		args := append(append([]string(nil), base...), "-store-paths")
		if validateOnly {
			args = append(args, "-validate-only")
		}
		var output bytes.Buffer
		err := execute(context.Background(), args, &output, io.Discard)
		if validateOnly {
			records := decodeCLIRecords(t, output.Bytes())
			if err != nil || len(records) != 1 || string(records[0]["type"]) != `"validation"` {
				t.Fatalf("generic validation unexpectedly checks CUDA capability: %v %s", err, &output)
			}
		} else {
			if err == nil || !strings.Contains(err.Error(), "CUDA capability unsupported") {
				t.Fatalf("CUDA unsupported model accepted: %v", err)
			}
			requireRuntimeRecord(t, output.Bytes(), 0, "CUDA capability unsupported")
		}
	}
	if _, err := os.Stat(log); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("bounds/model validation started the runner: %v", err)
	}
}

func TestCLICUDARangeUsesEffectiveCountForSessionValidation(t *testing.T) {
	runner, _ := dispatchRunner(t, "valid")
	for _, workers := range []string{"1", "1024"} {
		var output bytes.Buffer
		args := []string{"-backend", "cuda", "-cuda-runner", runner, "-paths", "1000000000", "-steps", "1000000", "-batch", "16384", "-workers", workers, "-start-id", "999999997", "-count", "3"}
		if err := execute(context.Background(), args, &output, io.Discard); err != nil {
			t.Fatalf("full Paths work/storage leaked into CUDA session validation: %v", err)
		}
		records := decodeCLIRecords(t, output.Bytes())
		checkRangeRecord(t, records[0], 999999997, 3)
		got := scenariosFromRecords(t, records)
		if len(got) != 3 || got[0].ID != 999999997 || got[2].ID != 999999999 {
			t.Fatalf("effective range IDs changed: %+v", got)
		}
	}
}

func TestCLISelectionPreviewMatchesNormalMetadata(t *testing.T) {
	for _, tc := range []struct {
		name, mode, reason string
		args               []string
		probeCount         int
	}{
		{"unsupported model", "valid", "unsupported-model", []string{"-paths", "3", "-steps", "2", "-store-paths"}, 0},
		{"below threshold", "valid", "below-threshold", []string{"-paths", "3", "-steps", "2"}, 0},
		{"unavailable", "unavailable", "cuda-unavailable", []string{"-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-workers", "1", "-batch", "2"}, 2},
		{"CUDA defaults", "valid", "cuda-threshold", []string{"-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps)}, 16384},
		{"CUDA overrides", "valid", "cuda-threshold", []string{"-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-workers", "2", "-batch", "3"}, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner, log := dispatchRunner(t, tc.mode)
			args := append([]string{"-backend", "auto", "-cuda-runner", runner, "-summary-only"}, tc.args...)
			var preview bytes.Buffer
			if err := execute(context.Background(), append(append([]string(nil), args...), "-select-only"), &preview, io.Discard); err != nil {
				t.Fatal(err)
			}
			records := decodeCLIRecords(t, preview.Bytes())
			if len(records) != 1 || string(records[0]["type"]) != `"selection"` || records[0]["summary"] != nil || records[0]["scenario"] != nil {
				t.Fatalf("preview emitted run records: %s", &preview)
			}
			var decision hybrid.Decision
			if err := json.Unmarshal(records[0]["selection"], &decision); err != nil || decision.Reason != tc.reason {
				t.Fatalf("preview decision: %+v %v", decision, err)
			}
			data, err := os.ReadFile(log)
			if tc.probeCount == 0 {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("CPU preview started CUDA: %s %v", data, err)
				}
			} else if err != nil || string(data) != fmt.Sprintf("start\nrequest 0 %d\nexit\n", tc.probeCount) {
				t.Fatalf("preview did not stop after one probe and close: %q %v", data, err)
			}
			// A normal run makes the same decision, then is canceled after one
			// delivered scenario to avoid completing the large fallback job.
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			diagnostic := &cancelOnWrite{match: []byte("delivered=1"), cancel: cancel}
			var normal bytes.Buffer
			err = execute(ctx, append(append([]string(nil), args...), "-progress-every", "1"), &normal, diagnostic)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("normal parity run did not stop at prefix: %v", err)
			}
			metadata := decodeCLIRecords(t, normal.Bytes())[0]
			for _, field := range []string{"config", "backend", "requested_backend", "selection", "execution_config", "precision", "go", "os", "arch", "gomaxprocs", "version"} {
				if records[0][field] == nil || !bytes.Equal(records[0][field], metadata[field]) {
					t.Fatalf("preview differs from normal metadata in %s:\n%s\n%s", field, &preview, &normal)
				}
			}
			if records[0]["started_utc"] == nil {
				t.Fatal("preview omitted start identity")
			}
		})
	}
}

func TestCLISelectionPreviewFailuresAndCleanup(t *testing.T) {
	for _, mode := range []string{"fail-first", "protocol-error", "shutdown-failure"} {
		t.Run(mode, func(t *testing.T) {
			runner, _ := dispatchRunner(t, mode)
			base := []string{"-backend", "auto", "-cuda-runner", runner, "-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-workers", "1", "-batch", "2", "-select-only"}
			for _, jsonErrors := range []bool{false, true} {
				args := append([]string(nil), base...)
				if jsonErrors {
					args = append(args, "-json-errors")
				}
				var output bytes.Buffer
				err := execute(context.Background(), args, &output, io.Discard)
				if err == nil {
					t.Fatal("failed probe/shutdown claimed successful selection")
				}
				if jsonErrors {
					requireStartupRecord(t, output.Bytes(), err.Error())
				} else if output.Len() != 0 {
					t.Fatalf("failed preview emitted a successful record: %s", &output)
				}
			}
		})
	}
	t.Run("output failure after closure", func(t *testing.T) {
		runner, log := dispatchRunner(t, "valid")
		writeErr := errors.New("output failed")
		output := diagnosticWriteFunc(func([]byte) (int, error) {
			data, err := os.ReadFile(log)
			if err != nil || !strings.HasSuffix(string(data), "exit\n") {
				t.Fatalf("preview wrote before closing runner: %q %v", data, err)
			}
			return 0, writeErr
		})
		args := []string{"-select-only", "-json-errors", "-cuda-runner", runner, "-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-batch", "2"}
		if err := execute(context.Background(), args, output, io.Discard); !errors.Is(err, writeErr) {
			t.Fatalf("preview output failure: %v", err)
		}
	})
}

func TestCLISelectionPreviewProcessAndModeValidation(t *testing.T) {
	binary := buildCLI(t)
	runner, log := dispatchRunner(t, "valid")
	base := []string{"-select-only", "-cuda-runner", runner, "-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-workers", "1", "-batch", "2"}
	output, diagnostic, err := diagnosticProcess(binary, base)
	if err != nil || diagnostic != "" {
		t.Fatalf("selection preview exit: %v %s", err, diagnostic)
	}
	records := decodeCLIRecords(t, output)
	if len(records) != 1 || string(records[0]["type"]) != `"selection"` {
		t.Fatalf("selection process emitted run records: %s", output)
	}
	data, err := os.ReadFile(log)
	if err != nil || string(data) != "start\nrequest 0 2\nexit\n" {
		t.Fatalf("selection process left runner open: %q %v", data, err)
	}
	if err := os.Remove(log); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range [][]string{{"-backend", "cpu"}, {"-backend", "cuda"}, {"-validate-only"}, {"-start-id", "0"}} {
		args := append(append([]string(nil), base...), invalid...)
		args = append(args, "-json-errors")
		output, diagnostic, err := diagnosticProcess(binary, args)
		requireDiagnosticFailure(t, err, diagnostic, "")
		requireStartupRecord(t, output, "")
	}
	if _, err := os.Stat(log); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("invalid preview options started a runner: %v", err)
	}
	t.Setenv("QS_CLI_TEST_CUDA_MODE", "protocol-error")
	output, diagnostic, err = diagnosticProcess(binary, append(base, "-json-errors"))
	requireDiagnosticFailure(t, err, diagnostic, "invalid CUDA runner response")
	requireStartupRecord(t, output, "invalid CUDA runner response")
}
