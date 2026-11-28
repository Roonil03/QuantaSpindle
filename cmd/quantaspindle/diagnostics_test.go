package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	qs "github.com/Roonil03/QuantaSpindle"
	"github.com/Roonil03/QuantaSpindle/cuda"
	"github.com/Roonil03/QuantaSpindle/hybrid"
)

func TestScanJSONErrors(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	fs.Bool("json-errors", false, "")
	fs.Bool("summary-only", false, "")
	fs.String("config", "", "")
	fs.Int("paths", 1, "")
	for _, tc := range []struct {
		args []string
		want bool
	}{
		{nil, false},
		{[]string{"-unknown", "-json-errors"}, true},
		{[]string{"-paths", "bad", "--json-errors"}, true},
		{[]string{"-paths=bad", "-json-errors"}, true},
		{[]string{"-summary-only=bad", "-json-errors"}, true},
		{[]string{"-config", "-json-errors"}, false},
		{[]string{"-paths", "-json-errors"}, false},
		{[]string{"-config=-json-errors"}, false},
		{[]string{"-config", "--", "-json-errors"}, true},
		{[]string{"-summary-only", "-json-errors"}, true},
		{[]string{"--", "-json-errors"}, false},
		{[]string{"extra", "-json-errors"}, false},
		{[]string{"-", "-json-errors"}, false},
		{[]string{"-json-errors", "--", "-json-errors=false"}, true},
		{[]string{"-json-errors", "extra", "-json-errors=false"}, true},
		{[]string{"---json-errors"}, false},
		{[]string{"-unknown=-json-errors"}, false},
		{[]string{"-json-errors", "-unknown", "--json-errors=false"}, false},
		{[]string{"-json-errors=false", "-unknown", "--json-errors=true"}, true},
		{[]string{"-json-errors=invalid"}, false},
		{[]string{"-json-errors", "-json-errors=invalid"}, true},
		{[]string{"-json-errors", "false"}, true},
	} {
		if got := scanJSONErrors(tc.args, fs); got != tc.want {
			t.Fatalf("scan %v: got %v, want %v", tc.args, got, tc.want)
		}
	}
	for _, prefix := range []string{"-json-errors", "--json-errors"} {
		if !scanJSONErrors([]string{prefix}, fs) {
			t.Fatal("bare opt-in not recognized", prefix)
		}
		for _, value := range []string{"1", "t", "T", "TRUE", "true", "True"} {
			if !scanJSONErrors([]string{prefix + "=" + value}, fs) {
				t.Fatal("true opt-in not recognized", prefix, value)
			}
		}
		for _, value := range []string{"0", "f", "F", "FALSE", "false", "False"} {
			if scanJSONErrors([]string{"-json-errors", prefix + "=" + value}, fs) {
				t.Fatal("false opt-in not recognized", prefix, value)
			}
		}
	}
}

func diagnosticProcess(binary string, args []string) ([]byte, string, error) {
	var output, diagnostic bytes.Buffer
	cmd := exec.Command(binary, args...)
	cmd.Stdout, cmd.Stderr = &output, &diagnostic
	err := cmd.Run()
	return output.Bytes(), diagnostic.String(), err
}

func requireDiagnosticFailure(t *testing.T, err error, diagnostic, message string) {
	t.Helper()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() == 0 || !strings.Contains(diagnostic, message) || !strings.Contains(diagnostic, "quantaspindle:") {
		t.Fatalf("failure exit=%v stderr=%q; want %q", err, diagnostic, message)
	}
}

func requireStartupRecord(t *testing.T, output []byte, message string) {
	t.Helper()
	records := decodeCLIRecords(t, output)
	if len(records) != 1 || len(records[0]) != 2 || string(records[0]["type"]) != `"error"` {
		t.Fatalf("expected one startup error: %s", output)
	}
	var text string
	if err := json.Unmarshal(records[0]["error"], &text); err != nil || !strings.Contains(text, message) {
		t.Fatalf("error text %q (decode: %v), want %q", text, err, message)
	}
}

func TestCLIStartupJSONErrorsProcess(t *testing.T) {
	binary := buildCLI(t)
	writeConfig := func(name, content string) string {
		path := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	for _, tc := range []struct {
		name, message string
		args          []string
	}{
		{"unknown flag", "flag provided but not defined", []string{"-unknown"}},
		{"bad scalar", "invalid value", []string{"-paths", "bad"}},
		{"bad bool", "invalid boolean value", []string{"-summary-only=bad"}},
		{"bad duration", "invalid value", []string{"-timeout", "later"}},
		{"malformed JSON", "config:", []string{"-config", writeConfig("malformed.json", `{"Paths":`)}},
		{"unknown field", "unknown field", []string{"-config", writeConfig("unknown.json", `{"Unknown":1}`)}},
		{"missing config", "missing.json", []string{"-config", filepath.Join(t.TempDir(), "missing.json")}},
		{"invalid config", "paths must be", []string{"-paths", "0"}},
		{"invalid option", "progress-every must be nonnegative", []string{"-progress-every", "-1"}},
		{"invalid range", "scenario range requires", []string{"-backend", "cpu", "-paths", "8", "-count", "0"}},
		{"invalid backend", "unknown backend", []string{"-backend", "fpga"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []string{"default", "before", "after", "disabled"} {
				args := append([]string(nil), tc.args...)
				switch mode {
				case "before":
					args = append([]string{"-json-errors"}, args...)
				case "after":
					args = append(args, "--json-errors=true")
				case "disabled":
					args = append(args, "-json-errors=false")
				}
				output, diagnostic, err := diagnosticProcess(binary, args)
				requireDiagnosticFailure(t, err, diagnostic, tc.message)
				if mode == "before" || mode == "after" {
					requireStartupRecord(t, output, tc.message)
				} else if len(output) != 0 {
					t.Fatalf("default/disabled mode emitted startup JSON: %s", output)
				}
			}
		})
	}
	for _, args := range [][]string{{"-config", "-json-errors"}, {"-paths", "-json-errors"}, {"-config=-json-errors"}, {"--", "-json-errors"}, {"extra", "-json-errors"}} {
		output, diagnostic, err := diagnosticProcess(binary, args)
		requireDiagnosticFailure(t, err, diagnostic, "")
		if len(output) != 0 {
			t.Fatalf("value or trailing argument enabled JSON errors: %v: %s", args, output)
		}
	}
	for _, help := range []string{"-h", "--help"} {
		output, diagnostic, err := diagnosticProcess(binary, []string{"-json-errors", help})
		if err != nil || len(output) != 0 || !strings.Contains(diagnostic, "Usage of quantaspindle:") {
			t.Fatalf("help changed: %v stdout=%s stderr=%s", err, output, diagnostic)
		}
	}
	for _, validateOnly := range []bool{false, true} {
		args := []string{"-json-errors", "-backend", "cpu", "-paths", "1", "-steps", "1"}
		if validateOnly {
			args = append(args, "-validate-only")
		}
		output, diagnostic, err := diagnosticProcess(binary, args)
		if err != nil || diagnostic != "" {
			t.Fatalf("JSON errors changed successful exit: %v stderr=%q", err, diagnostic)
		}
		records := decodeCLIRecords(t, output)
		if validateOnly {
			if len(records) != 1 || string(records[0]["type"]) != `"validation"` {
				t.Fatalf("validation sequence changed: %s", output)
			}
		} else if len(records) != 3 || string(records[0]["type"]) != `"metadata"` || string(records[1]["type"]) != `"scenario"` || string(records[2]["type"]) != `"summary"` {
			t.Fatalf("successful sequence changed: %s", output)
		}
	}
}

func TestCLIJSONErrorsPreservesSuccessfulRecords(t *testing.T) {
	for _, validation := range []bool{false, true} {
		args := []string{"-backend", "cpu", "-paths", "3", "-steps", "2", "-seed", "42", "-progress-every", "2"}
		if validation {
			args = append(args, "-validate-only")
		}
		var baseline, baselineDiagnostic, output, diagnostic bytes.Buffer
		if err := execute(context.Background(), args, &baseline, &baselineDiagnostic); err != nil {
			t.Fatal(err)
		}
		if err := execute(context.Background(), append(args, "-json-errors"), &output, &diagnostic); err != nil {
			t.Fatal(err)
		}
		want := decodeCLIRecords(t, baseline.Bytes())
		got := decodeCLIRecords(t, output.Bytes())
		if len(got) != len(want) || diagnostic.String() != baselineDiagnostic.String() {
			t.Fatalf("successful sequence changed: %s", &output)
		}
		for i, record := range got {
			for _, field := range []string{"type", "config", "valid", "scenario", "summary", "selection"} {
				if !bytes.Equal(record[field], want[i][field]) {
					t.Fatalf("JSON errors changed record %d field %s", i, field)
				}
			}
		}
	}
}

func requireRuntimeRecord(t *testing.T, output []byte, completed uint64, message string) ([]map[string]json.RawMessage, map[string]json.RawMessage) {
	t.Helper()
	records := decodeCLIRecords(t, output)
	if len(records) < 2 || string(records[0]["type"]) != `"metadata"` {
		t.Fatalf("runtime metadata missing: %s", output)
	}
	errorsFound := 0
	for _, record := range records {
		switch string(record["type"]) {
		case `"summary"`:
			t.Fatal("failed run emitted a successful summary")
		case `"error"`:
			errorsFound++
		}
	}
	last := records[len(records)-1]
	var text string
	var count uint64
	var elapsed int64
	if err := json.Unmarshal(last["error"], &text); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(last["completed"], &count); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(last["elapsed_ns"], &elapsed); err != nil {
		t.Fatal(err)
	}
	if errorsFound != 1 || count != completed || elapsed < 0 || !strings.Contains(text, message) {
		t.Fatalf("incorrect runtime error: %s", output)
	}
	return records, last
}

func TestCLICPURuntimeErrorDetailsProcess(t *testing.T) {
	binary := buildCLI(t)
	for _, backend := range []string{"cpu", "auto"} {
		for _, jsonErrors := range []bool{false, true} {
			args := []string{"-backend", backend, "-paths", "1", "-steps", "1", "-dt", "1e308"}
			if jsonErrors {
				args = append(args, "-json-errors")
			}
			output, diagnostic, err := diagnosticProcess(binary, args)
			requireDiagnosticFailure(t, err, diagnostic, "numerical overflow/underflow")
			records, record := requireRuntimeRecord(t, output, 0, "numerical overflow/underflow")
			if len(records) != 2 || record["cuda"] != nil || string(records[0]["backend"]) != fmt.Sprintf("%q", (qs.CPU{}).Name()) {
				t.Fatalf("CPU error claimed CUDA work: %s", output)
			}
			if !bytes.Equal(records[0]["selection"], record["selection"]) || (backend == "auto" && record["selection"] == nil) {
				t.Fatalf("error disagrees with selected-backend metadata: %s", output)
			}
		}
	}
}

// diagnosticCUDARunner exercises the real stream adapter with one successful
// batch followed by a device failure. No shell or actual GPU is required.
func diagnosticCUDARunner(mode string) int {
	log := func(message string) {
		if path := os.Getenv("QS_CLI_TEST_CUDA_LOG"); path != "" {
			f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
			if err == nil {
				fmt.Fprintln(f, message)
				f.Close()
			}
		}
	}
	log("start")
	defer log("exit")
	if len(os.Args) != 2 || os.Args[1] != "--stream" {
		return 2
	}
	if _, err := os.Stdout.Write([]byte("QSPNCS01")); err != nil {
		return 2
	}
	for batch := 0; ; batch++ {
		var request [72]byte
		if _, err := io.ReadFull(os.Stdin, request[:]); err == io.EOF {
			if mode == "shutdown-failure" {
				fmt.Fprintln(os.Stderr, "synthetic shutdown failure")
				return 7
			}
			return 0
		} else if err != nil {
			return 2
		}
		log(fmt.Sprintf("request %d %d", binary.LittleEndian.Uint64(request[24:]), binary.LittleEndian.Uint64(request[16:])))
		if mode == "unavailable" {
			fmt.Fprintln(os.Stderr, "quantaspindle-cuda: CUDA unavailable: synthetic unavailable device")
			return 7
		}
		if mode == "fail-first" || (mode == "fail-after-one" && batch > 0) {
			fmt.Fprintln(os.Stderr, "synthetic device failure")
			return 7
		}
		count := int(binary.LittleEndian.Uint64(request[16:]))
		first := binary.LittleEndian.Uint64(request[24:])
		response := make([]byte, 176+32*count)
		copy(response, "QSPNCR01")
		binary.LittleEndian.PutUint32(response[8:], 1)
		binary.LittleEndian.PutUint64(response[16:], uint64(count))
		binary.LittleEndian.PutUint64(response[24:], math.Float64bits(1.25))
		binary.LittleEndian.PutUint64(response[32:], math.Float64bits(2.5))
		copy(response[48:], "diagnostics-fixture")
		for i := 0; i < count; i++ {
			price := 100 + float64(first+uint64(i))
			for j, value := range []float64{price, price/100 - 1, .2, .1} {
				binary.LittleEndian.PutUint64(response[176+i*32+j*8:], math.Float64bits(value))
			}
		}
		if mode == "protocol-error" {
			response[0] = 'x'
		}
		if _, err := os.Stdout.Write(response); err != nil {
			return 2
		}
	}
}

func TestCLICUDAFailureStatisticsProcess(t *testing.T) {
	binary := buildCLI(t)
	runner, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, backend := range []string{"cuda", "auto"} {
		t.Run(backend, func(t *testing.T) {
			t.Setenv("QS_CLI_TEST_CUDA_MODE", "fail-after-one")
			paths, steps := 4, 2
			if backend == "auto" {
				paths, steps = hybrid.ColdMinPaths, hybrid.MinSteps
			}
			args := []string{"-json-errors", "-backend", backend, "-cuda-runner", runner, "-paths", fmt.Sprint(paths), "-steps", fmt.Sprint(steps), "-workers", "1", "-batch", "2"}
			output, diagnostic, err := diagnosticProcess(binary, args)
			requireDiagnosticFailure(t, err, diagnostic, "CUDA execution failed")
			records, record := requireRuntimeRecord(t, output, 2, "CUDA execution failed")
			if len(records) != 4 || string(records[0]["backend"]) != `"cuda"` {
				t.Fatalf("CUDA delivered prefix missing: %s", output)
			}
			var stats cuda.Statistics
			if err := json.Unmarshal(record["cuda"], &stats); err != nil {
				t.Fatal(err)
			}
			if stats.Batches != 1 || stats.Scenarios != 2 || stats.KernelTime != 1250*time.Microsecond || stats.HostTime != 2500*time.Microsecond || stats.WallTime < 0 || stats.SessionLifetime <= 0 || stats.Last.Device != "diagnostics-fixture" {
				t.Fatalf("failed batch counted as successful or stats missing: %+v", stats)
			}
			if !bytes.Equal(records[0]["selection"], record["selection"]) || (backend == "auto" && record["selection"] == nil) {
				t.Fatalf("CUDA selection disagrees with metadata: %s", output)
			}
		})
	}
	t.Run("selection failure", func(t *testing.T) {
		t.Setenv("QS_CLI_TEST_CUDA_MODE", "fail-first")
		args := []string{"-backend", "auto", "-cuda-runner", runner, "-paths", fmt.Sprint(hybrid.ColdMinPaths), "-steps", fmt.Sprint(hybrid.MinSteps), "-workers", "1", "-batch", "2"}
		for _, jsonErrors := range []bool{false, true} {
			modeArgs := append([]string(nil), args...)
			if jsonErrors {
				modeArgs = append(modeArgs, "-json-errors")
			}
			output, diagnostic, err := diagnosticProcess(binary, modeArgs)
			requireDiagnosticFailure(t, err, diagnostic, "CUDA execution failed")
			if jsonErrors {
				requireStartupRecord(t, output, "CUDA execution failed")
			} else if len(output) != 0 {
				t.Fatalf("selection failure changed default stdout: %s", output)
			}
		}
	})
	t.Run("successful CUDA summary", func(t *testing.T) {
		t.Setenv("QS_CLI_TEST_CUDA_MODE", "valid")
		args := []string{"-json-errors", "-backend", "cuda", "-cuda-runner", runner, "-paths", "2", "-steps", "2", "-workers", "1", "-batch", "2"}
		output, diagnostic, err := diagnosticProcess(binary, args)
		if err != nil || diagnostic != "" {
			t.Fatalf("successful CUDA exit changed: %v stderr=%q", err, diagnostic)
		}
		records := decodeCLIRecords(t, output)
		if len(records) != 4 || string(records[3]["type"]) != `"summary"` || records[3]["error"] != nil || records[3]["completed"] != nil {
			t.Fatalf("successful CUDA sequence changed: %s", output)
		}
		var stats cuda.Statistics
		if err := json.Unmarshal(records[3]["cuda"], &stats); err != nil || stats.Batches != 1 || stats.Scenarios != 2 || stats.KernelTime != 1250*time.Microsecond || stats.SessionLifetime <= 0 {
			t.Fatalf("successful CUDA statistics changed: %+v, %v", stats, err)
		}
	})
}

type diagnosticWriteFunc func([]byte) (int, error)

func (write diagnosticWriteFunc) Write(data []byte) (int, error) { return write(data) }

func TestCLIJSONErrorsOutputFailure(t *testing.T) {
	for _, startup := range []bool{false, true} {
		writeErr := errors.New("stdout disconnected")
		writes := 0
		output := diagnosticWriteFunc(func([]byte) (int, error) {
			writes++
			return 0, writeErr
		})
		args := []string{"-json-errors", "-backend", "cpu", "-paths", "1", "-steps", "1"}
		if startup {
			args = append(args, "-paths", "0")
		}
		err := execute(context.Background(), args, output, io.Discard)
		if !errors.Is(err, writeErr) || writes != 1 {
			t.Fatalf("output failure triggered duplicate error write: startup=%v writes=%d err=%v", startup, writes, err)
		}
		if startup && !strings.Contains(err.Error(), "paths must be") {
			t.Fatal("output failure lost the original validation error", err)
		}
	}
}
