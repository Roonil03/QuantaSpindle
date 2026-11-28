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
	"sort"
	"strings"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
)

func scenariosFromRecords(t *testing.T, records []map[string]json.RawMessage) []qs.Scenario {
	t.Helper()
	var scenarios []qs.Scenario
	for _, record := range records {
		if string(record["type"]) == `"scenario"` {
			var scenario qs.Scenario
			if err := json.Unmarshal(record["scenario"], &scenario); err != nil {
				t.Fatal(err)
			}
			scenarios = append(scenarios, scenario)
		}
	}
	return scenarios
}

func checkRangeRecord(t *testing.T, record map[string]json.RawMessage, startID, count uint64) {
	t.Helper()
	var bounds scenarioRange
	if err := json.Unmarshal(record["range"], &bounds); err != nil || bounds.StartID != startID || bounds.Count != count {
		t.Fatalf("range provenance: %s (decode: %v)", record["range"], err)
	}
}

func TestCLIRangesReplayFullRunAndPreserveDefaultSchema(t *testing.T) {
	args := []string{"-backend", "cpu", "-paths", "8", "-steps", "12", "-seed", "42", "-store-paths"}
	var output bytes.Buffer
	if err := execute(context.Background(), args, &output, io.Discard); err != nil {
		t.Fatal(err)
	}
	fullRecords := decodeCLIRecords(t, output.Bytes())
	full := scenariosFromRecords(t, fullRecords)
	var keys []string
	for key := range fullRecords[0] {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	wantKeys := []string{"arch", "backend", "config", "go", "gomaxprocs", "os", "precision", "requested_backend", "started_utc", "type", "version"}
	if !reflect.DeepEqual(keys, wantKeys) {
		t.Fatalf("default metadata schema changed: %v", keys)
	}
	for _, record := range fullRecords {
		if _, ok := record["range"]; ok {
			t.Fatal("default run acquired range metadata")
		}
		if _, ok := record["scope"]; ok {
			t.Fatal("default run acquired range scope")
		}
	}
	for _, setting := range [][2]int{{1, 1}, {2, 2}, {3, 5}, {8, 8}} {
		var joined []qs.Scenario
		for _, bounds := range [][2]uint64{{0, 3}, {3, 5}} {
			var ranged bytes.Buffer
			selectedArgs := append(append([]string(nil), args...), "-start-id", fmt.Sprint(bounds[0]), "-count", fmt.Sprint(bounds[1]), "-workers", fmt.Sprint(setting[0]), "-batch", fmt.Sprint(setting[1]))
			if err := execute(context.Background(), selectedArgs, &ranged, io.Discard); err != nil {
				t.Fatal(err)
			}
			records := decodeCLIRecords(t, ranged.Bytes())
			checkRangeRecord(t, records[0], bounds[0], bounds[1])
			last := records[len(records)-1]
			checkRangeRecord(t, last, bounds[0], bounds[1])
			if string(last["scope"]) != `"range"` {
				t.Fatal("summary does not identify range scope")
			}
			var summary qs.RunSummary
			if err := json.Unmarshal(last["summary"], &summary); err != nil || summary.Scenarios != bounds[1] {
				t.Fatalf("range summary: %+v, %v", summary, err)
			}
			var config qs.Config
			if err := json.Unmarshal(records[0]["config"], &config); err != nil || config.Paths != 8 {
				t.Fatalf("total ID space lost: %+v, %v", config, err)
			}
			joined = append(joined, scenariosFromRecords(t, records)...)
		}
		if !reflect.DeepEqual(joined, full) {
			t.Fatalf("ranged scenarios differ with workers=%d batch=%d", setting[0], setting[1])
		}
	}
}

func TestCLIRangeDefaultsAndValidationOnly(t *testing.T) {
	for _, tc := range []struct {
		args         []string
		start, count uint64
	}{{[]string{"-start-id", "3"}, 3, 5}, {[]string{"-count", "3"}, 0, 3}, {[]string{"-start-id", "0"}, 0, 8}} {
		args := append([]string{"-backend", "cpu", "-paths", "8", "-steps", "2"}, tc.args...)
		var normal, validation, diagnostic bytes.Buffer
		if err := execute(context.Background(), args, &normal, io.Discard); err != nil {
			t.Fatal(err)
		}
		if err := execute(context.Background(), append(args, "-validate-only", "-progress-every", "1"), &validation, &diagnostic); err != nil {
			t.Fatal(err)
		}
		records := decodeCLIRecords(t, validation.Bytes())
		if len(records) != 1 || string(records[0]["type"]) != `"validation"` || string(records[0]["valid"]) != "true" || diagnostic.Len() != 0 {
			t.Fatalf("validation emitted run output: %s; %s", &validation, &diagnostic)
		}
		checkRangeRecord(t, records[0], tc.start, tc.count)
		runRecords := decodeCLIRecords(t, normal.Bytes())
		if !bytes.Equal(runRecords[0]["config"], records[0]["config"]) {
			t.Fatal("validation config differs from normal range config")
		}
		scenarios := scenariosFromRecords(t, runRecords)
		if len(scenarios) != int(tc.count) || scenarios[0].ID != tc.start || scenarios[len(scenarios)-1].ID != tc.start+tc.count-1 {
			t.Fatalf("range defaults changed: %+v", scenarios)
		}
	}
}

func TestCLIRangeProgressCountsSelectedScenarios(t *testing.T) {
	for _, summaryOnly := range []bool{false, true} {
		args := []string{"-backend", "cpu", "-paths", "8", "-steps", "2", "-start-id", "3", "-count", "5", "-progress-every", "2"}
		if summaryOnly {
			args = append(args, "-summary-only")
		}
		var output, diagnostic bytes.Buffer
		if err := execute(context.Background(), args, &output, &diagnostic); err != nil {
			t.Fatal(err)
		}
		want := "quantaspindle: progress delivered=2\nquantaspindle: progress delivered=4\nquantaspindle: progress delivered=5\n"
		if diagnostic.String() != want {
			t.Fatalf("progress used absolute IDs: %q", diagnostic.String())
		}
		records := decodeCLIRecords(t, output.Bytes())
		checkRangeRecord(t, records[len(records)-1], 3, 5)
		if summaryOnly && len(records) != 2 {
			t.Fatal("summary-only emitted scenarios")
		}
	}
}

func TestCLIRangeCancellationPrefix(t *testing.T) {
	for _, summaryOnly := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		args := []string{"-backend", "cpu", "-paths", "8", "-start-id", "3", "-count", "5", "-workers", "1", "-batch", "1", "-progress-every", "2"}
		var output, diagnostic bytes.Buffer
		var out, diag io.Writer = &output, &diagnostic
		wantPrefix := "3"
		if summaryOnly {
			args = append(args, "-summary-only", "-steps", "2")
			diag = &cancelOnWrite{match: []byte("delivered=2"), cancel: cancel}
			wantPrefix = "2"
		} else {
			args = append(args, "-store-paths", "-steps", "2000")
			out = &cancelOnWrite{match: []byte(`"type":"scenario","scenario":{"ID":5,`), cancel: cancel}
		}
		err := execute(ctx, args, out, diag)
		cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("range cancellation: %v", err)
		}
		if w, ok := out.(*cancelOnWrite); ok {
			output = w.Buffer
		}
		if w, ok := diag.(*cancelOnWrite); ok {
			diagnostic = w.Buffer
		}
		records := decodeCLIRecords(t, output.Bytes())
		last := records[len(records)-1]
		if string(last["type"]) != `"error"` || string(last["completed"]) != wantPrefix || bytes.Contains(output.Bytes(), []byte(`"type":"summary"`)) {
			t.Fatalf("incorrect range prefix output: %s", &output)
		}
		wantProgress := "quantaspindle: progress delivered=2\n"
		if !summaryOnly {
			wantProgress += "quantaspindle: progress delivered=3\n"
		}
		if diagnostic.String() != wantProgress {
			t.Fatalf("incorrect range prefix progress: %q", diagnostic.String())
		}
	}
}

func TestCLIRangeRejectsInvalidRequests(t *testing.T) {
	for _, tc := range []struct {
		args    []string
		message string
	}{
		{[]string{"-start-id", "8", "-count", "1"}, "scenario range requires"},
		{[]string{"-start-id", "9"}, "scenario range requires"},
		{[]string{"-start-id", "3", "-count", "0"}, "scenario range requires"},
		{[]string{"-start-id", "7", "-count", "2"}, "scenario range requires"},
		{[]string{"-start-id", "18446744073709551615", "-count", "1"}, "scenario range requires"},
		{[]string{"-start-id", "1", "-count", "18446744073709551615"}, "scenario range requires"},
		{[]string{"-start-id", "-1"}, "invalid value"},
		{[]string{"-count", "-1"}, "invalid value"},
		{[]string{"-start-id", "0", "-backend", "auto"}, "require -backend cpu"},
	} {
		for _, validateOnly := range []bool{false, true} {
			args := append([]string{"-backend", "cpu", "-paths", "8", "-steps", "2"}, tc.args...)
			if validateOnly {
				args = append(args, "-validate-only")
			}
			var output bytes.Buffer
			err := execute(context.Background(), args, &output, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tc.message) || output.Len() != 0 {
				t.Fatalf("accepted range %v (validation=%v): err=%v output=%s", tc.args, validateOnly, err, &output)
			}
		}
	}
	var output bytes.Buffer
	if err := execute(context.Background(), []string{"-paths", "8", "-count", "3"}, &output, io.Discard); err == nil || output.Len() != 0 {
		t.Fatal("range accepted implicit auto backend")
	}
}

func TestCLIRangeResourceValidationUsesCount(t *testing.T) {
	args := []string{"-backend", "cpu", "-paths", "1000000", "-steps", "10000", "-workers", "1024", "-batch", "64", "-store-paths"}
	for _, validateOnly := range []bool{false, true} {
		for _, count := range []string{"1", "4000"} {
			modeArgs := append(append([]string(nil), args...), "-start-id", "500000", "-count", count)
			if validateOnly {
				modeArgs = append(modeArgs, "-validate-only")
			}
			var output bytes.Buffer
			err := execute(context.Background(), modeArgs, &output, io.Discard)
			if count == "4000" {
				if err == nil || !strings.Contains(err.Error(), "512 MiB") || output.Len() != 0 {
					t.Fatalf("oversized range accepted: %v", err)
				}
			} else if err != nil {
				t.Fatalf("small range rejected: %v", err)
			} else {
				records := decodeCLIRecords(t, output.Bytes())
				checkRangeRecord(t, records[0], 500000, 1)
			}
		}
	}
}

func TestCLIRangeProcessIntegration(t *testing.T) {
	binary := buildCLI(t)
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"Paths":12,"Steps":2,"Seed":21,"Workers":1,"BatchSize":2}`), 0600); err != nil {
		t.Fatal(err)
	}
	base := []string{"-backend", "cpu", "-config", path, "-paths", "8", "-seed", "42", "-store-paths"}
	run := func(args []string) ([]map[string]json.RawMessage, string) {
		t.Helper()
		var output, diagnostic bytes.Buffer
		cmd := exec.Command(binary, args...)
		cmd.Stdout, cmd.Stderr = &output, &diagnostic
		if err := cmd.Run(); err != nil {
			t.Fatalf("CLI %v: %v\n%s", args, err, &diagnostic)
		}
		return decodeCLIRecords(t, output.Bytes()), diagnostic.String()
	}
	full, _ := run(base)
	var joined []qs.Scenario
	for _, bounds := range [][2]int{{0, 3}, {3, 5}} {
		args := append(append([]string(nil), base...), "-start-id", fmt.Sprint(bounds[0]), "-count", fmt.Sprint(bounds[1]))
		records, _ := run(args)
		joined = append(joined, scenariosFromRecords(t, records)...)
	}
	if !reflect.DeepEqual(joined, scenariosFromRecords(t, full)) {
		t.Fatal("built CLI split ranges changed IDs, seeds, or paths")
	}
	args := append(append([]string(nil), base...), "-start-id", "3", "-progress-every", "2", "-summary-only")
	records, diagnostic := run(args)
	if len(records) != 2 || diagnostic != "quantaspindle: progress delivered=2\nquantaspindle: progress delivered=4\nquantaspindle: progress delivered=5\n" {
		t.Fatalf("built CLI progress: records=%d stderr=%q", len(records), diagnostic)
	}
	records, diagnostic = run(append(args, "-validate-only"))
	if len(records) != 1 || string(records[0]["type"]) != `"validation"` || diagnostic != "" {
		t.Fatalf("built CLI validation: records=%v stderr=%q", records, diagnostic)
	}
	checkRangeRecord(t, records[0], 3, 5)
	for _, invalid := range [][]string{{"-count", "0"}, {"-start-id", "8", "-count", "1"}, {"-start-id", "1", "-count", "18446744073709551615"}, {"-backend", "auto", "-start-id", "0"}} {
		for _, validateOnly := range []bool{false, true} {
			modeArgs := append(append([]string(nil), base...), invalid...)
			if validateOnly {
				modeArgs = append(modeArgs, "-validate-only")
			}
			output, err := exec.Command(binary, modeArgs...).CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() == 0 || bytes.Contains(output, []byte(`"type":"summary"`)) {
				t.Fatalf("built CLI accepted invalid range %v: err=%v output=%s", modeArgs, err, output)
			}
		}
	}
}
