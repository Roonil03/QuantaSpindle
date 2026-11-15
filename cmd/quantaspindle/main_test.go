package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
)

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
