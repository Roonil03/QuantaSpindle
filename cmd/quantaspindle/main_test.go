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
