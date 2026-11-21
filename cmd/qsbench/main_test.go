package main

import (
	"encoding/json"
	"flag"
	"io"
	"os"
	"reflect"
	"testing"

	qs "github.com/Roonil03/QuantaSpindle"
)

func TestBuildCases(t *testing.T) {
	type identity struct {
		mode                  string
		paths, batch, workers int
	}
	for _, tt := range []struct {
		name       string
		modes      string
		paths      []int
		workers    []int
		cpuBatch   int
		cpuBatches []int
		want       []identity
	}{
		{
			name: "CPU override", modes: "cpu", paths: []int{4096}, workers: []int{1}, cpuBatch: 64,
			want: []identity{{"cpu", 4096, 64, 1}},
		},
		{
			name: "mixed backends preserve first occurrence", modes: "cpu,cuda,cuda-batch", paths: []int{4096}, workers: []int{1}, cpuBatch: 64,
			want: []identity{{"cpu", 4096, 64, 1}, {"cuda", 4096, 128, 1}, {"cuda-batch", 4096, 128, 1}, {"cuda", 4096, 256, 1}, {"cuda-batch", 4096, 256, 1}},
		},
		{
			name: "CPU sweep overrides scalar", modes: "cpu", paths: []int{4096}, workers: []int{1, 2}, cpuBatch: 32, cpuBatches: []int{64, 128, 64},
			want: []identity{{"cpu", 4096, 64, 1}, {"cpu", 4096, 128, 1}, {"cpu", 4096, 64, 2}, {"cpu", 4096, 128, 2}},
		},
		{
			name: "implicit paths stay distinct", modes: "cpu", workers: []int{1}, cpuBatch: 64,
			want: []identity{{"cpu", 128, 64, 1}, {"cpu", 256, 64, 1}},
		},
		{
			name: "batch sizes stay distinct without override", modes: "cpu", paths: []int{4096}, workers: []int{1},
			want: []identity{{"cpu", 4096, 128, 1}, {"cpu", 4096, 256, 1}},
		},
		{
			name: "repeated input values", modes: "cpu,cpu", paths: []int{4096, 4096, 2048}, workers: []int{1, 1}, cpuBatch: 64,
			want: []identity{{"cpu", 4096, 64, 1}, {"cpu", 2048, 64, 1}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cases, err := buildCases(tt.modes, []int{128, 256}, tt.paths, tt.workers, tt.cpuBatch, tt.cpuBatches, 64)
			if err != nil {
				t.Fatal(err)
			}
			var got []identity
			for _, item := range cases {
				cfg := item.config
				got = append(got, identity{item.mode, cfg.Paths, cfg.BatchSize, cfg.Workers})
				wantConfig := qs.DefaultConfig()
				wantConfig.Paths, wantConfig.BatchSize, wantConfig.Workers, wantConfig.Steps = cfg.Paths, cfg.BatchSize, cfg.Workers, 64
				if !reflect.DeepEqual(cfg, wantConfig) {
					t.Errorf("case config changed defaults: %+v", cfg)
				}
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("cases = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRunDeduplicatesCPUOverrideAndReversesTrialOrder(t *testing.T) {
	for _, tt := range []struct {
		name    string
		workers string
		want    [][]int
	}{
		{"issue reproduction", "1", [][]int{{1}, {1}}},
		{"alternate trial order", "1,2", [][]int{{1, 2}, {2, 1}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			testRunCPUOverride(t, tt.workers, tt.want)
		})
	}
}

func testRunCPUOverride(t *testing.T, workers string, wantWorkers [][]int) {
	oldFlags, oldArgs, oldStdout := flag.CommandLine, os.Args, os.Stdout
	t.Cleanup(func() { flag.CommandLine, os.Args, os.Stdout = oldFlags, oldArgs, oldStdout })
	flag.CommandLine = flag.NewFlagSet("qsbench", flag.ContinueOnError)
	os.Args = []string{"qsbench", "-backend", "cpu", "-paths", "4096", "-steps", "64", "-workers", workers, "-batches", "128,256", "-cpu-batch", "64", "-trials", "2"}
	output, err := os.CreateTemp(t.TempDir(), "records-*.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	os.Stdout = output
	if err := run(); err != nil {
		t.Fatal(err)
	}
	if _, err := output.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(output)
	for trial, expectedWorkers := range wantWorkers {
		for order, worker := range expectedWorkers {
			var record struct {
				Mode         string
				Trial, Order int
				Config       qs.Config
			}
			if err := decoder.Decode(&record); err != nil {
				t.Fatal(err)
			}
			if record.Mode != "cpu" || record.Trial != trial+1 || record.Order != order+1 || record.Config.Seed != 1 || record.Config.BatchSize != 64 || record.Config.Paths != 4096 || record.Config.Steps != 64 || record.Config.Workers != worker {
				t.Fatalf("unexpected measured row: %+v", record)
			}
		}
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatalf("expected one row per trial and no warmup rows, got %v, %v", extra, err)
	}
}
