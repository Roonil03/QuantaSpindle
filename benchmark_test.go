package quantaspindle

import (
	"context"
	"fmt"
	"testing"
)

func BenchmarkRun(b *testing.B) {
	for _, workers := range []int{1, 2, 4} {
		for _, batch := range []int{16, 128} {
			b.Run(fmt.Sprintf("workers=%d/batch=%d", workers, batch), func(b *testing.B) {
				c := DefaultConfig()
				c.Paths = 1024
				c.Steps = 252
				c.Workers = workers
				c.BatchSize = batch
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := Run(context.Background(), c, CPU{}, nil); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.N*c.Paths)/b.Elapsed().Seconds(), "scenarios/s")
				b.ReportMetric(float64(b.N*c.Paths)/b.Elapsed().Seconds(), "paths/s")
			})
		}
	}
}

func BenchmarkStoredPaths(b *testing.B) {
	c := DefaultConfig()
	c.Paths = 128
	c.Steps = 252
	c.Workers = 2
	c.BatchSize = 16
	c.StorePaths = true
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Run(context.Background(), c, CPU{}, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.N*c.Paths)/b.Elapsed().Seconds(), "scenarios/s")
	b.ReportMetric(float64(b.N*c.Paths)/b.Elapsed().Seconds(), "paths/s")
}
