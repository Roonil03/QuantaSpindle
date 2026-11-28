package quantaspindle_test

import (
	"context"
	"fmt"

	qs "github.com/Roonil03/QuantaSpindle"
)

func ExampleRunRangeWithState() {
	c := qs.DefaultConfig()
	c.Paths, c.Steps = 7, 1
	c.Assets[0].Mu, c.Assets[0].Sigma = 0, 0
	// Every worker must agree on the configuration/replay identity.
	scope := "example-job/cpu/toolchain-v1"
	merged, err := qs.NewRunSummaryState(c, (qs.CPU{}).Name(), scope)
	if err != nil {
		panic(err)
	}
	for _, bounds := range [][2]uint64{{0, 2}, {2, 5}} {
		_, shard, err := qs.RunRangeWithState(context.Background(), c, qs.CPU{}, bounds[0], bounds[1], scope, nil)
		if err != nil {
			panic(err)
		}
		payload, err := shard.MarshalBinary()
		if err != nil {
			panic(err)
		}
		var received qs.RunSummaryState
		if err := received.UnmarshalBinary(payload); err != nil {
			panic(err)
		}
		if err := merged.Merge(&received); err != nil {
			panic(err)
		}
	}
	summary, err := merged.Summary()
	if err != nil {
		panic(err)
	}
	fmt.Println(summary.Scenarios, summary.MeanReturns)
	fmt.Println(merged.Ranges())
	// Output:
	// 7 [0]
	// [{0 7}]
}
