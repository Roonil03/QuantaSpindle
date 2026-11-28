package quantaspindle_test

import (
	"fmt"

	qs "github.com/Roonil03/QuantaSpindle"
)

func ExampleLossHistogram() {
	h, err := qs.NewLossHistogram(-2, 3, 5)
	if err != nil {
		panic(err)
	}
	// In a Run sink, call Add for each computed portfolio loss instead of
	// retaining the losses. Out-of-range observations return an error.
	for _, loss := range []float64{-2, 0, 3} {
		if err := h.Add(loss); err != nil {
			panic(err)
		}
	}
	r, err := h.Risk(.5)
	if err != nil {
		panic(err)
	}
	fmt.Println(r.Count, r.TailCount)
	fmt.Println(r.VaR.Lower, r.VaR.Upper)
	fmt.Println(r.ExpectedShortfall.Lower <= 1.5 && r.ExpectedShortfall.Upper >= 1.5)
	// Output:
	// 3 2
	// 0 1
	// true
}
