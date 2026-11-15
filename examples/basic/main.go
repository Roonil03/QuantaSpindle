// Command basic demonstrates bounded CPU streaming with explicit tail retention.
package main

import (
	"context"
	"fmt"
	"log"

	qs "github.com/Roonil03/QuantaSpindle"
)

func main() {
	cfg := qs.DefaultConfig()
	// Retain only the returns needed for exact empirical tail estimates.
	returns := make([]float64, 0, cfg.Paths)
	summary, err := qs.Run(context.Background(), cfg, qs.CPU{}, func(s qs.Scenario) error { returns = append(returns, s.Returns[0]); return nil })
	if err != nil {
		log.Fatal(err)
	}
	risk, err := qs.RiskFromReturns(returns, 0.95)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("scenarios=%d mean-return=%.6f mean-realized-volatility=%.6f VaR95=%.6f ES95=%.6f\n", summary.Scenarios, summary.MeanReturns[0], summary.MeanRealizedVolatility[0], risk.VaR, risk.ExpectedShortfall)
}
