package cuda

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// ResolveRunner uses the execution lookup rules and returns an absolute path.
// An empty runner selects QS_CUDA_RUNNER, then QUANTASPINDLE_CUDA_RUNNER, then
// quantaspindle-cuda. Explicit executable names are also searched on PATH.
// Resolving does not start, read, or establish trust in the executable. Callers
// retaining the path must keep the executable and symlink targets unchanged.
func ResolveRunner(runner string) (string, error) {
	if runner == "" {
		runner = os.Getenv("QS_CUDA_RUNNER")
	}
	if runner == "" {
		runner = os.Getenv("QUANTASPINDLE_CUDA_RUNNER")
	}
	if runner == "" {
		runner = "quantaspindle-cuda"
	}
	resolved, err := exec.LookPath(runner)
	if err == nil {
		resolved, err = filepath.Abs(resolved)
	}
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	return resolved, nil
}
