package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strconv"
	"time"

	"github.com/Roonil03/QuantaSpindle/cuda"
)

type buildIdentity struct {
	Available     bool   `json:"available"`
	ModulePath    string `json:"module_path,omitempty"`
	ModuleVersion string `json:"module_version,omitempty"`
	VCSRevision   string `json:"vcs_revision,omitempty"`
	VCSModified   *bool  `json:"vcs_modified,omitempty"`
}

func identifyBuild(info *debug.BuildInfo, ok bool) buildIdentity {
	if !ok || info == nil {
		return buildIdentity{}
	}
	identity := buildIdentity{Available: true, ModulePath: info.Main.Path, ModuleVersion: info.Main.Version}
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			identity.VCSRevision = setting.Value
		case "vcs.modified":
			if modified, err := strconv.ParseBool(setting.Value); err == nil {
				identity.VCSModified = &modified
			}
		}
	}
	return identity
}

type runnerIdentity struct {
	Available bool   `json:"available"`
	Path      string `json:"path,omitempty"`
	SHA256    string `json:"sha256,omitempty"`
	Error     string `json:"error,omitempty"`
}

func identifyRunner(runner string) runnerIdentity {
	path, err := cuda.ResolveRunner(runner)
	if err != nil {
		return runnerIdentity{Error: err.Error()}
	}
	return hashRunner(path, os.Open)
}

func hashRunner(path string, open func(string) (*os.File, error)) runnerIdentity {
	identity := runnerIdentity{Path: path}
	f, err := open(path)
	if err != nil {
		identity.Error = err.Error()
		return identity
	}
	hash := sha256.New()
	_, err = io.Copy(hash, f)
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		identity.Error = err.Error()
		return identity
	}
	identity.Available = true
	identity.SHA256 = fmt.Sprintf("%x", hash.Sum(nil))
	return identity
}

type benchmarkDependencies struct {
	build  func() (*debug.BuildInfo, bool)
	runner func(string) runnerIdentity
	now    func() time.Time
}

func defaultBenchmarkDependencies() benchmarkDependencies {
	return benchmarkDependencies{debug.ReadBuildInfo, identifyRunner, time.Now}
}
