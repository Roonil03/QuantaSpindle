param(
    [string]$Output = "$PSScriptRoot/../local/quantaspindle-cuda.exe",
    [string]$Nvcc = "nvcc",
    [string]$Architecture = "sm_86"
)
$ErrorActionPreference = 'Stop'
$compiler = (Get-Command $Nvcc -ErrorAction Stop).Source
$target = [IO.Path]::GetFullPath($Output)
$source = [IO.Path]::GetFullPath("$PSScriptRoot/../accelerators/cuda/runner.cu")
New-Item -ItemType Directory -Force ([IO.Path]::GetDirectoryName($target)) | Out-Null
# Run in a Developer PowerShell/Command Prompt, or initialize vcvars64.bat first.
& $compiler -std=c++17 -O3 --fmad=false "-arch=$Architecture" -cudart=static $source -o $target
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Output $target
