$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $PackageDir
$Python = Join-Path $RepoRoot '.venv-infer\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python (Join-Path $ScriptDir 'extract_metrics.py')
& $Python (Join-Path $ScriptDir 'generate_figures.py')
& $Python (Join-Path $ScriptDir 'generate_diagrams.py')
& $Python (Join-Path $ScriptDir 'validate_package.py')

$Latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($null -eq $Latexmk) {
    Write-Warning 'latexmk is not installed. Generated and validated assets; skipping PDF build.'
    exit 0
}

Push-Location (Join-Path $PackageDir 'paper')
try {
    & $Latexmk.Source -pdf -interaction=nonstopmode -halt-on-error main.tex
}
finally {
    Pop-Location
}
