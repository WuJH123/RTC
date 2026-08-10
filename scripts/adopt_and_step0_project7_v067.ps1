param(
  [string]$Base = "E:\RTC_sewer\Project7",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$Repo = Join-Path $Base "repo"
$Inputs = Join-Path $Base "inputs"
$Study = Join-Path $Base "study_v067"
$Logs = Join-Path $Base "logs"

New-Item -ItemType Directory -Force $Base,$Logs | Out-Null
if (-not (Test-Path -LiteralPath $Inputs -PathType Container)) {
  throw "Expected extracted v0.6.7 bundle under $Inputs"
}

if (-not (Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
  git clone https://github.com/WuJH123/RTC.git $Repo
}
Set-Location $Repo
if ((git status --porcelain).Length -gt 0) {
  throw "Project7\repo has local modifications. Resolve them before syncing authoritative main."
}
git fetch origin
git checkout $Branch
git pull --ff-only origin $Branch

python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print('wuhan-rtc', importlib.metadata.version('wuhan-rtc'))"

$Adoption = Join-Path $Logs "input_adoption_v067.json"
rtc-adopt-method-testbed-v067 --input-root $Inputs --out $Adoption
$Adopt = Get-Content -LiteralPath $Adoption -Raw | ConvertFrom-Json
if (-not $Adopt.passed) { throw "v0.6.7 input adoption did not pass" }
if ($Adopt.events_verified -ne 30 -or $Adopt.rainfall_files_verified -ne 30) {
  throw "Expected 30 verified event INPs and 30 verified rainfall files"
}
$InputRoot = [string]$Adopt.resolved_input_root

if (Test-Path -LiteralPath $Study) {
  if (@(Get-ChildItem -LiteralPath $Study -Force).Count -gt 0) {
    throw "$Study is not empty. Do not mix prior study artifacts into a fresh v0.6.7 run."
  }
} else {
  New-Item -ItemType Directory -Force $Study | Out-Null
}

$Network = Join-Path $InputRoot "network\wuhan_method_testbed_v067.inp"
$Events = Join-Path $InputRoot "contracts\events_with_splits.csv"
$RainProv = Join-Path $InputRoot "contracts\rainfall_provenance.v067.json"
$ActScope = Join-Path $InputRoot "contracts\actuator_scope.v067.json"
$SensorCopy = Join-Path $InputRoot "contracts\sensor_nodes.txt"
$PriorityCopy = Join-Path $InputRoot "contracts\priority_nodes.txt"
$SensorProv = Join-Path $InputRoot "contracts\sensor_layout_provenance.project7.v1.json"

Copy-Item -LiteralPath (Join-Path $Repo "configs\sensor_layout_provenance.project7.v1.json") `
  -Destination $SensorProv -Force

rtc-validate-rainfall-design --events $Events --out (Join-Path $Logs "rainfall_design_v067.json")

rtc-init-fresh-workspace `
  --root $Study `
  --inp $Network `
  --priority $PriorityCopy `
  --events $Events

rtc-inp-audit-v2 `
  --inp $Network `
  --priority $PriorityCopy `
  --out (Join-Path $Study "audit\inp_preflight.json")

rtc-compile-formal-assets `
  --inp $Network `
  --priority $PriorityCopy `
  --sensors $SensorCopy `
  --out-dir (Join-Path $Study "formal_assets")

rtc-check-study-readiness `
  --events $Events `
  --frozen-inp $Network `
  --sensors $SensorCopy `
  --sensor-provenance $SensorProv `
  --rainfall-provenance $RainProv `
  --actuator-scope $ActScope `
  --history-span-minutes 60 `
  --minimum-post-rain-tail-minutes 360 `
  --out (Join-Path $Logs "study_readiness_v067.json")

Write-Host "v0.6.7 extracted bundle adopted and Step 0 completed."
Write-Host "Repo:      $Repo"
Write-Host "InputRoot: $InputRoot"
Write-Host "Study:     $Study"
Write-Host "Adoption:  $Adoption"
