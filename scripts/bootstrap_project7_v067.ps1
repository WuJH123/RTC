param(
  [string]$Base = "E:\RTC_sewer\Project7",
  [string]$SourceInp = "E:\RTC_sewer\Project7\source\wuhan_with_controls.inp",
  [string]$Sensors = "E:\RTC_sewer\Project7\source\sensor_nodes.txt",
  [string]$Priority = "E:\RTC_sewer\Project7\source\priority_nodes.txt",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$Repo = Join-Path $Base "repo"
$Inputs = Join-Path $Base "inputs"
$Study = Join-Path $Base "study_v067"
$Logs = Join-Path $Base "logs"

New-Item -ItemType Directory -Force $Base,$Logs | Out-Null
foreach ($p in @($SourceInp,$Sensors,$Priority)) {
  if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing source-only asset: $p" }
}

if (-not (Test-Path -LiteralPath (Join-Path $Repo ".git"))) {
  git clone https://github.com/WuJH123/RTC.git $Repo
}
Set-Location $Repo
if ((git status --porcelain).Length -gt 0) {
  throw "Project7\repo has local modifications. Resolve them before fresh v0.6.7 bootstrap."
}
git fetch origin
git checkout $Branch
git pull --ff-only origin $Branch

python -m pip install -e ".[dev,swmm]"
python -m pytest -q
python -c "import importlib.metadata; print('wuhan-rtc', importlib.metadata.version('wuhan-rtc'))"

foreach ($fresh in @($Inputs,$Study)) {
  if (Test-Path -LiteralPath $fresh) {
    $items = @(Get-ChildItem -LiteralPath $fresh -Force)
    if ($items.Count -gt 0) {
      throw "$fresh is not empty. v0.6.7 refuses to mix historical inputs/study artifacts."
    }
  } else {
    New-Item -ItemType Directory -Force $fresh | Out-Null
  }
}

rtc-build-method-testbed-v067 `
  --source-inp $SourceInp `
  --sensors $Sensors `
  --priority $Priority `
  --out-root $Inputs `
  --warmup-minutes 60 `
  --recession-minutes 360 `
  --orifice-travel-minutes 10

Copy-Item -LiteralPath (Join-Path $Repo "configs\sensor_layout_provenance.project7.v1.json") `
  -Destination (Join-Path $Inputs "contracts\sensor_layout_provenance.project7.v1.json") -Force

$Network = Join-Path $Inputs "network\wuhan_method_testbed_v067.inp"
$Events = Join-Path $Inputs "contracts\events_with_splits.csv"
$RainProv = Join-Path $Inputs "contracts\rainfall_provenance.v067.json"
$ActScope = Join-Path $Inputs "contracts\actuator_scope.v067.json"
$SensorCopy = Join-Path $Inputs "contracts\sensor_nodes.txt"
$PriorityCopy = Join-Path $Inputs "contracts\priority_nodes.txt"
$SensorProv = Join-Path $Inputs "contracts\sensor_layout_provenance.project7.v1.json"

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

Write-Host "v0.6.7 fresh Step 0 complete. Do not reuse historical D0/D1/D2/D3 or models."
Write-Host "Repo:   $Repo"
Write-Host "Inputs: $Inputs"
Write-Host "Study:  $Study"
