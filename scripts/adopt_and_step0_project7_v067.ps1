param(
  [string]$Base = "E:\RTC_sewer\Project7",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$Repo = Join-Path $Base "repo"
$Inputs = Join-Path $Base "inputs"
$Prepared = Join-Path $Base "prepared_v069"
$PreparedEvents = Join-Path $Prepared "events"
$PreparedRegistry = Join-Path $Prepared "events_with_splits.prepared120.csv"
$Study = Join-Path $Base "study_v069"
$Logs = Join-Path $Base "logs"

New-Item -ItemType Directory -Force $Base,$Logs | Out-Null
if (-not (Test-Path -LiteralPath $Inputs -PathType Container)) {
  throw "Expected extracted v0.6.7 physical/rainfall bundle under $Inputs"
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

$Adoption = Join-Path $Logs "input_adoption_v069.json"
rtc-adopt-method-testbed-v067 --input-root $Inputs --out $Adoption
$Adopt = Get-Content -LiteralPath $Adoption -Raw | ConvertFrom-Json
if (-not $Adopt.passed) { throw "Project7 v0.6.9 input adoption did not pass" }
if ($Adopt.events_verified -ne 30 -or $Adopt.rainfall_files_verified -ne 30) {
  throw "Expected 30 verified event INPs and 30 verified rainfall files"
}
if ($Adopt.scientific_split_counts.development -ne 24 -or $Adopt.scientific_split_counts.final -ne 6) {
  throw "Expected frozen top-level split development=24/final=6"
}
if ($Adopt.development_fold_counts.train -ne 18 -or $Adopt.development_fold_counts.validation -ne 6) {
  throw "Expected frozen development split Train=18/Validation=6"
}
$InputRoot = [string]$Adopt.resolved_input_root

if (Test-Path -LiteralPath $Study) {
  if (@(Get-ChildItem -LiteralPath $Study -Force).Count -gt 0) {
    throw "$Study is not empty. Do not mix prior study artifacts into the fresh v0.6.9 execution."
  }
} else {
  New-Item -ItemType Directory -Force $Study | Out-Null
}
New-Item -ItemType Directory -Force $Prepared,$PreparedEvents | Out-Null

$Network = Join-Path $InputRoot "network\wuhan_method_testbed_v067.inp"
$SourceEvents = Join-Path $InputRoot "contracts\events_with_splits.csv"
$RainProv = Join-Path $InputRoot "contracts\rainfall_provenance.v067.json"
$ActScope = Join-Path $InputRoot "contracts\actuator_scope.v067.json"
$SensorCopy = Join-Path $InputRoot "contracts\sensor_nodes.txt"
$PriorityCopy = Join-Path $InputRoot "contracts\priority_nodes.txt"
$SensorProv = Join-Path $InputRoot "contracts\sensor_layout_provenance.project7.v1.json"

Copy-Item -LiteralPath (Join-Path $Repo "configs\sensor_layout_provenance.project7.v1.json") `
  -Destination $SensorProv -Force

# Validate the source 18/6/6 registry, then deterministically prepare the effective-120 event clock.
rtc-validate-rainfall-design `
  --events $SourceEvents `
  --out (Join-Path $Logs "rainfall_design_source_v069.json")

rtc-prepare-event-suite `
  --events $SourceEvents `
  --out-dir $PreparedEvents `
  --out-registry $PreparedRegistry `
  --target-effective-warmup-minutes 120 `
  --post-rain-tail-minutes 360

rtc-validate-rainfall-design `
  --events $PreparedRegistry `
  --out (Join-Path $Logs "rainfall_design_prepared120_v069.json")

$PreparedFrame = Import-Csv -LiteralPath $PreparedRegistry
if ($PreparedFrame.Count -ne 30) { throw "Prepared registry must contain exactly 30 events" }
$BadWarmup = @($PreparedFrame | Where-Object { [math]::Abs([double]$_.effective_warmup_minutes - 120.0) -gt 1e-6 })
if ($BadWarmup.Count -gt 0) { throw "Prepared registry contains events that are not effective-120 min" }
$BadTail = @($PreparedFrame | Where-Object { [math]::Abs([double]$_.post_rain_tail_minutes - 360.0) -gt 1e-6 })
if ($BadTail.Count -gt 0) { throw "Prepared registry contains events that are not 360-min post-rain tail" }

# The fresh workspace MUST bind the prepared effective-120 registry, because every downstream
# D0/D1/D2/D3/model/runtime/Policy-Lock artifact uses that event clock.
rtc-init-fresh-workspace `
  --root $Study `
  --inp $Network `
  --priority $PriorityCopy `
  --events $PreparedRegistry

rtc-inp-audit-v2 `
  --inp $Network `
  --priority $PriorityCopy `
  --out (Join-Path $Study "preflight\inp_audit.json")

rtc-compile-formal-assets `
  --inp $Network `
  --priority $PriorityCopy `
  --sensors $SensorCopy `
  --out-dir (Join-Path $Study "formal_assets")

rtc-check-study-readiness `
  --events $PreparedRegistry `
  --frozen-inp $Network `
  --sensors $SensorCopy `
  --sensor-provenance $SensorProv `
  --rainfall-provenance $RainProv `
  --actuator-scope $ActScope `
  --history-span-minutes 60 `
  --minimum-post-rain-tail-minutes 360 `
  --out (Join-Path $Study "contracts\study_readiness.json")

# Keep a copy in logs for operator inspection without changing the canonical Policy-Lock path.
Copy-Item -LiteralPath (Join-Path $Study "contracts\study_readiness.json") `
  -Destination (Join-Path $Logs "study_readiness_v069.json") -Force

Write-Host "Project7 v0.6.9 Step 0 completed with the frozen 18/6/6 split and effective-120 prepared registry."
Write-Host "Repo:             $Repo"
Write-Host "InputRoot:        $InputRoot"
Write-Host "PreparedRegistry: $PreparedRegistry"
Write-Host "Study:            $Study"
Write-Host "Adoption:         $Adoption"
