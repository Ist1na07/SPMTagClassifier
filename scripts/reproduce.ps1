# Reproduce the current bounded joint studies; use a NEW output root.
param(
    [int]$Workers = 4,
    [string]$OutputRoot = 'research/rework/results/reproduced',
    [string]$FitRoot = '.cache/rework-reproduced-fits',
    [switch]$Legacy
)
$ErrorActionPreference = 'Stop'
if ($Legacy) { & "$PSScriptRoot/reproduce-v1.2.ps1"; exit $LASTEXITCODE }
function Run-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}
if ($Workers -lt 1) { throw 'Workers must be positive.' }
foreach ($file in @('dataset/spm_dataset-1.1.1/labels.json', 'dataset/spm_dataset-4k-0.2.0/labels.json', 'dataset/spm_dataset_6k-0.1.0/labels.json')) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing supplied dataset: $file" }
}
# Reuse validated inputs. Fresh extraction directly creates corrected R2;
# historical R1 caches and the one-time repair migration are not required.
if (-not ((Test-Path -LiteralPath '.cache/rework7.jsonl') -and (Test-Path -LiteralPath '.cache/rework-schema.json') -and (Test-Path -LiteralPath '.cache/archived-model-reference.json'))) {
    Run-Checked node @('research/rework/extract.cjs')
}
if (-not (Test-Path -LiteralPath '.cache/rework-legacy-full.json')) { Run-Checked node @('research/rework/aggregate-baseline.cjs') }
if (-not (Test-Path -LiteralPath '.cache/rework-family-meta.jsonl')) { Run-Checked node @('research/rework/family-meta-extract.cjs') }
if (-not ((Test-Path -LiteralPath '.cache/rework-release-geometry-r2.jsonl') -and (Test-Path -LiteralPath '.cache/rework-release-geometry-r2-schema.json'))) {
    Run-Checked node @('research/rework/release-extract.cjs')
}
if (-not (Test-Path -LiteralPath '.cache/rework-coarse-r2/inventory.json')) { Run-Checked node @('research/rework/coarse-extract.cjs') }
if (-not (Test-Path -LiteralPath '.cache/rework-six-physical-r2/inventory.json')) { Run-Checked node @('research/rework/six-extract.cjs') }
Run-Checked node @('research/rework/validate-native-cache.cjs', '.cache/rework-coarse-r2')
Run-Checked node @('research/rework/validate-native-cache.cjs', '.cache/rework-six-physical-r2')

Run-Checked python @('-X','utf8','research/rework/coarse_search.py','--workers',"$Workers",'--output',"$OutputRoot/coarse-raw",'--checkpoints',"$FitRoot/coarse")
Run-Checked python @('-X','utf8','research/rework/coarse_report.py',"$OutputRoot/coarse-raw")
Run-Checked python @('-X','utf8','research/rework/coarse_support_grid.py','--workers',"$Workers",'--output',"$OutputRoot/coarse-supported",'--fits',"$FitRoot/coarse",'--outer-fits',"$FitRoot/coarse-supported-outer")
Run-Checked python @('-X','utf8','research/rework/coarse_support_equivalence.py',"$OutputRoot/coarse-supported",'--parent',"$OutputRoot/coarse-raw")
Run-Checked python @('-X','utf8','research/rework/coarse_support_summary.py',"$OutputRoot/coarse-supported",'--parent',"$OutputRoot/coarse-raw")
Run-Checked python @('-X','utf8','research/rework/coarse_final.py','--source',"$OutputRoot/coarse-supported",'--output',"$OutputRoot/coarse-final")
Run-Checked node @('research/rework/six_runtime_audit.cjs',"$OutputRoot/coarse-final/6k/profile.json", "$OutputRoot/six-runtime")
foreach ($key in @('4k','6k')) {
    Run-Checked node @('research/rework/coarse_sanity.cjs',"$OutputRoot/coarse-final/$key/profile.json", "$OutputRoot/coarse-final/$key/semantic-contract.json",'both')
}

Run-Checked python @('-X','utf8','research/rework/ln_adapter_search.py','--workers',"$Workers",'--output',"$OutputRoot/ln-adapter",'--checkpoints',"$FitRoot/ln")
Run-Checked python @('-X','utf8','research/rework/ln_adapter_report.py',"$OutputRoot/ln-adapter")
Run-Checked python @('-X','utf8','research/rework/ln_gate_search.py','--workers',"$Workers",'--output',"$OutputRoot/ln-gate",'--parent',"$OutputRoot/ln-adapter",'--checkpoints',"$FitRoot/ln")
Run-Checked python @('-X','utf8','research/rework/ln_gate_report.py',"$OutputRoot/ln-gate")
Run-Checked python @('-X','utf8','research/rework/ln_final.py','--source',"$OutputRoot/ln-gate",'--output',"$OutputRoot/ln-final")
Write-Output "Joint-study results and all-training profiles: $OutputRoot"
Write-Output 'Published library models are not overwritten. Historical staged reproduction remains available with -Legacy.'
