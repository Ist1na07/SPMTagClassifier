# Run from the repository root. Uses the user's local dataset; never uploads raw charts.
$ErrorActionPreference = 'Stop'
function Run-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit code $LASTEXITCODE" }
}
if (-not (Test-Path -LiteralPath 'dataset/spm_dataset-1.1.1/labels.json')) { throw 'Place the supplied dataset folders in dataset/ first.' }
& node research/extract.cjs
$inventory = Get-Content -Raw research/results/inventory.json | ConvertFrom-Json
if ($inventory.errors.Count -ne 1 -or $inventory.errors[0].uid -notlike '*bm6k_0778*' -or $inventory.errors[0].error -ne 'InvalidHoldTail') {
    throw 'Unexpected extraction failure; inspect research/results/inventory.json.'
}
Run-Checked node @('research/legacy-features.cjs')
Run-Checked python @('-X','utf8','research/train.py')
$names = @('LABELLED_ONLY','SKIP_LEGACY','ROW_MERGE_MS','FEATURE_OUTPUT','INVENTORY_OUTPUT','SKIP_MANUAL','VARIANTS','SPLIT_INPUT_DIR','FEATURE_INPUT','RESULT_DIR','MODEL_DIR')
$saved = @{}
foreach ($n in $names) { $saved[$n] = [Environment]::GetEnvironmentVariable($n, 'Process') }
try {
    $env:LABELLED_ONLY='1'; $env:SKIP_LEGACY='1'; $env:SKIP_MANUAL='1'; $env:VARIANTS='full'; $env:SPLIT_INPUT_DIR='research/results'
    foreach ($merge in @(0,5)) {
        $env:ROW_MERGE_MS="$merge"; $env:FEATURE_OUTPUT=".cache/features-merge$merge.jsonl"; $env:INVENTORY_OUTPUT=".cache/inventory-merge$merge.json"
        Run-Checked node @('research/extract.cjs')
        $env:FEATURE_INPUT=$env:FEATURE_OUTPUT; $env:RESULT_DIR="research/results/rowmerge$merge"; $env:MODEL_DIR=".cache/models-merge$merge"
        Run-Checked python @('-X','utf8','research/train.py')
    }
} finally {
    foreach ($n in $names) { [Environment]::SetEnvironmentVariable($n, $saved[$n], 'Process') }
}
Run-Checked python @('-X','utf8','research/select-rowmerge.py')
Run-Checked node @('research/project6.cjs')
Run-Checked python @('-X','utf8','research/tune6.py')
Run-Checked node @('research/tune-sort.cjs')
Run-Checked node @('scripts/build.cjs')
Run-Checked node @('--test','test/*.test.cjs')
Run-Checked node @('research/runtime-parity.cjs')
Run-Checked node @('research/smoke.cjs')
Run-Checked node @('research/apolar.cjs')
Run-Checked python @('-X','utf8','research/report.py')
