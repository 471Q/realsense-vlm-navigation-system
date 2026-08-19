# Validates the frozen schema set against its fixtures.
#
# Requires PowerShell 7 or later: Test-Json's -SchemaFile parameter is the schema engine the
# freeze was recorded against, and Windows PowerShell 5.1 does not support it. The complementary
# Python checks in tests/test_schema_freeze.py run anywhere and cover manifest drift and the
# runtime validator binding, which this script does not.

$ErrorActionPreference = 'Stop'

$schemaRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$validRoot = Join-Path $schemaRoot 'fixtures/valid'
$invalidRoot = Join-Path $schemaRoot 'fixtures/invalid'
$manifestPath = Join-Path $schemaRoot 'schema-manifest.v2.json'

$validCases = @(
    @('hdsg.fact_packet.v1.schema.json', 'hdsg.fact_packet.v1.json'),
    @('hdsg.prompt_packet.v1.schema.json', 'hdsg.prompt_packet.v1.json'),
    @('hdsg.vlm_candidate.v1.schema.json', 'hdsg.vlm_candidate.v1.json'),
    @('hdsg.release.v1.schema.json', 'hdsg.release.v1.json')
)

$invalidCases = @(
    @('hdsg.fact_packet.v1.schema.json', 'hdsg.fact_packet.v1.stationary_box.json'),
    @('hdsg.prompt_packet.v1.schema.json', 'hdsg.prompt_packet.v1.automatic_visuals.json'),
    @('hdsg.vlm_candidate.v1.schema.json', 'hdsg.vlm_candidate.v1.extra_action.json'),
    @('hdsg.release.v1.schema.json', 'hdsg.release.v1.accepted_fallback_code.json')
)

$failures = [System.Collections.Generic.List[string]]::new()

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($file in $manifest.files) {
    $filePath = Join-Path $schemaRoot $file.path
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
        $failures.Add("Manifest file is missing: $($file.path)")
        continue
    }
    $actualHash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $file.sha256) {
        $failures.Add("Manifest hash does not match: $($file.path)")
    }
}

foreach ($case in $validCases) {
    $schemaPath = Join-Path $schemaRoot $case[0]
    $fixturePath = Join-Path $validRoot $case[1]
    try {
        $accepted = Test-Json -LiteralPath $fixturePath -SchemaFile $schemaPath -ErrorAction Stop
        if (-not $accepted) {
            $failures.Add("Valid fixture was rejected: $($case[1])")
        }
    }
    catch {
        $failures.Add("Valid fixture was rejected: $($case[1])")
    }
}

foreach ($case in $invalidCases) {
    $schemaPath = Join-Path $schemaRoot $case[0]
    $fixturePath = Join-Path $invalidRoot $case[1]
    $rejected = $false
    try {
        $accepted = Test-Json -LiteralPath $fixturePath -SchemaFile $schemaPath -ErrorAction Stop
        $rejected = -not $accepted
    }
    catch {
        $rejected = $true
    }
    if (-not $rejected) {
        $failures.Add("Invalid fixture was accepted: $($case[1])")
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output "Schema validation passed: manifest hashes matched, $($validCases.Count) valid fixtures were accepted and $($invalidCases.Count) invalid fixtures were rejected."
