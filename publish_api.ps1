$ErrorActionPreference = "Stop"

function Restore-MainAfterFailedPublish {
    git reset --hard HEAD *> $null
    git clean -fd *> $null
    git checkout private-main *> $null
}

$protectedPaths = @(
    "AGENTS.md",
    ".env",
    ".omx",
    ".sisyphus",
    "data",
    "logs",
    "reports",
    "best_optimized_parameters",
    "equity.csv",
    "trades.csv",
    "live_equity.csv",
    "live_trades.csv",
    "strategies",
    "lumina_quant/indicators",
    "lumina_quant/data_sync.py",
    "lumina_quant/data_collector.py",
    "scripts/sync_binance_ohlcv.py",
    "scripts/collect_market_data.py",
    "scripts/collect_universe_1s.py",
    "tests/test_data_sync.py"
)

$scriptPath = $MyInvocation.MyCommand.Path
if ($scriptPath -like "*\Quants-agent\LuminaQuant\*") {
    $tempPath = Join-Path $env:TEMP "publish_api.ps1"
    Write-Host "Copying script to temp: $tempPath" -ForegroundColor DarkGray
    Copy-Item $scriptPath $tempPath -Force
    & $tempPath
    exit
}

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "private-main") {
    Write-Host "Please run this script from the 'private-main' branch." -ForegroundColor Red
    exit 1
}

$status = git status --porcelain
if ($status) {
    Write-Host "You have uncommitted changes. Please commit or stash them first." -ForegroundColor Red
    exit 1
}

Write-Host "Switching to main..." -ForegroundColor Cyan
git checkout main

Write-Host "Merging changes from private-main (without committing)..." -ForegroundColor Cyan
$mergeFailed = $false
try {
    git merge private-main --no-commit --no-ff
}
catch {
    $mergeFailed = $true
}
if ($LASTEXITCODE -ne 0) {
    $mergeFailed = $true
}
if ($mergeFailed) {
    Write-Host "Merge had conflicts. Preferring private-main content before filtering..." -ForegroundColor Yellow
    git checkout --theirs -- . *> $null
    git add -A *> $null
}

Write-Host "Enforcing public .gitignore..." -ForegroundColor Cyan
git checkout HEAD -- .gitignore

Write-Host "Preparing staged public set..." -ForegroundColor Cyan
git reset
git add .

Write-Host "Removing protected/sensitive paths from staging..." -ForegroundColor Cyan
foreach ($p in $protectedPaths) {
    git rm -r --cached --ignore-unmatch -- $p *> $null
}

Write-Host "Validating sensitive paths are absent from staged tree..." -ForegroundColor Cyan
$sensitiveRegex = '^strategies/|^lumina_quant/indicators/|^data/|^logs/|^reports/|^best_optimized_parameters/|^\.omx/|^\.sisyphus/|^AGENTS\.md$|^\.env$|^lumina_quant/data_sync\.py$|^lumina_quant/data_collector\.py$|^scripts/sync_binance_ohlcv\.py$|^scripts/collect_market_data\.py$|^scripts/collect_universe_1s\.py$|^tests/test_data_sync\.py$|(^|/)live_?equity\.csv$|(^|/)live_?trades\.csv$|(^|/)equity\.csv$|(^|/)trades\.csv$'
$staged = git diff --cached --name-only --diff-filter=ACMRT
$hasSensitive = $false
if ($staged) {
    foreach ($line in ($staged -split "`n")) {
        $trimmed = $line.Trim()
        if ($trimmed -and ($trimmed -match $sensitiveRegex)) {
            $hasSensitive = $true
            break
        }
    }
}
if ($hasSensitive) {
    Write-Host "Sensitive files are still staged. Aborting publish." -ForegroundColor Red
    Restore-MainAfterFailedPublish
    exit 1
}

$stagedNames = git diff --name-only --cached
if (-not $stagedNames) {
    Write-Host "No public changes to publish." -ForegroundColor Yellow
    Restore-MainAfterFailedPublish
    exit 0
}

Write-Host "Committing public changes..." -ForegroundColor Cyan
git commit -m "chore: publish updates from private repository"

Write-Host "Pushing to origin main..." -ForegroundColor Cyan
git push origin main

Write-Host "Switching back to private-main..." -ForegroundColor Cyan
git checkout private-main

Write-Host "Done! Public API published to 'main'." -ForegroundColor Green
