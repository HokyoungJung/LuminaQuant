# Self-preserving logic: Copy to temp and run from there if running from repo
$scriptPath = $MyInvocation.MyCommand.Path
if ($scriptPath -like "*\Quants-agent\LuminaQuant\*") {
    $tempPath = Join-Path $env:TEMP "sync_private.ps1"
    Write-Host "Copying script to temp: $tempPath" -ForegroundColor DarkGray
    Copy-Item $scriptPath $tempPath -Force
    
    # Run the temp copy and exit
    & $tempPath
    exit
}

$currentBranch = git branch --show-current

if ($currentBranch -eq "private-main") {
    Write-Host "Already on private-main. Proceeding with merge..." -ForegroundColor Cyan
}
else {
    # If ran from somewhere else (e.g. valid path), switch
    Write-Host "Switching to private-main..." -ForegroundColor Cyan
    git checkout private-main
}

# Check for uncommitted changes
$status = git status --porcelain
if ($status) {
    Write-Host "You have uncommitted changes. Please commit or stash them first." -ForegroundColor Red
    exit
}

Write-Host "Merging changes from main..." -ForegroundColor Cyan
git merge main -m "sync: merge from main"

Write-Host "Adding all files (including private ones)..." -ForegroundColor Cyan
git add .
# Force add private files ignored by .gitignore
git add -f strategies/
git add -f data/
git add -f reports/
git add -f equity.csv
git add -f trades.csv
git add -f live_equity.csv
git add -f live_trades.csv
git add -f best_optimized_parameters/
git add -f .env

git commit -m "sync: update private repository"

Write-Host "Pushing to private repository..." -ForegroundColor Cyan
git push private private-main:main

Write-Host "Done! Private repository is up to date." -ForegroundColor Green
$currentBranch = git branch --show-current

if ($currentBranch -eq "private-main") {
    Write-Host "You are already on private-main. Please checkout main first." -ForegroundColor Red
    exit
}

# Check for uncommitted changes
$status = git status --porcelain
if ($status) {
    Write-Host "You have uncommitted changes. Please commit or stash them first." -ForegroundColor Red
    exit
}

Write-Host "Switching to private-main..." -ForegroundColor Cyan
git checkout private-main

Write-Host "Merging changes from main..." -ForegroundColor Cyan
git merge main -m "sync: merge from main"

Write-Host "Adding all files (including private ones)..." -ForegroundColor Cyan
git add .
# Force add private files ignored by .gitignore
git add -f strategies/
git add -f data/
git add -f reports/
git add -f equity.csv
git add -f trades.csv
git add -f live_equity.csv
git add -f live_trades.csv
git add -f best_optimized_parameters/
git add -f .env

git commit -m "sync: update private repository"

Write-Host "Pushing to private repository..." -ForegroundColor Cyan
git push private private-main:main

Write-Host "Switching back to $currentBranch..." -ForegroundColor Cyan
git checkout $currentBranch

Write-Host "Done! Private repository is up to date." -ForegroundColor Green
