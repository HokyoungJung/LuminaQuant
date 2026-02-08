# Self-preserving logic: Copy to temp and run from there if running from repo
$scriptPath = $MyInvocation.MyCommand.Path
if ($scriptPath -like "*\Quants-agent\LuminaQuant\*") {
    $tempPath = Join-Path $env:TEMP "publish_api.ps1"
    Write-Host "Copying script to temp: $tempPath" -ForegroundColor DarkGray
    Copy-Item $scriptPath $tempPath -Force
    
    # Run the temp copy and exit
    & $tempPath
    exit
}

$currentBranch = git branch --show-current

if ($currentBranch -ne "private-main") {
    Write-Host "Please run this script from the 'private-main' branch." -ForegroundColor Red
    exit
}

# Check for uncommitted changes
$status = git status --porcelain
if ($status) {
    Write-Host "You have uncommitted changes. Please commit or stash them first." -ForegroundColor Red
    exit
}

Write-Host "Switching to main..." -ForegroundColor Cyan
git checkout main

Write-Host "Merging changes from private-main (without committing)..." -ForegroundColor Cyan
# --no-commit: stop before committing to allow us to filter files
# --no-ff: always create a merge commit (easier to see history)
git merge private-main --no-commit --no-ff

Write-Host "Enforcing public .gitignore..." -ForegroundColor Cyan
# Restore .gitignore from main (HEAD) to ensure we don't accidentally unwanted rules
git checkout HEAD -- .gitignore

Write-Host "Filtering private files..." -ForegroundColor Cyan
# Unstage EVERYTHING (mixed reset). This keeps the file changes in your folder
# but clears the "ready to be committed" list.
git reset

# Now add files again. Since we restored the public .gitignore,
# 'git add .' will IGNORING the private files (strategies, data, etc.)
git add .

$staged = git diff --name-only --cached
if (-not $staged) {
    Write-Host "No public changes to publish." -ForegroundColor Yellow
    git merge --abort
    git checkout private-main
    exit
}

Write-Host "Committing public changes..." -ForegroundColor Cyan
git commit -m "chore: publish updates from private repository"

Write-Host "Pushing to origin main..." -ForegroundColor Cyan
git push origin main

Write-Host "Switching back to private-main..." -ForegroundColor Cyan
git checkout private-main

Write-Host "Done! Public API published to 'main'." -ForegroundColor Green
# Self-preserving logic: Copy to temp and run from there if running from repo
$scriptPath = $MyInvocation.MyCommand.Path
if ($scriptPath -like "*\Quants-agent\LuminaQuant\*") {
    $tempPath = Join-Path $env:TEMP "publish_api.ps1"
    Write-Host "Copying script to temp: $tempPath" -ForegroundColor DarkGray
    Copy-Item $scriptPath $tempPath -Force
    
    # Run the temp copy and exit
    & $tempPath
    exit
}

$currentBranch = git branch --show-current

if ($currentBranch -ne "private-main") {
    Write-Host "Please run this script from the 'private-main' branch." -ForegroundColor Red
    exit
}

# Check for uncommitted changes
$status = git status --porcelain
if ($status) {
    Write-Host "You have uncommitted changes. Please commit or stash them first." -ForegroundColor Red
    exit
}

Write-Host "Switching to main..." -ForegroundColor Cyan
git checkout main

Write-Host "Merging changes from private-main (without committing)..." -ForegroundColor Cyan
# --no-commit: stop before committing to allow us to filter files
# --no-ff: always create a merge commit (easier to see history)
git merge private-main --no-commit --no-ff

Write-Host "Enforcing public .gitignore..." -ForegroundColor Cyan
# Restore .gitignore from main (HEAD) to ensure we don't accidentally unwanted rules
git checkout HEAD -- .gitignore

Write-Host "Filtering private files..." -ForegroundColor Cyan
# Unstage EVERYTHING (mixed reset). This keeps the file changes in your folder
# but clears the "ready to be committed" list.
git reset

# Now add files again. Since we restored the public .gitignore,
# 'git add .' will IGNORING the private files (strategies, data, etc.)
git add .

$staged = git diff --name-only --cached
if (-not $staged) {
    Write-Host "No public changes to publish." -ForegroundColor Yellow
    git merge --abort
    git checkout private-main
    exit
}

Write-Host "Committing public changes..." -ForegroundColor Cyan
git commit -m "chore: publish updates from private repository"

Write-Host "Pushing to origin main..." -ForegroundColor Cyan
git push origin main

Write-Host "Switching back to private-main..." -ForegroundColor Cyan
git checkout private-main

Write-Host "Done! Public API published to 'main'." -ForegroundColor Green
