#!/bin/bash

# Self-preserving logic: Run from temp
script_path=$(realpath "$0")
if [[ "$script_path" == *"LuminaQuant"* ]]; then
    temp_path="/tmp/publish_api.sh"
    echo "Copying script to temp: $temp_path"
    cp "$script_path" "$temp_path"
    chmod +x "$temp_path"
    exec "$temp_path"
fi

# Get current branch
current_branch=$(git branch --show-current)

if [ "$current_branch" != "private-main" ]; then
    echo -e "\033[0;31mPlease run this script from the 'private-main' branch.\033[0m"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "\033[0;31mYou have uncommitted changes. Please commit or stash them first.\033[0m"
    exit 1
fi

echo -e "\033[0;36mSwitching to main...\033[0m"
git checkout main

echo -e "\033[0;36mMerging changes from private-main (without committing)...\033[0m"
# --no-commit: stop before committing to allow us to filter files
# --no-ff: always create a merge commit
git merge private-main --no-commit --no-ff

echo -e "\033[0;36mEnforcing public .gitignore...\033[0m"
# Restore .gitignore from main (HEAD) to ensure we don't accidentally use the private one
git checkout HEAD -- .gitignore

echo -e "\033[0;36mFiltering private files...\033[0m"
# Unstage EVERYTHING. 
git reset

# Add files again. Respected public .gitignore will filter out private files.
git add .

# Check if there are any staged changes
if git diff --cached --quiet; then
    echo -e "\033[0;33mNo public changes to publish.\033[0m"
    git merge --abort
    git checkout private-main
    exit 0
fi

echo -e "\033[0;36mCommitting public changes...\033[0m"
git commit -m "chore: publish updates from private repository"

echo -e "\033[0;36mPushing to origin main...\033[0m"
git push origin main

echo -e "\033[0;36mSwitching back to private-main...\033[0m"
git checkout private-main

echo -e "\033[0;32mDone! Public API published to 'main'.\033[0m"
