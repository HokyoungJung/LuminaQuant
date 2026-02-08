#!/bin/bash

# Self-preserving logic: Run from temp
script_path=$(realpath "$0")
if [[ "$script_path" == *"LuminaQuant"* ]]; then
    temp_path="/tmp/sync_private.sh"
    echo "Copying script to temp: $temp_path"
    cp "$script_path" "$temp_path"
    chmod +x "$temp_path"
    exec "$temp_path"
fi

# Get current branch
current_branch=$(git branch --show-current)

if [ "$current_branch" == "private-main" ]; then
    echo -e "\033[0;36mAlready on private-main. Proceeding with merge...\033[0m"
else
    echo -e "\033[0;36mSwitching to private-main...\033[0m"
    git checkout private-main
fi

echo -e "\033[0;36mMerging changes from main...\033[0m"
git merge main -m "sync: merge from main"

echo -e "\033[0;36mAdding all files (including private ones)...\033[0m"
git add .
git commit -m "sync: update private repository"

echo -e "\033[0;36mPushing to private repository...\033[0m"
git push private private-main:main

echo -e "\033[0;36mSwitching back to $current_branch...\033[0m"
git checkout "$current_branch"

echo -e "\033[0;32mDone! Private repository is up to date.\033[0m"
