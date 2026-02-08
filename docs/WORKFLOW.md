# Developer Workflow

This repository uses a dual-branch strategy to maintain a **private codebase** (with strategies/data) and a **public API** (core engine only).

## 1. Branch Structure

- **`private-main`**: The primary development branch. Contains ALL files including sensitive strategies, data, and logs.
- **`main`**: The public branch. strict `.gitignore` filters out private files.

## 2. Automation Scripts

We provide automation scripts for both Windows (PowerShell) and Mac/Linux (Bash).

### A. Syncing to Private Repository
**Goal**: Backup all your work (including secrets) to your private GitHub repo.

*   **Usage**: Switch to `private-main` first (`git checkout private-main`).
*   **Windows**:
    ```powershell
    .\sync_private.ps1
    ```
*   **Mac/Linux**:
    ```bash
    ./sync_private.sh
    ```

### B. Publishing to Public API
**Goal**: Update the public `main` branch with core engine changes, but *exclude* your private strategies and data.

*   **Windows**:
    ```powershell
    .\publish_api.ps1
    ```
*   **Mac/Linux**:
    ```bash
    chmod +x publish_api.sh  # Run once
    ./publish_api.sh
    ```

## 3. Manual Process (If scripts fail)

If you need to do this manually:

**Sync Private:**
```bash
git checkout private-main
git merge main
git add .
git commit -m "sync"
git push private private-main:main
```

**Publish Public:**
```bash
git checkout main
git merge private-main --no-commit --no-ff
git checkout HEAD -- .gitignore
git reset
git add .
git commit -m "chore: publish"
git push origin main
```
