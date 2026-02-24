# WSL + MT5 Command Set

This project can run in WSL and still use MT5 by calling a Windows Python bridge.

## 1) Windows host setup (PowerShell)

```powershell
# Install MT5 extra only on Windows host Python env
uv sync --extra mt5

# Quick check: MetaTrader5 import works
uv run python -c "import MetaTrader5 as mt5; print(mt5.__version__)"
```

## 2) WSL setup (bash)

```bash
# Base deps for WSL runtime (no uv build)
uv sync --extra live --extra dev

# Optional: if TA-Lib wheel is not available, install OS deps first
sudo apt-get update
sudo apt-get install -y build-essential ta-lib libta-lib0 libta-lib-dev
```

## 3) Configure MT5 bridge in WSL (.env)

Set the Windows Python executable path. Example:

```bash
cat >> .env <<'EOF'
LQ__LIVE__EXCHANGE__DRIVER=mt5
LQ__LIVE__MT5_BRIDGE_PYTHON=/mnt/c/Users/<WINDOWS_USER>/AppData/Local/Programs/Python/Python313/python.exe
LQ__LIVE__MT5_BRIDGE_SCRIPT=scripts/mt5_bridge_worker.py
LQ__LIVE__MT5_BRIDGE_USE_WSLPATH=true
EOF
```

If you use the local Postgres + Parquet stack:

```bash
cat >> .env <<'EOF'
LQ__STORAGE__BACKEND=parquet-postgres
LQ__STORAGE__MARKET_DATA_PARQUET_PATH=data/market_parquet
LQ_POSTGRES_DSN=postgresql://<USER>:<PASS>@127.0.0.1:5432/<DB>
EOF
```

## 4) Bridge sanity checks (from WSL)

```bash
# Verify the bridge worker can be invoked by Windows Python
"/mnt/c/Users/<WINDOWS_USER>/AppData/Local/Programs/Python/Python313/python.exe" \
  "$(wslpath -w "$PWD/scripts/mt5_bridge_worker.py")" \
  --action connect --payload '{}'
```

Expected output:

```json
{"ok": true, "result": {"connected": true, ...}, "error": ""}
```

## 5) Runtime checks

```bash
# Configuration and core tests
python -m pytest tests/test_runtime_config_loader.py tests/test_mt5_exchange.py tests/test_data_sync.py

# Start live runner in paper mode first
uv run python run_live.py
```

## 6) Common troubleshooting

```bash
# If bridge path conversion fails, force explicit Windows path to script
export LQ__LIVE__MT5_BRIDGE_SCRIPT="C:\\Users\\<WINDOWS_USER>\\...\\LuminaQuant\\scripts\\mt5_bridge_worker.py"

# If Windows python cannot import MetaTrader5, re-install extra on host
powershell.exe -NoProfile -Command "cd C:\\path\\to\\LuminaQuant; uv sync --extra mt5"
```
