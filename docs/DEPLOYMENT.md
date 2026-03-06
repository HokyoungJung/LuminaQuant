# 🚀 Deployment Guide (Local / UV-Only)

This guide explains how to run **LuminaQuant** in the current local-first stack.
Docker deployment is intentionally out of scope for this runtime profile.

## ☁️ Linux Service (Systemd, uv-only)

### 1. Install Dependencies
```bash
apt update && apt install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra optimize --extra live
# Optional for NVIDIA GPU nodes (Linux x86_64 + CUDA 12):
# uv sync --extra gpu
```

### 2. Create Service File
Create `/etc/systemd/system/lumina.service`:

```ini
[Unit]
Description=LuminaQuant Live Trader
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/{REPO_DIR}
ExecStart=/home/ubuntu/.local/bin/uv run lq live
Restart=always
EnvironmentFile=/opt/{REPO_DIR}/.env

[Install]
WantedBy=multi-user.target
```

`{REPO_DIR}` should match your actual clone folder name:
- `LuminaQuant` (public repo)
- `Quants-agent` (private repo)

If you prefer WebSocket feed in production, replace `ExecStart` with:

```ini
ExecStart=/home/ubuntu/.local/bin/uv run lq live --transport ws
```

Entrypoint choice:
- `lq live --transport poll`: polling-based live runner (default/simple ops)
- `lq live --transport ws`: WebSocket-based live runner (lower latency path)
- Effective source is controlled by `config.yaml`:
  - `live.market_data_source: committed` (default)
  - `live.market_data_source: binance_live` (real Binance stream)

Raw-first live data lifecycle (recommended for committed mode):
1. `scripts/collect_binance_aggtrades_raw.py` (raw collector, checkpoint resume, periodic loop)
2. `scripts/materialize_market_windows.py` (raw -> committed 1s+timeframe bundle, periodic loop)
3. `lq live --transport poll|ws` (committed reader by default, optional `binance_live` source)

First bootstrap recommendation:
```bash
uv run python scripts/collect_binance_aggtrades_raw.py \
  --symbols BTC/USDT,ETH/USDT \
  --since 2026-03-01T00:00:00Z \
  --no-periodic
```
- If `--since` is omitted and checkpoint is missing, collector starts from
  `now - storage.collector_bootstrap_lookback_hours` (default 24h).

Example periodic startup:
```bash
uv run python scripts/collect_binance_aggtrades_raw.py --symbols BTC/USDT,ETH/USDT --periodic --poll-seconds 2
uv run python scripts/materialize_market_windows.py --symbols BTC/USDT,ETH/USDT --timeframes 1s,1m,5m,15m,30m,1h,4h,1d --periodic --poll-seconds 5
uv run lq live
```

Fail-fast contract:
- background reader fatal (missing committed/parity decode error) is propagated to main
- live entrypoints exit with code `2`
- recovery requires restoring committed data then restarting collector -> materializer -> live

Binance live mode (non-HFT incremental rollout):
- set `live.market_data_source: binance_live`
- optional stream authority: `live.order_state_source: user_stream`
- keep `live.reconciliation_poll_fallback_enabled: true` for safety recovery

Verify committed data before live start:
```bash
uv run python - <<'PY'
from lumina_quant.storage.parquet import ParquetMarketDataRepository

repo = ParquetMarketDataRepository("data/market_parquet")
for symbol in ("BTC/USDT", "ETH/USDT"):
    frame = repo.load_committed_ohlcv_chunked(exchange="binance", symbol=symbol, timeframe="1s")
    print(symbol, "rows=", frame.height, "latest=", frame["datetime"].max())
PY
```

Before deploy, run the live decoupling architecture gate:
```bash
bash scripts/ci/architecture_gate_live_data.sh
bash scripts/ci/architecture_gate_market_window_contract.sh
```

Real mode safety:
- keep `live.require_real_enable_flag: true` in `config.yaml`
- arm real mode explicitly with `LUMINA_ENABLE_LIVE_REAL=true` and `--enable-live-real`
- for controlled shutdown, pass `--stop-file /tmp/lq.stop`

### 3. Enable & Start
```bash
sudo systemctl enable lumina
sudo systemctl start lumina
```

---

## 🤖 Telegram Bot Setup

1. Search for **@BotFather** on Telegram.
2. Send `/newbot` and follow instructions.
3. Get the **Token**.
4. Search for **@userinfobot** to get your **Chat ID**.
5. Add to `.env`:
   ```ini
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
   TELEGRAM_CHAT_ID=12345678
   ```
