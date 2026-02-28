# 🚀 Deployment Guide (Local / UV-Only)

This guide explains how to run **LuminaQuant** in the current local-first stack.
Docker deployment is intentionally out of scope for this runtime profile.

## ☁️ Linux Service (Systemd, uv-only)

### 1. Install Dependencies
```bash
apt update && apt install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
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
ExecStart=/home/ubuntu/.local/bin/uv run python run_live.py
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
ExecStart=/home/ubuntu/.local/bin/uv run python run_live_ws.py
```

Entrypoint choice:
- `run_live.py`: polling-based live runner (default/simple ops)
- `run_live_ws.py`: WebSocket-based live runner (lower latency path)

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
