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
WorkingDirectory=/home/ubuntu/lumina-quant
ExecStart=/home/ubuntu/.local/bin/uv run python run_live_ws.py
Restart=always
EnvironmentFile=/home/ubuntu/lumina-quant/.env

[Install]
WantedBy=multi-user.target
```

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
