# 🚀 Deployment Guide

This guide explains how to deploy **LuminaQuant** to a remote server (AWS, GCP, VPS) for 24/7 trading.

## 📦 Option 1: Docker (Recommended)

Docker ensures the environment is exactly the same as development.

### 1. Prerequisites
- Docker & Docker Compose installed on the server.
- `.env` file with API keys.

### 2. Setup
Copy the project to your server:
```bash
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd lumina-quant
```

Create/Edit `.env`:
```bash
nano .env
# Paste your keys
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

### 3. Run
Start in detached mode (background):
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop:
```bash
docker-compose down
```

---

## ☁️ Option 2: Linux Service (Systemd)

If you prefer running directly on the OS.

### 1. Install Dependencies
```bash
apt update && apt install python3-pip
pip install .
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
ExecStart=/usr/bin/python3 run_live_ws.py
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
