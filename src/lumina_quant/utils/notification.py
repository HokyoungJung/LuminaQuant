import logging

try:
    import requests
except Exception:
    requests = None


class NotificationManager:
    """Sends notifications via Telegram."""

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger("NotificationManager")
        self.enabled = bool(bot_token and chat_id)

        if not self.enabled:
            self.logger.warning("Telegram Bot Token or Chat ID missing. Notifications disabled.")

    def send_message(self, message):
        """Sends a text message to the configured Telegram chat."""
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        try:
            if requests is None:
                raise RuntimeError("requests dependency is unavailable")
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                # SECURITY: never log response.text or the raw URL — both may
                # echo the bot token.  Log only the HTTP status code.
                self.logger.error("Failed to send Telegram message: HTTP %s", response.status_code)
        except Exception as e:
            # SECURITY: the request URL embeds the bot token, and the raw
            # exception string (e.g. requests.ConnectionError) includes that
            # URL.  Log only the exception type so the token never reaches logs.
            self.logger.error("Error sending Telegram message: %s", type(e).__name__)
