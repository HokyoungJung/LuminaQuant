import requests
import logging


class NotificationManager:
    """
    Sends notifications via Telegram.
    """

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger("NotificationManager")
        self.enabled = bool(bot_token and chat_id)

        if not self.enabled:
            self.logger.warning(
                "Telegram Bot Token or Chat ID missing. Notifications disabled."
            )

    def send_message(self, message):
        """
        Sends a text message to the configured Telegram chat.
        """
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Failed to send Telegram message: {response.text}")
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
