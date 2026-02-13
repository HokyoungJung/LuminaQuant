import logging
import os
import json
from logging.handlers import RotatingFileHandler
from lumina_quant.config import BaseConfig


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(name="lumina_quant"):
    """
    Sets up a logger with a StreamHandler and FileHandler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(BaseConfig.LOG_LEVEL)
    logger.propagate = False

    # Prevent duplicate handlers when setup_logging is called multiple times.
    if logger.handlers:
        return logger

    plain_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    json_formatter = JsonLogFormatter()
    use_json = os.getenv("LUMINA_JSON_LOG", "0").strip().lower() in {"1", "true", "yes"}
    formatter = json_formatter if use_json else plain_formatter

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Rotating: 10MB limit, 5 backups)
    if not os.path.exists("logs"):
        os.makedirs("logs")

    fh = RotatingFileHandler(
        f"logs/{name}.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
