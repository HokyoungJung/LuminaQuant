import json
import logging
import os
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


def _resolve_logs_dir() -> str:
    candidate = str(os.getenv("LQ_LOG_DIR", "logs") or "").strip()
    return candidate or "logs"


def _ensure_root_log_handler(formatter: logging.Formatter) -> None:
    root = logging.getLogger()
    root.setLevel(BaseConfig.LOG_LEVEL)
    for handler in list(root.handlers):
        if bool(getattr(handler, "_lumina_root_file_handler", False)):
            return

    logs_dir = _resolve_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)
    root_file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "lumina_quant.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    root_file_handler.setFormatter(formatter)
    root_file_handler._lumina_root_file_handler = True
    root.addHandler(root_file_handler)


def setup_logging(name="lumina_quant"):
    """Sets up a logger with a StreamHandler and FileHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(BaseConfig.LOG_LEVEL)
    logger.propagate = False

    # Prevent duplicate handlers when setup_logging is called multiple times.
    if logger.handlers:
        return logger

    plain_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    json_formatter = JsonLogFormatter()
    use_json = os.getenv("LUMINA_JSON_LOG", "0").strip().lower() in {"1", "true", "yes"}
    formatter = json_formatter if use_json else plain_formatter
    _ensure_root_log_handler(formatter)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Rotating: 10MB limit, 5 backups)
    logs_dir = _resolve_logs_dir()
    os.makedirs(logs_dir, exist_ok=True)
    fh = RotatingFileHandler(
        os.path.join(logs_dir, f"{name}.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
