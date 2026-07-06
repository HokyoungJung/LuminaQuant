import json
import logging
import os
from logging.handlers import RotatingFileHandler


def _get_log_level() -> str:
    """Resolve log level: config.yaml system.log_level → LQ_LOG_LEVEL env → INFO."""
    try:
        from lumina_quant.configuration import get_default_runtime_config

        return get_default_runtime_config().system.log_level
    except Exception:
        return os.getenv("LQ_LOG_LEVEL", "INFO")


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


def _console_logging_disabled() -> bool:
    """Whether to skip the stderr console handler.

    The resilient wrapper (run_bot.sh / run_bot.bat) redirects stdout+stderr into
    ``logs/crash.log`` while the RotatingFileHandler already captures structured
    logs.  Setting ``LQ_DISABLE_CONSOLE_LOG=1`` (done by the wrappers) stops the
    console handler from duplicating every rotated line into the unbounded
    crash.log (audit O5).  Default (unset) keeps the console handler, so normal
    interactive/test behavior is byte-identical.
    """
    return os.getenv("LQ_DISABLE_CONSOLE_LOG", "0").strip().lower() in {"1", "true", "yes"}


def _ensure_root_log_handler(formatter: logging.Formatter) -> None:
    root = logging.getLogger()
    root.setLevel(_get_log_level())
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
    logger.setLevel(_get_log_level())
    logger.propagate = False

    # Prevent duplicate handlers when setup_logging is called multiple times.
    if logger.handlers:
        return logger

    plain_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    json_formatter = JsonLogFormatter()
    use_json = os.getenv("LUMINA_JSON_LOG", "0").strip().lower() in {"1", "true", "yes"}
    formatter = json_formatter if use_json else plain_formatter
    _ensure_root_log_handler(formatter)

    # Console Handler (skipped when the resilient wrapper already captures
    # stdout/stderr into crash.log — avoids duplicating every rotated line).
    if not _console_logging_disabled():
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
