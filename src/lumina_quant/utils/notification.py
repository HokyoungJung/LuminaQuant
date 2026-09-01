import json as _json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request

# Short socket timeout so a stalled/blackholed network can never pin the sender
# thread.  Delivery runs off the caller thread, so this never blocks the trading /
# order-managing loop regardless.
_HTTP_TIMEOUT_SEC = 5.0

# Bounded outbox: under a sustained Telegram/network outage the queue fills and
# the *oldest* alert is dropped so the freshest state always wins (a stale
# "freeze" alert is worthless once a newer "flatten" alert exists).
_DEFAULT_MAX_QUEUE = 256

# The background worker exits after this much idle time and is lazily restarted
# on the next enqueue, so a mostly-quiet manager never holds a live thread.
_WORKER_IDLE_TIMEOUT_SEC = 30.0

_TELEGRAM_API = "https://api.telegram.org"


def _async_delivery_default() -> bool:
    """Off-thread delivery is the safe default.

    ``LQ_NOTIFY_SYNC=1`` forces the legacy fully-synchronous send for
    constrained/embedded contexts and deterministic unit tests.
    """
    return os.getenv("LQ_NOTIFY_SYNC", "0").strip().lower() not in {"1", "true", "yes"}


def _urlopen(request: urllib.request.Request, timeout: float):
    """Seam around ``urllib.request.urlopen`` (patched in tests).

    Uses only the stdlib so Telegram alerting works on any install — including
    the base install without the optional ``live`` extra — instead of silently
    degrading to a log line when a third-party HTTP client is absent (audit O1).
    """
    return urllib.request.urlopen(request, timeout=timeout)


class NotificationManager:
    """Sends notifications via Telegram off the caller thread.

    Network I/O never runs on the caller (trading / order-managing) thread: each
    :meth:`send_message` enqueues onto a bounded, drop-oldest queue drained by a
    background daemon worker using a short socket timeout.  The public
    ``send_message(message)`` signature is unchanged.  Delivery uses stdlib
    ``urllib`` so it has no third-party dependency.
    """

    def __init__(
        self,
        bot_token,
        chat_id,
        *,
        max_queue_size: int = _DEFAULT_MAX_QUEUE,
        async_delivery: bool | None = None,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger("NotificationManager")
        self.enabled = bool(bot_token and chat_id)
        self._async_delivery = (
            _async_delivery_default() if async_delivery is None else bool(async_delivery)
        )
        try:
            capacity = int(max_queue_size)
        except TypeError, ValueError:
            capacity = _DEFAULT_MAX_QUEUE
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, capacity))
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._dropped_count = 0

        if not self.enabled:
            self.logger.warning("Telegram Bot Token or Chat ID missing. Notifications disabled.")

    # ------------------------------------------------------------------ API
    def deliverable(self) -> bool:
        """True when a live send can actually reach Telegram.

        Real-mode preflight should refuse to start (or loudly warn) when this is
        False so that FLATTEN/freeze/drift/hard-halt alerts are not silently lost
        (audit O1).  Delivery uses stdlib ``urllib``, so this is simply whether a
        token+chat are configured.
        """
        return bool(self.enabled)

    def send_message(self, message):
        """Queue a text message for delivery to the configured Telegram chat.

        Returns immediately: the network send happens on a background thread so
        no alert I/O runs on the caller thread.
        """
        if not self.enabled:
            return
        if not self._async_delivery:
            self._deliver(message)
            return
        self._enqueue(message)
        self._ensure_worker()

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until queued messages are delivered.

        Test / graceful-shutdown helper.  Returns True if the queue drained
        within ``timeout`` seconds, False on timeout.
        """
        if not self._async_delivery:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    # ------------------------------------------------------------ internals
    def _enqueue(self, message) -> None:
        try:
            self._queue.put_nowait(message)
            return
        except queue.Full:
            pass
        # Drop the oldest queued alert to make room for the newest one.
        try:
            self._queue.get_nowait()
            self._queue.task_done()
            self._dropped_count += 1
            self.logger.warning(
                "Notification queue full; dropped oldest alert (total dropped=%d)",
                self._dropped_count,
            )
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            pass

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            worker = threading.Thread(
                target=self._run,
                name="NotificationSender",
                daemon=True,
            )
            self._worker = worker
            worker.start()

    def _run(self) -> None:
        while True:
            try:
                message = self._queue.get(timeout=_WORKER_IDLE_TIMEOUT_SEC)
            except queue.Empty:
                # Idle: exit and let the next enqueue restart us.  The producer
                # always enqueues *before* calling _ensure_worker, so a message
                # racing this check is either already visible here (queue not
                # empty -> keep running) or triggers a fresh worker start.
                with self._worker_lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                self._deliver(message)
            finally:
                self._queue.task_done()

    def _deliver(self, message) -> None:
        url = f"{_TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        body = _json.dumps(
            {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with _urlopen(request, timeout=_HTTP_TIMEOUT_SEC) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                if status != 200:
                    # SECURITY: never log the URL — it embeds the bot token.
                    self.logger.error("Failed to send Telegram message: HTTP %s", status)
        except urllib.error.HTTPError as e:
            # SECURITY: HTTPError carries `.url`/`.filename` (token-bearing) and
            # its body may echo request context.  Log only the status code.
            self.logger.error("Failed to send Telegram message: HTTP %s", e.code)
        except Exception as e:
            # SECURITY: the request URL embeds the bot token, and a raw exception
            # string (e.g. URLError) can include that URL.  Log only the
            # exception type so the token never reaches logs.
            self.logger.error("Error sending Telegram message: %s", type(e).__name__)
