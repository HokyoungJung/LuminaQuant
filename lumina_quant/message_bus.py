"""Lightweight in-process message bus with deterministic dispatch order."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Handler = Callable[[Any], Any]


@dataclass(slots=True)
class MessageBusStats:
    published: int = 0
    delivered: int = 0
    requests: int = 0
    routed: int = 0


class MessageBus:
    """Supports publish/subscribe, request/response, and point-to-point routing."""

    def __init__(self):
        self._topics: dict[str, list[Handler]] = defaultdict(list)
        self._request_handlers: dict[str, Handler] = {}
        self._routes: dict[str, deque[Handler]] = defaultdict(deque)
        self.stats = MessageBusStats()

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._topics[str(topic)].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        handlers = self._topics.get(str(topic), [])
        self._topics[str(topic)] = [candidate for candidate in handlers if candidate is not handler]

    def publish(self, topic: str, payload: Any) -> int:
        handlers = self._topics.get(str(topic), [])
        self.stats.published += 1
        delivered = 0
        for handler in handlers:
            handler(payload)
            delivered += 1
        self.stats.delivered += delivered
        return delivered

    def register_request_handler(self, endpoint: str, handler: Handler) -> None:
        self._request_handlers[str(endpoint)] = handler

    def request(self, endpoint: str, payload: Any) -> Any:
        handler = self._request_handlers.get(str(endpoint))
        if handler is None:
            raise KeyError(f"No request handler registered for endpoint '{endpoint}'")
        self.stats.requests += 1
        return handler(payload)

    def register_route(self, route: str, handler: Handler) -> None:
        self._routes[str(route)].append(handler)

    def route(self, route: str, payload: Any) -> Any:
        handlers = self._routes.get(str(route))
        if not handlers:
            raise KeyError(f"No route handlers registered for '{route}'")
        handler = handlers[0]
        handlers.rotate(-1)
        self.stats.routed += 1
        return handler(payload)
