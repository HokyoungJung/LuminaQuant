from __future__ import annotations

from lumina_quant.message_bus import MessageBus


def test_publish_subscribe_delivery_order():
    bus = MessageBus()
    received: list[str] = []

    bus.subscribe("topic.a", lambda payload: received.append(f"h1:{payload}"))
    bus.subscribe("topic.a", lambda payload: received.append(f"h2:{payload}"))
    delivered = bus.publish("topic.a", "x")

    assert delivered == 2
    assert received == ["h1:x", "h2:x"]


def test_request_response_handler():
    bus = MessageBus()
    bus.register_request_handler("risk.check", lambda payload: payload.get("ok", False))
    assert bus.request("risk.check", {"ok": True}) is True


def test_point_to_point_round_robin_route():
    bus = MessageBus()
    bus.register_route("order.route", lambda payload: f"a:{payload}")
    bus.register_route("order.route", lambda payload: f"b:{payload}")

    assert bus.route("order.route", "x") == "a:x"
    assert bus.route("order.route", "x") == "b:x"
    assert bus.route("order.route", "x") == "a:x"
