"""O1 real-money-readiness fix: the live preflight must refuse (or loudly warn)
when the notifier cannot deliver.

On the documented Binance-only install, every kill-switch/freeze/FLATTEN alert
silently fails when TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are unset. Real-money
gates (mode='real', or go_live_stage in {canary, full}) must hard-fail; paper/
testnet must keep today's behavior unchanged (a loud warning only, never a new
failure).
"""

from __future__ import annotations

from types import SimpleNamespace

from lumina_quant.cli.live import _check_notifier_deliverability

_TOKEN = "123456:ABC-DEF"
_CHAT_ID = "-100123456"


def _live_cfg(*, token: str | None, chat_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(telegram_bot_token=token, telegram_chat_id=chat_id)


def test_real_mode_hard_fails_when_notifier_not_configured():
    live_cfg = _live_cfg(token=None, chat_id=None)
    result = _check_notifier_deliverability(live_cfg, real_money_gate=True, mode_label="REAL")
    assert result is not None
    assert "FATAL" in result
    assert "Refusing to start real-money trading" in result


def test_real_mode_passes_when_notifier_configured():
    live_cfg = _live_cfg(token=_TOKEN, chat_id=_CHAT_ID)
    result = _check_notifier_deliverability(live_cfg, real_money_gate=True, mode_label="REAL")
    assert result is None


def test_paper_mode_warns_but_does_not_fail(capsys):
    live_cfg = _live_cfg(token=None, chat_id=None)
    result = _check_notifier_deliverability(live_cfg, real_money_gate=False, mode_label="PAPER")
    assert result is None  # no new failure for paper/testnet
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "Notifier preflight FAILED" in captured.out


def test_paper_mode_configured_notifier_is_silent(capsys):
    live_cfg = _live_cfg(token=_TOKEN, chat_id=_CHAT_ID)
    result = _check_notifier_deliverability(live_cfg, real_money_gate=False, mode_label="PAPER")
    assert result is None
    captured = capsys.readouterr()
    assert "[WARN]" not in captured.out


def test_canary_stage_gate_hard_fails_even_labeled_paper():
    """Stage alone (canary/full) is enough to trip the real-money gate,
    independent of the mode label the caller passes in for display."""
    live_cfg = _live_cfg(token=None, chat_id=None)
    result = _check_notifier_deliverability(live_cfg, real_money_gate=True, mode_label="PAPER")
    assert result is not None
    assert "FATAL" in result
