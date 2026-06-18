"""Regression tests: `lq config show` must never print raw secrets.

Covers api_key, secret_key, telegram_bot_token and the postgres DSN password
(see credential-exposure audit finding #2).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from lumina_quant.cli import config as cfg
from lumina_quant.configuration.schema import RuntimeConfig

_API_KEY = "AKIAFAKEKEY1234567890"
_SECRET_KEY = "shhh-super-secret-value-xyz"
_TELEGRAM_TOKEN = "123456:ABCDEF-FAKE-TELEGRAM-TOKEN"
_DSN_PASSWORD = "hunter2supersecret"
_DSN = f"postgresql://lquser:{_DSN_PASSWORD}@db.internal:5432/lumina"

_RAW_SECRETS = (_API_KEY, _SECRET_KEY, _TELEGRAM_TOKEN, _DSN_PASSWORD)


def _rt_with_secrets() -> RuntimeConfig:
    rt = RuntimeConfig()
    rt.live.api_key = _API_KEY
    rt.live.secret_key = _SECRET_KEY
    rt.live.telegram_bot_token = _TELEGRAM_TOKEN
    rt.storage.postgres_dsn = _DSN
    return rt


def test_rt_to_dict_masks_all_known_secret_fields() -> None:
    payload = cfg._rt_to_dict(_rt_with_secrets())

    assert payload["live"]["api_key"] != _API_KEY
    assert payload["live"]["api_key"].startswith("***")
    assert payload["live"]["secret_key"] != _SECRET_KEY
    assert payload["live"]["secret_key"].startswith("***")
    assert payload["live"]["telegram_bot_token"] != _TELEGRAM_TOKEN
    assert payload["live"]["telegram_bot_token"].startswith("***")

    # DSN keeps host/user but scrubs the embedded password.
    dsn_out = payload["storage"]["postgres_dsn"]
    assert _DSN_PASSWORD not in dsn_out
    assert "db.internal:5432/lumina" in dsn_out
    assert "lquser" in dsn_out


def test_rt_to_dict_serialised_output_contains_no_raw_secret() -> None:
    payload = cfg._rt_to_dict(_rt_with_secrets())
    serialised = json.dumps(payload, default=str)
    for raw in _RAW_SECRETS:
        assert raw not in serialised, f"secret leaked into config dump: {raw!r}"


def test_cmd_show_output_is_masked(monkeypatch) -> None:
    rt = _rt_with_secrets()
    monkeypatch.setattr(
        "lumina_quant.configuration.load_runtime_config",
        lambda _path: rt,
        raising=False,
    )

    class _Args:
        config = "config.yaml"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cfg.cmd_show(_Args())

    out = buf.getvalue()
    assert rc == 0
    for raw in _RAW_SECRETS:
        assert raw not in out, f"secret leaked into `lq config show` output: {raw!r}"
    # Sanity: masking marker is present so the field is shown (not silently dropped).
    assert "***" in out


def test_redact_secrets_handles_none_and_empty() -> None:
    rt = RuntimeConfig()  # telegram_bot_token defaults to None, dsn to ""
    payload = cfg._rt_to_dict(rt)
    assert payload["live"]["telegram_bot_token"] is None
    assert payload["storage"]["postgres_dsn"] == ""


def test_scrub_dsn_leaves_passwordless_dsn_untouched() -> None:
    plain = "postgresql://localhost:5432/lumina"
    assert cfg._scrub_dsn_password(plain) == plain
