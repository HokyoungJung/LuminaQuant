"""Regression tests: MT5 password must never appear on the subprocess argv.

Covers credential-exposure audit finding #1 — argv is world-readable via
``/proc/<pid>/cmdline``, so the auth payload (login/password/server) is routed
through a private 0600 temp file passed as ``--payload-file`` instead.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from unittest.mock import patch

import lumina_quant.exchanges.mt5_exchange as mt5_module
from lumina_quant.exchanges.mt5_exchange import MT5Exchange

_PASSWORD = "S3cr3t-MT5-Pass!"
_LOGIN = 555111
_SERVER = "Broker-Demo"


class _BridgeConfig:
    MT5_LOGIN = _LOGIN
    MT5_PASSWORD = _PASSWORD
    MT5_SERVER = _SERVER
    MT5_MAGIC = 234000
    MT5_DEVIATION = 20


def _proc(payload: dict) -> object:
    class _Result:
        returncode = 0
        stdout = json.dumps(payload, ensure_ascii=True)
        stderr = ""

    return _Result()


class TestMT5SecretArgv(unittest.TestCase):
    def setUp(self):
        self.original_mt5 = mt5_module.mt5
        mt5_module.mt5 = None
        self.original_env = {
            k: os.environ.get(k)
            for k in (
                "LQ_MT5_BRIDGE_PYTHON",
                "LQ_MT5_BRIDGE_SCRIPT",
                "LQ_MT5_BRIDGE_USE_WSLPATH",
            )
        }
        os.environ["LQ_MT5_BRIDGE_PYTHON"] = "python.exe"
        os.environ.pop("LQ_MT5_BRIDGE_SCRIPT", None)
        os.environ["LQ_MT5_BRIDGE_USE_WSLPATH"] = "0"

    def tearDown(self):
        mt5_module.mt5 = self.original_mt5
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_password_not_on_argv_and_written_to_private_file(self, run_mock):
        captured = {}

        def _record(cmd, *args, **kwargs):
            # Snapshot the payload file contents/mode while it still exists
            # (it is unlinked in the finally block after subprocess.run returns).
            captured["cmd"] = list(cmd)
            if "--payload-file" in cmd:
                path = cmd[cmd.index("--payload-file") + 1]
                captured["file_mode"] = stat.S_IMODE(os.stat(path).st_mode)
                with open(path, encoding="ascii") as handle:
                    captured["file_payload"] = json.load(handle)
            return _proc({"ok": True, "result": {"connected": True}, "error": ""})

        run_mock.side_effect = _record

        exchange = MT5Exchange(_BridgeConfig())
        self.assertTrue(exchange.connected)

        cmd = captured["cmd"]
        # 1) The password must not appear anywhere on argv.
        joined = "\x00".join(str(tok) for tok in cmd)
        self.assertNotIn(_PASSWORD, joined)
        self.assertNotIn(_SERVER, joined)
        self.assertNotIn(str(_LOGIN), joined)

        # 2) The argv --payload value (legacy worker channel) carries no secret.
        payload_idx = cmd.index("--payload")
        argv_payload = json.loads(cmd[payload_idx + 1])
        self.assertNotIn("password", argv_payload)
        self.assertNotIn("login", argv_payload)
        self.assertNotIn("server", argv_payload)

        # 3) The secret reaches the worker via a 0600 temp file.
        self.assertEqual(captured["file_mode"], 0o600)
        self.assertEqual(captured["file_payload"]["password"], _PASSWORD)
        self.assertEqual(captured["file_payload"]["login"], _LOGIN)
        self.assertEqual(captured["file_payload"]["server"], _SERVER)

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_temp_payload_file_removed_after_call(self, run_mock):
        seen_paths = []

        def _record(cmd, *args, **kwargs):
            if "--payload-file" in cmd:
                seen_paths.append(cmd[cmd.index("--payload-file") + 1])
            return _proc({"ok": True, "result": {"connected": True}, "error": ""})

        run_mock.side_effect = _record
        MT5Exchange(_BridgeConfig())

        self.assertTrue(seen_paths)
        for path in seen_paths:
            self.assertFalse(os.path.exists(path), f"temp secret file not cleaned: {path}")

    @patch("lumina_quant.exchanges.mt5_exchange.subprocess.run")
    def test_routing_args_still_on_argv(self, run_mock):
        run_mock.return_value = _proc({"ok": True, "result": {"connected": True}, "error": ""})
        exchange = MT5Exchange(_BridgeConfig())
        run_mock.reset_mock()
        run_mock.return_value = _proc(
            {"ok": True, "result": [[1, 1.0, 2.0, 0.5, 1.5, 10.0]], "error": ""}
        )
        exchange.fetch_ohlcv("EURUSD", "1m", 1)

        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--action") + 1], "fetch_ohlcv")
        argv_payload = json.loads(cmd[cmd.index("--payload") + 1])
        # Non-secret routing fields remain on argv for backward compatibility.
        self.assertEqual(argv_payload["symbol"], "EURUSD")


if __name__ == "__main__":
    unittest.main()
