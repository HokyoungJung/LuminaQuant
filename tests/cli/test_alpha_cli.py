"""Smoke + determinism tests for the `lq alpha` CLI (rank / card / promote).

Each sub-command is driven off the deterministic synthetic panel so the tests are
byte-reproducible and never touch config/schema or research source. They assert
exit code 0, well-formed output, and that re-running yields identical output.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from lumina_quant.cli import alpha as alpha_cli
from lumina_quant.cli import main as cli_main


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = alpha_cli.main(argv)
    return rc, buf.getvalue()


def test_rank_smoke_json_deterministic() -> None:
    rc1, out1 = _run(["rank", "--json", "--n-symbols", "6", "--n-periods", "80"])
    rc2, out2 = _run(["rank", "--json", "--n-symbols", "6", "--n-periods", "80"])
    assert rc1 == 0
    assert rc2 == 0
    assert out1 == out2  # deterministic

    payload = json.loads(out1)
    assert payload["artifact_kind"] == "alpha_factor_ranking"
    ranking = payload["ranking"]
    assert len(ranking) >= 1
    # Ranking is ordered by descending IC-IR with a stable rank index.
    irs = [float(row["ic_ir"]) for row in ranking]
    assert irs == sorted(irs, reverse=True)
    assert [row["rank"] for row in ranking] == list(range(1, len(ranking) + 1))


def test_rank_top_and_factor_filter() -> None:
    rc, out = _run(
        ["rank", "--json", "--factors", "close", "volume", "ret", "--top", "2"]
    )
    assert rc == 0
    payload = json.loads(out)
    assert len(payload["ranking"]) == 2
    factors = {row["factor"] for row in payload["ranking"]}
    assert factors <= {"close", "volume", "ret"}


def test_rank_text_mode_exit_zero() -> None:
    rc, out = _run(["rank"])
    assert rc == 0
    assert "Factor ranking" in out


def test_card_stdout_json_deterministic() -> None:
    rc1, out1 = _run(["card", "--factor", "close", "--json"])
    rc2, out2 = _run(["card", "--factor", "close", "--json"])
    assert rc1 == 0
    assert out1 == out2

    payload = json.loads(out1)
    assert payload["artifact_kind"] == "alpha_factor_card"
    assert payload["factor"] == "close"
    assert payload["metrics"]["factor"] == "close"
    assert "ic_ir" in payload["metrics"]


def test_card_writes_files(tmp_path) -> None:
    out_dir = tmp_path / "cards"
    rc, _out = _run(["card", "--factor", "volume", "--out", str(out_dir)])
    assert rc == 0
    json_path = out_dir / "volume.card.json"
    md_path = out_dir / "volume.card.md"
    assert json_path.exists()
    assert md_path.exists()
    card = json.loads(json_path.read_text(encoding="utf-8"))
    assert card["factor"] == "volume"
    assert md_path.read_text(encoding="utf-8").startswith("# Factor Card: volume")


def test_card_unknown_factor_returns_one() -> None:
    rc, _out = _run(["card", "--factor", "does_not_exist", "--json"])
    assert rc == 1


def test_promote_smoke_ledger_deterministic(tmp_path) -> None:
    ledger_a = tmp_path / "a.jsonl"
    ledger_b = tmp_path / "b.jsonl"
    rc1, out1 = _run(
        [
            "promote",
            "--json",
            "--n-symbols",
            "6",
            "--n-periods",
            "120",
            "--max-candidates",
            "16",
            "--ledger",
            str(ledger_a),
        ]
    )
    rc2, out2 = _run(
        [
            "promote",
            "--json",
            "--n-symbols",
            "6",
            "--n-periods",
            "120",
            "--max-candidates",
            "16",
            "--ledger",
            str(ledger_b),
        ]
    )
    assert rc1 == 0
    assert rc2 == 0

    p1 = json.loads(out1)
    p2 = json.loads(out2)
    assert p1["artifact_kind"] == "alpha_promotion_report"
    assert p1["promoted_ids"] == p2["promoted_ids"]  # deterministic promotion
    assert p1["evaluated_n"] == p2["evaluated_n"]
    assert p1["promoted_count"] == len(p1["promoted_ids"])
    assert p1["ledger"]["survived_count"] == p1["promoted_count"]


def test_promote_without_ledger_exit_zero() -> None:
    rc, out = _run(["promote", "--json", "--n-symbols", "5", "--n-periods", "80"])
    assert rc == 0
    payload = json.loads(out)
    assert "ledger" not in payload
    assert payload["evaluated_n"] >= 1


def test_alpha_registered_in_root_dispatch() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main.main(["alpha", "rank", "--json", "--n-periods", "40"])
    assert rc == 0
    assert json.loads(buf.getvalue())["artifact_kind"] == "alpha_factor_ranking"


def test_no_subcommand_prints_help() -> None:
    rc, _out = _run([])
    assert rc == 0
