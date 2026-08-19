"""Generate ``configs/research/named_quant_claude_suite_v1.json`` (deterministic).

The suite JSON doubles as (a) a ``--manifest`` for ``scripts/run_research_candidates.py``
(only the top-level ``candidates`` array is read there), (b) a ``--score-config``
(``candidate_research`` block) and (c) a pre-registered allocation cell spec for
``scripts/research/build_quality_gated_allocation.py --validate-cell-spec``.

Every candidate's ``params`` starts from the strategy's REAL ``get_param_schema()``
defaults and only the listed overrides are applied, so the file cannot drift
from the code.  Re-run this script after changing a schema and commit the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lumina_quant.research_universe import BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED
from lumina_quant.strategies.registry import resolve_strategy_class

OUTPUT = Path("configs/research/named_quant_claude_suite_v1.json")
CREATED_UTC = "2026-08-19T00:00:00Z"

CRYPTO10 = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "TRX/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "TON/USDT",
]
PRECIOUS = ["XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"]
ENERGY_INDUSTRIAL = ["COPPER/USDT", "CL/USDT", "BZ/USDT", "NATGAS/USDT"]
ETF_INDEX = [
    "QQQ/USDT",
    "SPY/USDT",
    "EWY/USDT",
    "EWJ/USDT",
    "SOXL/USDT",
    "EWT/USDT",
    "IWM/USDT",
    "EWZ/USDT",
    "XLE/USDT",
    "URNM/USDT",
    "UVXY/USDT",
    "STXX/USDT",
]
PREMARKET = ["SPCX/USDT", "OPENAI/USDT", "QNTX/USDT", "ANTHROPIC/USDT"]
_NON_EQUITY = set(PRECIOUS + ENERGY_INDUSTRIAL + ETF_INDEX + PREMARKET)
EQUITY = [
    symbol
    for symbol in BINANCE_TRADFI_PERP_RESEARCH_SYMBOLS_SLASHED
    if symbol not in _NON_EQUITY and symbol not in CRYPTO10
]
TRADFI_LIQUID20 = ETF_INDEX + PRECIOUS + ENERGY_INDUSTRIAL
TRADFI_ALL = ETF_INDEX + PRECIOUS + ENERGY_INDUSTRIAL + EQUITY + PREMARKET

_TAGS = ["research_only", "preregistered", "cost_realistic_required", "claude_lane"]
_UNIVERSE_NOTE = (
    "frozen smoke snapshot; data-PC must replace with point-in-time membership receipts"
)


# Evidence registry rows shared VERBATIM with the sibling lane
# (configs/research/named_quant_crypto_tradfi_suite_v1.json). No new source is
# minted here: every ``hypothesis_refs`` entry below resolves to one of these
# ``source_id``s (regression-tested). Rows added by this lane were confirmed
# against their primary source by the maintainer before being minted.
_SHARED_EVIDENCE_SOURCES: list[dict[str, Any]] = json.loads(
    """
[
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/systrader79_public_post.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Adjacent verified public author source; not assumed identical to requested systrader32."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "systrader79_public_post",
        "update_cadence": "static literature or public educational reference",
        "url": "https://stock79.tistory.com/entry/systrader79-%ED%8A%B8%EB%A0%88%EC%9D%B4%EB%94%A9-%EB%A7%88%EC%8A%A4%ED%84%B0%ED%81%B4%EB%9E%98%EC%8A%A4-%ED%8C%A8%ED%82%A4%EC%A7%80-%EC%98%A4%ED%94%88"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/brock_technical_rules_1992.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Primary moving-average and breakout-rule evidence."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "brock_technical_rules_1992",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1111/j.1540-6261.1992.tb04385.x"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/moreira_muir_2017.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Primary volatility-managed portfolio reference."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "moreira_muir_2017",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1111/jofi.12513"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/multanchanbap_coin_preview.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Public preview documents 120-day filter, 20/10 Turtle rule and ATR stop."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "multanchanbap_coin_preview",
        "update_cadence": "static literature or public educational reference",
        "url": "https://resource.newsystock.com/Admin/Academy/%28Preview%29_MultanChanbab_coin_strategy.pdf"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/multanchanbap_basic_preview.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Public preview documents IBS below 0.3, medium trend and bearish prior bar."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "multanchanbap_basic_preview",
        "update_cadence": "static literature or public educational reference",
        "url": "https://resource.newsystock.com/Admin/Academy/%28Preview%29%20Developing%20the%20basic%20strategy%20of%20Multanchanbap%2816%29.pdf"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/amateurquant_profile.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Public profile supports microstructure, pairs and factor research scope only."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "amateurquant_profile",
        "update_cadence": "static literature or public educational reference",
        "url": "https://insightcampus.co.kr/teachers/%EC%A1%B0%EC%84%B1%ED%98%84-%EA%B0%95%EC%82%AC%EB%8B%98/"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/flightf_rsi_divergence_20210124.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "First-person educational example supports 10-minute RSI divergence and staged RSI exits; pivot and sizing rules remain undisclosed."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "flightf_rsi_divergence_20210124",
        "update_cadence": "static literature or public educational reference",
        "url": "https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=548338"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/flightf_trading_principles_20200908.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "First-person post separates range and one-way regimes and emphasizes stops, volume and multiple timeframes."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "flightf_trading_principles_20200908",
        "update_cadence": "static literature or public educational reference",
        "url": "https://gall.dcinside.com/mgallery/board/view/?id=electronicmoney&no=187860"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "diagnostic_only",
        "cache_path": "var/cache/external_sources/dacapogo_public_repo_633ba5d.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Secondary reproducible research notebook at commit 633ba5d; its trader rules are explicitly proxies.",
            "Pinned commit; diagnostic/proxy provenance only (comparison-proxy documents and ledger analysis), NOT rule evidence."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "dacapogo_public_repo_633ba5d",
        "update_cadence": "static literature or public educational reference",
        "url": "https://github.com/HokyoungJung/dacapogo/tree/633ba5d6bc0c84a20696af6b2bf807cf55d21248"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/aoa_bitmex_interview_20250611.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Official first-person interview supports liquid-major preference and portfolio risk limits, not an entry formula."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "aoa_bitmex_interview_20250611",
        "update_cadence": "static literature or public educational reference",
        "url": "https://www.bitmex.com/blog/whale-trader-talks-aoa"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/dolpago_dogdrip_20210802.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "First-person competition certification; no complete trading formula is disclosed."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "dolpago_dogdrip_20210802",
        "update_cadence": "static literature or public educational reference",
        "url": "https://www.dogdrip.net/341084283"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/albatross_risk_governance.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Publisher page supports rules and capital-management inspiration, not a concrete alpha rule."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "albatross_risk_governance",
        "update_cadence": "static literature or public educational reference",
        "url": "https://www.yes24.com/Product/Goods/141814145"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/lopez_de_prado_hrp_2016.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "HRP reference; allocator diversifies risk and does not forecast alpha."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "lopez_de_prado_hrp_2016",
        "update_cadence": "static literature or public educational reference",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/herc_raffinot_2018.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Primary HERC paper; no full dendrogram HERC optimizer is implemented in this repository."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "herc_raffinot_2018",
        "update_cadence": "static literature or public educational reference",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/nco_lopez_de_prado.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Primary NCO paper; no NCO optimizer is implemented in this repository."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "nco_lopez_de_prado",
        "update_cadence": "static literature or public educational reference",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/constrained_hrp_wp14_2019.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Constrained HRP working paper; current code only reuses per-sleeve caps and quality gates."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "constrained_hrp_wp14_2019",
        "update_cadence": "static literature or public educational reference",
        "url": "https://www.ekon.sun.ac.za/wpapers/2019/wp142019/wp142019.pdf"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/network_risk_parity_2024.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Primary network-risk-parity reference; graph allocation remains diagnostic-only here."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "network_risk_parity_2024",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1057/s41260-023-00347-8"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/faith_way_of_turtle_2007.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Curtis Faith, Way of the Turtle (McGraw-Hill 2007): published Turtle unit sizing (risk% / N), 1/2N pyramiding and 2N stop; parameter values used here are this lane's independent choices."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "faith_way_of_turtle_2007",
        "update_cadence": "static literature or public educational reference",
        "url": "https://www.mheducation.com/highered/mhp/product/way-turtle-secret-methods-turned-ordinary-people-into-legendary-traders.html"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/avellaneda_lee_2010.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Avellaneda & Lee (2010) Quantitative Finance: PCA eigenportfolio residual s-score stat-arb; scope evidence only, thresholds are the paper's published defaults reused as hypothesis values."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "avellaneda_lee_2010",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1080/14697680903124632"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/chan_algorithmic_trading_2013.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "E. Chan, Algorithmic Trading (Wiley 2013): Kalman-filter dynamic hedge ratio for pairs; independent 2-state implementation."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "chan_algorithmic_trading_2013",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1002/9781118676998"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/triantafyllopoulos_montana_2011.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Triantafyllopoulos & Montana (2011) Computational Management Science: dynamic-hedging pairs trading with time-varying regression (Kalman) -- supporting evidence for the state-space hedge ratio."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "triantafyllopoulos_montana_2011",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1007/s10287-009-0105-8"
    },
    {
        "access": "free_unauthenticated",
        "allowed_usage_label": "strategy_class_evidence",
        "cache_path": "var/cache/external_sources/blanchet_chen_zhou_2022_wasserstein_mv.html",
        "credential_requirement": "none",
        "fallback_behavior": "fail_closed",
        "license_note": "Public reference/link only; do not redistribute source content.",
        "notes": [
            "Blanchet, Chen & Zhou (2022) Management Science: exact type-2 Wasserstein DRO mean-variance (p=q=2) -- the formulation implemented by wasserstein_dro_weights."
        ],
        "release_lag_policy": "evidence only; no market-time feature is consumed",
        "source_id": "blanchet_chen_zhou_2022_wasserstein_mv",
        "update_cadence": "static literature or public educational reference",
        "url": "https://doi.org/10.1287/mnsc.2021.4155"
    }
]
"""
)
_EVIDENCE_IDS = {row["source_id"] for row in _SHARED_EVIDENCE_SOURCES}
_KALMAN_REFS = [
    "chan_algorithmic_trading_2013",
    "triantafyllopoulos_montana_2011",
    "amateurquant_profile",
]
_PCA_REFS = ["avellaneda_lee_2010", "amateurquant_profile"]


def _default_params(strategy_class: str) -> dict[str, Any]:
    cls = resolve_strategy_class(strategy_class, strict=True)
    schema = cls.get_param_schema()
    return {name: param.default for name, param in schema.items()}


def _candidate(
    *,
    candidate_id: str,
    name: str,
    family: str,
    strategy_class: str,
    timeframe: str,
    symbols: list[str],
    overrides: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    hypothesis_refs: list[str],
    notes: str,
    admission_route: str | None = None,
) -> dict[str, Any]:
    params = _default_params(strategy_class)
    for key, value in dict(overrides or {}).items():
        if key not in params:
            raise KeyError(f"{strategy_class} has no param {key!r}")
        params[key] = value
    params.update(dict(extra_params or {}))
    unresolved = sorted(set(hypothesis_refs) - _EVIDENCE_IDS)
    if unresolved:
        raise KeyError(f"{candidate_id}: hypothesis_refs not in evidence registry: {unresolved}")
    metadata: dict[str, Any] = {
        "hypothesis_refs": list(hypothesis_refs),
        "promotion_eligible": False,
        "universe_membership": _UNIVERSE_NOTE,
        "lane": "claude",
    }
    if admission_route:
        metadata["admission_route"] = admission_route
    tags = list(_TAGS)
    if len(symbols) > 1:
        tags.append("cross_sectional")
    return {
        "candidate_id": candidate_id,
        "name": name,
        "family": family,
        "strategy_class": strategy_class,
        "strategy": strategy_class,
        "strategy_timeframe": timeframe,
        "timeframe": timeframe,
        "symbols": list(symbols),
        "params": params,
        "notes": notes,
        "tags": tags,
        "metadata": metadata,
    }


def build_candidates() -> list[dict[str, Any]]:
    turtle_child = _default_params("TurtleUnitPyramidingStrategy")
    rotation_child = _default_params("MaScoreVolTargetRotationStrategy")
    rotation_child["max_weight"] = 0.25
    rows = [
        # ---------------- crypto top-10 ----------------
        _candidate(
            candidate_id="crypto_vb_noise_session_1h_v1",
            name="Crypto Session Volatility Breakout (noise K, MA score, vol target)",
            family="session_volatility_breakout",
            strategy_class="NoiseFilteredVolatilityBreakoutStrategy",
            timeframe="1h",
            symbols=CRYPTO10,
            overrides={"max_symbols_by_noise": 5},
            hypothesis_refs=["systrader79_public_post", "brock_technical_rules_1992"],
            notes="UTC-day session open + noise-adaptive K x prev range; MA-score and range vol-target scale size; time-cut exit at next session.",
        ),
        _candidate(
            candidate_id="crypto_ma_score_vol_target_1d_v1",
            name="Crypto MA-Score x Inverse-Vol x Vol-Target Rotation",
            family="ma_score_dynamic_allocation",
            strategy_class="MaScoreVolTargetRotationStrategy",
            timeframe="1d",
            symbols=CRYPTO10,
            hypothesis_refs=["systrader79_public_post", "moreira_muir_2017"],
            notes="Long-only dynamic allocation: fraction of 3/5/10/20d SMAs below price x inverse-vol risk parity x per-bar vol-target clamp.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="crypto_multanchanbap_20_10_public_rule_1d_v1",
            name="Crypto 20d close-high / 10d close-low / -3.5% stop / 120d MA gate (public rule)",
            family="turtle_unit_pyramiding",
            strategy_class="TurtleUnitPyramidingStrategy",
            timeframe="1d",
            symbols=CRYPTO10,
            overrides={
                "channel_source": "close",
                "entry_lookback": 20,
                "exit_lookback": 10,
                "stop_loss_pct": 0.035,
                "trend_ma_window": 120,
                "max_units": 1,
                "use_n_stop": False,
                "allow_short": False,
                "unit_risk_pct": 0.02,
            },
            hypothesis_refs=["multanchanbap_coin_preview", "multanchanbap_basic_preview"],
            notes="Exact public rule set (close-channel 20/10, fixed -3.5% stop, long only above the 120-day MA); one unit, no pyramiding, no N-stop. Sizing (unit_risk_pct) is the author's choice.",
        ),
        _candidate(
            candidate_id="crypto_turtle_unit_pyramid_1d_v1",
            name="Crypto Turtle 55/20 Unit Pyramiding",
            family="turtle_unit_pyramiding",
            strategy_class="TurtleUnitPyramidingStrategy",
            timeframe="1d",
            symbols=CRYPTO10,
            hypothesis_refs=["faith_way_of_turtle_2007", "multanchanbap_coin_preview"],
            notes="Published Turtle rule shape (Faith 2007): risk%/N unit sizing, +0.5N adds, whole-position 2N stop from the last fill, 55/20 channels; the numeric values (1% risk, 4 units, caps) and the 물탄찬밥 pyramiding framing are this lane's independent choices, not quoted from either source.",
        ),
        _candidate(
            candidate_id="crypto_kalman_pair_eth_btc_1h_v1",
            name="Kalman Pair ETH/BTC (1h)",
            family="kalman_pairs_stat_arb",
            strategy_class="KalmanPairsStatArbStrategy",
            timeframe="1h",
            symbols=["ETH/USDT", "BTC/USDT"],
            overrides={"symbol_y": "ETH/USDT", "symbol_x": "BTC/USDT"},
            hypothesis_refs=_KALMAN_REFS,
            notes="2-state Kalman hedge on log prices, standardized innovation z entry/exit, ADF gate, half-life hold cap.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="crypto_kalman_pair_sol_avax_4h_v1",
            name="Kalman Pair SOL/AVAX (4h)",
            family="kalman_pairs_stat_arb",
            strategy_class="KalmanPairsStatArbStrategy",
            timeframe="4h",
            symbols=["SOL/USDT", "AVAX/USDT"],
            overrides={"symbol_y": "SOL/USDT", "symbol_x": "AVAX/USDT", "max_hold_bars": 60},
            hypothesis_refs=_KALMAN_REFS,
            notes="Same engine on an L1 pair at 4h.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="crypto_pca_residual_statarb_1d_v1",
            name="Crypto PCA Residual s-score Stat-Arb (k=1)",
            family="pca_residual_stat_arb",
            strategy_class="PcaResidualStatArbStrategy",
            timeframe="1d",
            symbols=CRYPTO10,
            overrides={"n_factors": 1, "min_symbols": 6},
            hypothesis_refs=_PCA_REFS,
            notes="Eigenportfolio residual OU s-scores; open |s|>1.25, close 0.5/0.75; dollar-neutral caps.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="crypto_rsi_divergence_btc_eth_10m_v1",
            name="RSI(11) Divergence (FlightF proxy) BTC/ETH 10m",
            family="rsi_divergence_flight",
            strategy_class="RsiDivergenceScaleOutStrategy",
            timeframe="10m",
            symbols=["BTC/USDT", "ETH/USDT"],
            overrides={
                "oversold": 20.0,
                "overbought": 80.0,
                "exit_rsi_first": 45.0,
                "exit_rsi_second": 60.0,
                "first_exit_fraction": 0.6,
                "max_hold_bars": 72,
                "require_htf_confirmation": True,
                "htf_multiple": 6,
                "htf_ma_window": 20,
                "opposing_volume_multiple": 1.5,
            },
            hypothesis_refs=[
                "flightf_rsi_divergence_20210124",
                "flightf_trading_principles_20200908",
                "dacapogo_public_repo_633ba5d",
            ],
            notes="Audited public rule: 10m BTC futures, RSI<20 divergence; staged exit = 60% of the position (EXIT with metadata.exit_fraction=0.6, 'more than half' in the source) at RSI 45, the remaining 40% as a full EXIT at RSI 60; opposing-volume invalidation; 1h (6x10m) trend confirmation as the higher-timeframe proxy. Signals fire on bar close and fill at the next bar open (engine contract).",
        ),
        _candidate(
            candidate_id="crypto_prev_day_box_15m_v1",
            name="Prev-Day Box Quartile Reversion (independent proxy) BTC/ETH 15m",
            family="prev_day_box_quartile",
            strategy_class="PrevDayBoxQuartileReversionStrategy",
            timeframe="15m",
            symbols=["BTC/USDT", "ETH/USDT"],
            hypothesis_refs=["dacapogo_public_repo_633ba5d"],
            notes="INDEPENDENT comparison-proxy rule set (previous UTC-day box, 25/50/75 levels, wick>=body rebound, volume>prev-day median, TP mid / SL box end / flat at day end) as documented in the dacapogo repo; the AOA interview states no box/quartile rule, so it is deliberately NOT cited as evidence.",
        ),
        _candidate(
            candidate_id="crypto_session_high_scalp_1m_v1",
            name="Session-High Breakout Scalp (Dolpago-observed proxy) 1m",
            family="session_high_breakout_scalp",
            strategy_class="SessionHighBreakoutScalpStrategy",
            timeframe="1m",
            symbols=CRYPTO10,
            overrides={
                "min_session_bars": 30,
                "surge_bars": 5,
                "max_hold_bars": 5,
                "max_symbols_by_turnover": 5,
            },
            hypothesis_refs=["dolpago_dogdrip_20210802", "dacapogo_public_repo_633ba5d"],
            notes="Bar proxy of tape behaviour: prior-session-high break + volume surge in the first 4h of the UTC session, TP 1.5% / SL 0.7% / 5-bar time stop. Fill/queue realism required before any inference.",
        ),
        _candidate(
            candidate_id="crypto_kill_switch_overlay_turtle_1d_v1",
            name="Equity-Curve Kill-Switch Overlay over Crypto Turtle",
            family="equity_curve_kill_switch_overlay",
            strategy_class="EquityCurveKillSwitchOverlayStrategy",
            timeframe="1d",
            symbols=CRYPTO10,
            overrides={"child_strategy_class": "TurtleUnitPyramidingStrategy"},
            extra_params={"child_params": turtle_child},
            hypothesis_refs=["albatross_risk_governance", "aoa_bitmex_interview_20250611"],
            notes="Proxy-equity drawdown ladder (5/10/15/20% -> 0.75/0.5/0.25/0), consecutive-loss halving from the 3rd loss, 10% monthly loss kill, re-risk hysteresis.",
        ),
        # ---------------- TradFi perps ----------------
        _candidate(
            candidate_id="tradfi_vb_noise_session_1h_v1",
            name="TradFi Session Volatility Breakout (ETF/metals/energy)",
            family="session_volatility_breakout",
            strategy_class="NoiseFilteredVolatilityBreakoutStrategy",
            timeframe="1h",
            symbols=TRADFI_LIQUID20,
            overrides={"max_symbols_by_noise": 8},
            hypothesis_refs=["systrader79_public_post", "brock_technical_rules_1992"],
            notes="Same engine on 20 liquid TradFi perps; UTC session is a proxy for the underlying's cash session.",
        ),
        _candidate(
            candidate_id="tradfi_ma_score_vol_target_1d_v1",
            name="TradFi MA-Score Rotation (ETF/metals/energy)",
            family="ma_score_dynamic_allocation",
            strategy_class="MaScoreVolTargetRotationStrategy",
            timeframe="1d",
            symbols=TRADFI_LIQUID20,
            overrides={"max_weight": 0.25},
            hypothesis_refs=["systrader79_public_post", "moreira_muir_2017"],
            notes="Long-only tactical allocation across index ETFs, precious metals and energy.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_multanchanbap_20_10_public_rule_1d_v1",
            name="TradFi 20d close-high / 10d close-low / -3.5% stop / 120d MA gate (public rule)",
            family="turtle_unit_pyramiding",
            strategy_class="TurtleUnitPyramidingStrategy",
            timeframe="1d",
            symbols=TRADFI_LIQUID20,
            overrides={
                "channel_source": "close",
                "entry_lookback": 20,
                "exit_lookback": 10,
                "stop_loss_pct": 0.035,
                "trend_ma_window": 120,
                "max_units": 1,
                "use_n_stop": False,
                "allow_short": False,
                "unit_risk_pct": 0.02,
            },
            hypothesis_refs=["multanchanbap_coin_preview", "multanchanbap_basic_preview"],
            notes="Hypothesis transfer of the exact public rule set to ETF/metal/energy perps.",
        ),
        _candidate(
            candidate_id="tradfi_turtle_unit_pyramid_1d_v1",
            name="TradFi Turtle 55/20 Unit Pyramiding",
            family="turtle_unit_pyramiding",
            strategy_class="TurtleUnitPyramidingStrategy",
            timeframe="1d",
            symbols=TRADFI_LIQUID20,
            hypothesis_refs=["faith_way_of_turtle_2007", "multanchanbap_coin_preview"],
            notes="Classic Turtle universe analogue (metals, energy, index ETFs); same rule shape as the crypto sleeve, values are independent choices.",
        ),
        _candidate(
            candidate_id="tradfi_kalman_pair_qqq_spy_1h_v1",
            name="Kalman Pair QQQ/SPY (1h)",
            family="kalman_pairs_stat_arb",
            strategy_class="KalmanPairsStatArbStrategy",
            timeframe="1h",
            symbols=["QQQ/USDT", "SPY/USDT"],
            overrides={"symbol_y": "QQQ/USDT", "symbol_x": "SPY/USDT"},
            hypothesis_refs=_KALMAN_REFS,
            notes="Index-ETF pair with drifting beta.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_kalman_pair_ewy_ewt_4h_v1",
            name="Kalman Pair EWY/EWT (4h)",
            family="kalman_pairs_stat_arb",
            strategy_class="KalmanPairsStatArbStrategy",
            timeframe="4h",
            symbols=["EWY/USDT", "EWT/USDT"],
            overrides={"symbol_y": "EWY/USDT", "symbol_x": "EWT/USDT", "max_hold_bars": 60},
            hypothesis_refs=_KALMAN_REFS,
            notes="Korea vs Taiwan country-ETF pair (semiconductor-heavy both).",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_kalman_pair_nvda_amd_1h_v1",
            name="Kalman Pair NVDA/AMD (1h)",
            family="kalman_pairs_stat_arb",
            strategy_class="KalmanPairsStatArbStrategy",
            timeframe="1h",
            symbols=["NVDA/USDT", "AMD/USDT"],
            overrides={"symbol_y": "NVDA/USDT", "symbol_x": "AMD/USDT"},
            hypothesis_refs=_KALMAN_REFS,
            notes="Single-stock sector pair.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_pca_residual_statarb_equity_1d_v1",
            name="TradFi Equity PCA Residual Stat-Arb (k=3)",
            family="pca_residual_stat_arb",
            strategy_class="PcaResidualStatArbStrategy",
            timeframe="1d",
            symbols=EQUITY,
            overrides={
                "n_factors": 3,
                "min_symbols": 20,
                "max_longs": 8,
                "max_shorts": 8,
                "gross_cap": 0.8,
                "max_position_allocation": 0.1,
            },
            hypothesis_refs=_PCA_REFS,
            notes="Avellaneda-Lee on the single-stock perp book; 3 eigenportfolios.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_pca_residual_statarb_etf_cmdty_1d_v1",
            name="TradFi ETF+Commodity PCA Residual Stat-Arb (k=2)",
            family="pca_residual_stat_arb",
            strategy_class="PcaResidualStatArbStrategy",
            timeframe="1d",
            symbols=TRADFI_LIQUID20,
            overrides={"n_factors": 2, "min_symbols": 10, "max_longs": 4, "max_shorts": 4},
            hypothesis_refs=_PCA_REFS,
            notes="Cross-asset residual reversion among index ETFs, metals and energy.",
            admission_route="allow_multi_asset_handoff",
        ),
        _candidate(
            candidate_id="tradfi_rsi_divergence_xau_cl_10m_v1",
            name="RSI(11) Divergence (FlightF proxy) XAU/CL 10m",
            family="rsi_divergence_flight",
            strategy_class="RsiDivergenceScaleOutStrategy",
            timeframe="10m",
            symbols=["XAU/USDT", "CL/USDT"],
            overrides={
                "oversold": 20.0,
                "overbought": 80.0,
                "first_exit_fraction": 0.6,
                "max_hold_bars": 72,
                "require_htf_confirmation": True,
                "htf_multiple": 6,
                "htf_ma_window": 20,
                "opposing_volume_multiple": 1.5,
            },
            hypothesis_refs=[
                "flightf_rsi_divergence_20210124",
                "flightf_trading_principles_20200908",
                "dacapogo_public_repo_633ba5d",
            ],
            notes="Hypothesis transfer of the BTC-futures RSI method to gold and crude perps at the same 10m cadence (60% exit at RSI 45, remainder at 60).",
        ),
        _candidate(
            candidate_id="tradfi_prev_day_box_15m_v1",
            name="Prev-Day Box Quartile Reversion (independent proxy) XAU/SPY/QQQ 15m",
            family="prev_day_box_quartile",
            strategy_class="PrevDayBoxQuartileReversionStrategy",
            timeframe="15m",
            symbols=["XAU/USDT", "SPY/USDT", "QQQ/USDT"],
            hypothesis_refs=["dacapogo_public_repo_633ba5d"],
            notes="Same independent comparison-proxy rule set on macro perps; UTC-day box; no attribution to any trader's stated rule.",
        ),
        _candidate(
            candidate_id="tradfi_session_high_scalp_1m_v1",
            name="Session-High Breakout Scalp (Dolpago-observed proxy) equities 1m",
            family="session_high_breakout_scalp",
            strategy_class="SessionHighBreakoutScalpStrategy",
            timeframe="1m",
            symbols=EQUITY,
            overrides={
                "min_session_bars": 30,
                "surge_bars": 5,
                "max_hold_bars": 5,
                "max_symbols_by_turnover": 10,
                "entry_start_minute": 810,
                "entry_end_minute": 990,
            },
            hypothesis_refs=["dolpago_dogdrip_20210802", "dacapogo_public_repo_633ba5d"],
            notes="Entry window 13:30-16:30 UTC (US cash open) as the morning-concentration analogue; top-10 turnover names of the previous session.",
        ),
        _candidate(
            candidate_id="tradfi_kill_switch_overlay_ma_rotation_1d_v1",
            name="Equity-Curve Kill-Switch Overlay over TradFi MA-Score Rotation",
            family="equity_curve_kill_switch_overlay",
            strategy_class="EquityCurveKillSwitchOverlayStrategy",
            timeframe="1d",
            symbols=TRADFI_LIQUID20,
            overrides={
                "child_strategy_class": "MaScoreVolTargetRotationStrategy",
                "equity_ma_window": 60,
            },
            extra_params={"child_params": rotation_child},
            hypothesis_refs=["albatross_risk_governance"],
            notes="Adds the 60-bar equity-curve MA filter on top of the drawdown ladder.",
            admission_route="allow_multi_asset_handoff",
        ),
        # ---------------- cross-asset ----------------
        _candidate(
            candidate_id="xasset_ma_score_vol_target_1d_v1",
            name="Cross-Asset MA-Score Rotation (crypto10 + metals/energy + index ETFs)",
            family="ma_score_dynamic_allocation",
            strategy_class="MaScoreVolTargetRotationStrategy",
            timeframe="1d",
            symbols=CRYPTO10 + PRECIOUS + ENERGY_INDUSTRIAL + ETF_INDEX,
            overrides={"max_weight": 0.15, "min_symbols": 5},
            hypothesis_refs=["systrader79_public_post", "moreira_muir_2017"],
            notes="30-asset tactical book; the timing overlay for the asset-level hierarchical allocation study.",
            admission_route="allow_multi_asset_handoff",
        ),
    ]
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    return rows


def _sleeves(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in candidates:
        out[row["candidate_id"]] = {
            "family": row["family"],
            "returns": None,
            "turnover": None,
            "source_artifact_id": "named_quant_claude_data_pc_walkforward",
            "returns_source": {
                "artifact": "named_quant_claude_data_pc_walkforward",
                "candidate_id": row["candidate_id"],
                "stream": (
                    "common-date aligned train+validation NET returns only; the locked-OOS "
                    "segment is NEVER an allocator/quality-gate input (reported separately)"
                ),
                "selection_inputs": ["train", "validation"],
                "locked_oos_used_for_weights": False,
                "turnover_source": (
                    "same candidate, train+validation folds, after fees, funding, slippage "
                    "and impact"
                ),
            },
        }
    return out


def build_suite() -> dict[str, Any]:
    candidates = build_candidates()
    families = sorted({row["family"] for row in candidates})
    return {
        "schema_version": 1,
        "suite_id": "named_quant_claude_v1",
        "created_utc": CREATED_UTC,
        "kind": "preregistered_allocation_cell",
        "artifact_kind": "preregistered_research_suite",
        "status": "hypotheses_pending_data_pc_backtest",
        "research_only": True,
        "allow_real_money": False,
        "promotion_eligible": False,
        "lane": "claude",
        "sibling_lane": "configs/research/named_quant_crypto_tradfi_suite_v1.json",
        "attribution_policy": (
            "Independent adaptations of publicly described rules (systrader79, 물탄찬밥, "
            "아마추어퀀트, 알바트로스, FlightF, 워뇨띠/AOA, 돌파고 as documented in the "
            "dacapogo repo); not a reproduction, endorsement or performance claim."
        ),
        "candidate_research": {
            "research": {
                "cost_rate_multiplier": 2.2,
                "dsr_gate_floor": 0.9,
                "hac_inference": True,
                "spa_gate_ceiling": 0.05,
                "strict_selection_gate": True,
                # Read defensively by the runner: makes the runner execute these
                # registered classes instead of the generic proxy once the honest
                # profile / schema carries the flag.
                "route_unmapped_registered_strategies": True,
                "emit_candidate_overfit_stats": True,
            },
            "shortlist_selection": {
                "allow_multi_asset": True,
                "drop_single_without_metrics": True,
                "include_weights": False,
                "max_per_family": 2,
                "max_per_timeframe": 10,
                "single_min_return": 0.0,
                "single_min_sharpe": 0.35,
                "single_min_trades": 5,
            },
        },
        "candidates": candidates,
        "evidence_sources": _SHARED_EVIDENCE_SOURCES,
        "allocator_evidence_refs": {
            "hrp_dendrogram": ["lopez_de_prado_hrp_2016"],
            "constrained_hrp": ["lopez_de_prado_hrp_2016", "constrained_hrp_wp14_2019"],
            "herc": ["herc_raffinot_2018"],
            "nco": ["nco_lopez_de_prado"],
            "wasserstein_dro": ["blanchet_chen_zhou_2022_wasserstein_mv"],
            "graph_inverse_centrality": ["network_risk_parity_2024"],
        },
        # ---- allocation cell (validated by --validate-cell-spec) ----
        "cell_id": "named_quant_claude_hier_cell_v1",
        "method": "hrp_dendrogram",
        "allocator": {
            "method": "hrp_dendrogram",
            "min_sleeves": 6,
            "min_families": 5,
            "upper": 0.25,
            "gross_cap": 1.0,
            "notes": "Full Lopez de Prado dendrogram HRP (single linkage) is the pre-registered primary; the variants below are run on the SAME materialized input for comparison, never cherry-picked.",
        },
        "allocator_params": {"linkage_method": "single"},
        "allocator_variants_execution": (
            "build_quality_gated_allocation.py emits ONE manifest per run (top-level method + "
            "allocator_params). The variants below are executed by "
            "scripts/research/compare_hierarchical_allocators.py --variants on the same "
            "materialized input; every row is reported, none is cherry-picked."
        ),
        "allocator_variants": [
            {"method": "hrp_dendrogram", "allocator_params": {"linkage_method": "single"}},
            {"method": "hrp_dendrogram", "allocator_params": {"linkage_method": "ward"}},
            {
                "method": "constrained_hrp",
                "allocator_params": {"linkage_method": "single", "lower": 0.0, "upper_bound": 0.2},
            },
            {"method": "herc", "allocator_params": {"linkage_method": "ward"}},
            {"method": "nco", "allocator_params": {"use_mean": False}},
            # BCZ radius is in squared-return units: 1e-5 ~ (0.32% per bar)^2 ambiguity
            # on a daily NET-return panel; target_return is per bar.
            {"method": "wasserstein_dro", "allocator_params": {"radius": 1e-5}},
            {
                "method": "wasserstein_dro",
                "allocator_params": {"radius": 1e-4, "target_return": 0.0002},
            },
            {"method": "graph_inverse_centrality", "allocator_params": {"floor": 1e-6}},
            {"method": "hrp", "allocator_params": {}},
            {"method": "erc", "allocator_params": {}},
        ],
        "allocation_input_policy": {
            "weights_and_quality_gate_inputs": ["train", "validation"],
            "locked_oos_used_for_weights": False,
            "locked_oos_role": "report-only out-of-sample evaluation of the pre-registered weights",
            "alignment": "common-date intersection of NET returns before any allocator call",
        },
        "upper": 0.25,
        "min_sleeves": 6,
        "gross_cap": 1.0,
        "membership_is_preregistered_weights_are_measured": True,
        "sleeves": _sleeves(candidates),
        "source_artifacts": [
            {
                "id": "named_quant_claude_data_pc_walkforward",
                "path": "data-PC materializes and records path here",
                "sha256": None,
                "max_age_hours": 8760,
                "ready": False,
                "portfolio_ready": False,
                "note": "Set ready only after point-in-time universe, exact common-date alignment and locked-OOS net-return checks pass.",
            }
        ],
        "portfolio_recipes": [
            {"name": "equal_weight_baseline", "method": "equal_weight"},
            {"name": "inverse_vol_baseline", "method": "inverse_vol"},
            {"name": "erc", "method": "erc"},
            {"name": "hrp_threshold_legacy", "method": "hrp"},
            {"name": "hrp_dendrogram_single", "method": "hrp_dendrogram"},
            {"name": "constrained_hrp_cap20", "method": "constrained_hrp"},
            {"name": "herc_ward_silhouette", "method": "herc"},
            {"name": "nco_min_variance", "method": "nco"},
            {"name": "wasserstein_dro_bcz", "method": "wasserstein_dro"},
            {"name": "graph_inverse_centrality_heuristic", "method": "graph_inverse_centrality"},
        ],
        "asset_level_allocation_study": {
            "universe": "crypto10 + tradfi groups (precious_metals, energy_industrial, etf_index, equity)",
            "input": "daily NET log returns per asset, common-date aligned, fold-locked",
            "methods": [
                "hrp_dendrogram",
                "constrained_hrp",
                "herc",
                "nco",
                "wasserstein_dro",
                "graph_inverse_centrality",
            ],
            "compare_script": "scripts/research/compare_hierarchical_allocators.py",
            "rebalance": ["weekly", "monthly"],
            "notes": "Same CLI as sleeves (sleeves = assets). Report effN, diversification ratio, turnover and fold stability per method; no single winner is pre-declared.",
        },
        "families": families,
        "universe": {
            "crypto_top10": {
                "selection_rule": "point_in_time_market_cap_top10_intersect_binance_usdm_TRADING_perpetuals",
                "static_smoke_symbols": CRYPTO10,
                "snapshot_is_market_cap_proof": False,
            },
            "tradfi": {
                "selection_rule": "all Binance USD-M TRADIFI_PERPETUAL contracts with status TRADING at each decision date",
                "groups": {
                    "precious_metals": PRECIOUS,
                    "energy_industrial": ENERGY_INDUSTRIAL,
                    "etf_index": ETF_INDEX,
                    "equity": EQUITY,
                    "premarket": PREMARKET,
                },
                "static_smoke_symbols": TRADFI_ALL,
            },
        },
        "required_data_contracts": [
            "timestamped OHLCV and next-open execution",
            "mark/index price, funding rate and actual settlement timestamp per symbol",
            "maker/taker fee, spread, slippage, sqrt-impact and liquidation simulation",
            "1s/1m bars with per-bar volume for the scalp candidates; queue-position model before any inference",
            "point-in-time market-cap and listing/delisting membership receipts",
            "common-date aligned net sleeve return streams before allocation",
        ],
        "source_resolution": [
            {
                "requested_label": "systrader32",
                "status": "unverified",
                "action": "no attribution; not mapped to any other handle (systrader79 material below is cited on its own, adjacent public source only)",
            },
            {
                "requested_label": "systrader79",
                "status": "public_blog_and_books",
                "action": "volatility breakout / noise / MA score / vol control rules",
            },
            {
                "requested_label": "물탄찬밥",
                "status": "verified_public_preview",
                "action": "exact public rule candidate (20d close-high entry / 10d close-low exit / -3.5% stop / 120d MA gate: *_multanchanbap_20_10_public_rule_1d_v1) plus the Turtle unit-sizing/pyramiding extension; IBS variant lives in the sibling lane",
            },
            {
                "requested_label": "아마추어퀀트",
                "status": "public_profile",
                "action": "pairs / stat-arb research scope only",
            },
            {
                "requested_label": "알바트로스",
                "status": "verified_as_성필규",
                "action": "capital-management / kill-switch inspiration only",
            },
            {
                "requested_label": "부동심",
                "status": "unverified",
                "action": "no attribution of any strategy or principle; no public reproducible rules found",
            },
            {
                "requested_label": "FlightF",
                "status": "primary_posts_catalogued_in_dacapogo",
                "action": "RSI divergence proxy",
            },
            {
                "requested_label": "워뇨띠/AOA",
                "status": "bitmex_interview_catalogued_in_dacapogo",
                "action": "prev-day box quartile proxy",
            },
            {
                "requested_label": "돌파고",
                "status": "ledger_observations_only",
                "action": "session-high breakout scalp proxy; formula not reproduced",
            },
        ],
    }


def main() -> int:
    payload = build_suite()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT} ({len(payload['candidates'])} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
