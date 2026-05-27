"""Batch strict leverage retune for grouped allocator candidates using shared research resources."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.util
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.portfolio_split_contract import FOLLOWUP_ROOT
from lumina_quant.strategy_factory.runtime_settings import current_research_market_data_settings
from lumina_quant.timeframe_aggregator import resample_1s_frame

_validation_spec = importlib.util.spec_from_file_location(
    "run_grouped_allocator_strict_leverage_validation",
    Path(__file__).resolve().parent / "run_grouped_allocator_strict_leverage_validation.py",
)
if _validation_spec is None or _validation_spec.loader is None:
    raise RuntimeError("Failed to load strict leverage validation helpers")
_validation = importlib.util.module_from_spec(_validation_spec)
sys.modules[_validation_spec.name] = _validation
_validation_spec.loader.exec_module(_validation)
_VALIDATION_CORE = _validation._validation

_runner_spec = importlib.util.spec_from_file_location(
    "lumina_quant.strategy_factory.research_runner",
    Path(__file__).resolve().parents[2]
    / "src"
    / "lumina_quant"
    / "strategy_factory"
    / "research_runner.py",
)
if _runner_spec is None or _runner_spec.loader is None:
    raise RuntimeError("Failed to load research runner helpers")
_runner = importlib.util.module_from_spec(_runner_spec)
sys.modules[_runner_spec.name] = _runner
_runner_spec.loader.exec_module(_runner)

DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_allocator_strict_leverage_batch_retune_current"
)
DEFAULT_SHARED_RESOURCE_CACHE_DIR = (
    Path(__file__).resolve().parents[2]
    / "var"
    / "cache"
    / "grouped_allocator_strict_leverage_shared_resources"
)
SHARED_RESOURCE_CACHE_VERSION = "v1"
_FEATURE_SUPPORT_STRATEGY_CLASSES = {
    "PerpCrowdingCarryStrategy",
    "CompositeTrendStrategy",
}


@dataclass(slots=True)
class SharedStrictResources:
    data_sources: dict[str, list[str]]
    cache: dict[tuple[str, str], Any]
    feature_cache: dict[str, Any]
    benchmark: dict[str, Any]


@dataclass(slots=True)
class SharedStrictResearchContext:
    base_tf: str
    normalized_timeframes: list[str]
    universe: list[str]
    resolved_split: dict[str, Any]
    scoring: Any
    resources: SharedStrictResources
    source_candidates: list[dict[str, Any]]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value, key=str)]
    return value


def _shared_feature_symbols(candidates: list[dict[str, Any]]) -> list[str]:
    symbols: set[str] = set()
    for row in candidates:
        strategy_class = str(row.get("strategy_class") or row.get("strategy") or "")
        if strategy_class not in _FEATURE_SUPPORT_STRATEGY_CLASSES:
            continue
        for symbol in list(row.get("symbols") or []):
            token = str(symbol).strip()
            if token:
                symbols.add(token)
    return sorted(symbols)


def _candidate_universe(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(symbol).strip()
            for row in candidates
            for symbol in list(row.get("symbols") or [])
            if str(symbol).strip()
        }
    )


def _candidate_timeframes(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip().lower()
            for row in candidates
            if str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip()
        }
    )


def _shared_resource_cache_key_payload(
    *,
    normalized_timeframes: list[str],
    universe: list[str],
    resolved_split: dict[str, Any],
    candidates: list[dict[str, Any]],
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cache_version": SHARED_RESOURCE_CACHE_VERSION,
        "normalized_timeframes": list(normalized_timeframes),
        "universe": list(universe),
        "feature_symbols": _shared_feature_symbols(candidates),
        "resolved_split": _json_ready(dict(resolved_split)),
        "data_mode": str(data_mode or "legacy"),
        "allow_csv_fallback": bool(allow_csv_fallback),
        "allow_synthetic_fallback": bool(allow_synthetic_fallback),
        "min_bundle_bars": max(1, int(min_bundle_bars)),
        "market_data_settings": _json_ready(dict(market_data_settings or {})),
    }


def _shared_resource_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _shared_resource_cache_paths(*, cache_dir: Path, cache_key: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{cache_key}.pkl",
        cache_dir / f"{cache_key}.meta.json",
    )


def _load_shared_resource_cache(
    *,
    cache_dir: Path,
    cache_key: str,
) -> SharedStrictResources | None:
    cache_path, _meta_path = _shared_resource_cache_paths(cache_dir=cache_dir, cache_key=cache_key)
    if not cache_path.exists():
        return None
    try:
        payload = pickle.loads(cache_path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    cache = payload.get("cache")
    data_sources = payload.get("data_sources")
    feature_cache = payload.get("feature_cache")
    benchmark = payload.get("benchmark")
    if (
        not isinstance(cache, dict)
        or not isinstance(data_sources, dict)
        or not isinstance(feature_cache, dict)
        or not isinstance(benchmark, dict)
    ):
        return None
    return SharedStrictResources(
        data_sources=_copy_data_sources(data_sources),
        cache=cache,
        feature_cache=feature_cache,
        benchmark=benchmark,
    )


def _write_shared_resource_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    cache_key_payload: dict[str, Any],
    resources: SharedStrictResources,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path = _shared_resource_cache_paths(cache_dir=cache_dir, cache_key=cache_key)
    cache_path.write_bytes(
        pickle.dumps(
            {
                "data_sources": _copy_data_sources(resources.data_sources),
                "cache": resources.cache,
                "feature_cache": resources.feature_cache,
                "benchmark": resources.benchmark,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    meta_path.write_text(
        json.dumps(
            {
                "artifact_kind": "grouped_allocator_strict_leverage_shared_resource_cache",
                "generated_at": _utc_now_iso(),
                "cache_key": cache_key,
                "cache_key_payload": _json_ready(cache_key_payload),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _release_process_memory() -> None:
    import gc

    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        return


def _load_legacy_shared_resources_fast(
    *,
    normalized_timeframes: list[str],
    universe: list[str],
    resolved_split: dict[str, Any],
    candidates: list[dict[str, Any]],
    market_data_settings: dict[str, Any],
) -> SharedStrictResources:
    load_start, load_end = _runner._split_window_bounds(resolved_split)
    start_iso = _runner._datetime_to_iso_z(load_start)
    end_iso = _runner._datetime_to_iso_z(load_end)
    parquet_root = str(market_data_settings["parquet_root"])
    exchange = str(market_data_settings["exchange"])
    one_second_frames = load_data_dict_from_parquet(
        parquet_root,
        exchange=exchange,
        symbol_list=list(universe),
        timeframe="1s",
        start_date=start_iso,
        end_date=end_iso,
        data_mode="legacy",
    )

    missing_symbols = [
        symbol
        for symbol in universe
        if symbol not in one_second_frames or one_second_frames[symbol].is_empty()
    ]
    if missing_symbols:
        raise RuntimeError(
            "Missing legacy 1s parquet rows for strict shared preload: "
            + ", ".join(missing_symbols)
        )

    cache: dict[tuple[str, str], Any] = {}
    source_map: dict[str, list[str]] = {"parquet": [], "csv": [], "synthetic": []}
    for symbol in universe:
        frame_1s = one_second_frames[symbol]
        for timeframe in normalized_timeframes:
            frame = (
                frame_1s if timeframe == "1s" else resample_1s_frame(frame_1s, timeframe=timeframe)
            )
            if frame.is_empty():
                raise RuntimeError(
                    f"Legacy resample produced no rows for strict shared preload: {symbol}@{timeframe}"
                )
            cache[(symbol, timeframe)] = _runner._frame_to_bundle(symbol, timeframe, frame)
            source_map["parquet"].append(f"{symbol}@{timeframe}")

    feature_cache = _runner._load_feature_cache(
        symbols=_shared_feature_symbols(candidates),
        start_date=start_iso,
        end_date=end_iso,
        market_data_settings=market_data_settings,
    )
    benchmark = _runner._benchmark_cache(cache, normalized_timeframes)
    return SharedStrictResources(
        data_sources=source_map,
        cache=cache,
        feature_cache=feature_cache,
        benchmark=benchmark,
    )


def _parse_leverage_list(raw: str) -> list[int]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    resolved = sorted({max(1, int(token)) for token in values})
    if not resolved:
        raise ValueError("at least one leverage value is required")
    return resolved


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Grouped Allocator Strict Leverage Batch Retune", ""]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(
                f"- inc {row['incumbent_leverage']}x / auto {row['autoresearch_leverage']}x | ERROR | `{row.get('error')}`"
            )
            continue
        strict_allocator = dict(row.get("strict_allocator") or {})
        train = dict(strict_allocator.get("train") or {})
        val = dict(strict_allocator.get("val") or {})
        oos = dict(strict_allocator.get("oos") or {})
        delta = dict((row.get("comparison_vs_promoted_challenger") or {}).get("oos") or {})
        liquidation_counts = dict(
            (row.get("state_leverage_validation") or {}).get("liquidation_counts") or {}
        )
        total_liquidations = int(sum(int(value) for value in liquidation_counts.values()))
        lines.append(
            f"- inc {row['incumbent_leverage']}x / auto {row['autoresearch_leverage']}x | "
            f"train {float(train.get('total_return') or 0.0):.4%} / {float(train.get('sharpe') or 0.0):.4f} | "
            f"val {float(val.get('total_return') or 0.0):.4%} / {float(val.get('sharpe') or 0.0):.4f} | "
            f"OOS return {float(oos.get('total_return') or 0.0):.4%} | "
            f"Sharpe {float(oos.get('sharpe') or 0.0):.4f} | "
            f"maxDD {float(oos.get('max_drawdown') or 0.0):.4%} | "
            f"liq {total_liquidations} | "
            f"Δreturn {float(delta.get('total_return_delta') or 0.0):.4%} | "
            f"Δsharpe {float(delta.get('sharpe_delta') or 0.0):.4f} | "
            f"elapsed {float(row.get('elapsed_seconds') or 0.0):.2f}s"
        )
    return "\n".join(lines).strip() + "\n"


def _copy_data_sources(data_sources: dict[str, list[str]]) -> dict[str, list[str]]:
    return {str(key): list(values) for key, values in dict(data_sources or {}).items()}


def _prepare_shared_resources(
    *,
    candidates: list[dict[str, Any]],
    split: dict[str, str],
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
) -> tuple[SharedStrictResources, str, str | None]:
    adapted = _runner._adapt_candidate_inputs(candidates, max_candidates=max(1, len(candidates)))
    explicit_timeframes = _candidate_timeframes(candidates)
    explicit_universe = _candidate_universe(candidates)
    normalized_timeframes, universe = _runner._resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=explicit_timeframes,
        symbol_universe=explicit_universe,
    )
    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    resolved_split = _runner._resolve_split_config(split, strategy_timeframe=split_timeframe)
    market_data_settings = current_research_market_data_settings()
    cache_key_payload = _shared_resource_cache_key_payload(
        normalized_timeframes=list(normalized_timeframes),
        universe=list(universe),
        resolved_split=dict(resolved_split),
        candidates=list(candidates),
        data_mode=_VALIDATION_CORE.STRICT_VALIDATION_DATA_MODE,
        allow_csv_fallback=False,
        allow_synthetic_fallback=False,
        min_bundle_bars=1,
        market_data_settings=market_data_settings,
    )
    cache_key = _shared_resource_cache_key(cache_key_payload)

    if cache_dir is not None and not refresh_cache:
        cached = _load_shared_resource_cache(cache_dir=cache_dir, cache_key=cache_key)
        if cached is not None:
            return cached, "disk_cache", cache_key

    if str(_VALIDATION_CORE.STRICT_VALIDATION_DATA_MODE) == "legacy":
        resources = _load_legacy_shared_resources_fast(
            normalized_timeframes=list(normalized_timeframes),
            universe=list(universe),
            resolved_split=dict(resolved_split),
            candidates=list(candidates),
            market_data_settings=market_data_settings,
        )
    else:
        cache, data_sources, feature_cache, benchmark = _runner._load_research_run_resources(
            adapted=adapted,
            normalized_timeframes=normalized_timeframes,
            universe=universe,
            resolved_split=resolved_split,
            data_mode=_VALIDATION_CORE.STRICT_VALIDATION_DATA_MODE,
            allow_csv_fallback=False,
            allow_synthetic_fallback=False,
            min_bundle_bars=1,
            market_data_settings=market_data_settings,
        )
        resources = SharedStrictResources(
            data_sources=_copy_data_sources(data_sources),
            cache=cache,
            feature_cache=feature_cache,
            benchmark=benchmark,
        )
    if cache_dir is not None:
        _write_shared_resource_cache(
            cache_dir=cache_dir,
            cache_key=cache_key,
            cache_key_payload=cache_key_payload,
            resources=resources,
        )
    return resources, "fresh_load", cache_key


def _prepare_shared_context(
    *,
    candidates: list[dict[str, Any]],
    split: dict[str, str],
    resources: SharedStrictResources,
) -> SharedStrictResearchContext:
    if not candidates:
        raise RuntimeError("shared strict context requires at least one candidate")
    adapted = _runner._adapt_candidate_inputs(candidates, max_candidates=max(1, len(candidates)))
    explicit_timeframes = _candidate_timeframes(candidates)
    explicit_universe = _candidate_universe(candidates)
    normalized_timeframes, universe = _runner._resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=explicit_timeframes,
        symbol_universe=explicit_universe,
    )
    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    resolved_split = _runner._resolve_split_config(split, strategy_timeframe=split_timeframe)
    scoring = _runner._resolve_research_run_scoring_config(score_config=None, stage1_keep_ratio=1.0)
    base_tf = _runner._normalize_candidate_research_base_timeframe("1s")
    return SharedStrictResearchContext(
        base_tf=base_tf,
        normalized_timeframes=list(normalized_timeframes),
        universe=list(universe),
        resolved_split=dict(resolved_split),
        scoring=scoring,
        resources=resources,
        source_candidates=[copy.deepcopy(row) for row in candidates],
    )


def _run_report_from_context(
    *,
    context: SharedStrictResearchContext,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    adapted = _runner._adapt_candidate_inputs(candidates, max_candidates=max(1, len(candidates)))
    stage2_results = [
        _runner._evaluate_candidate_with_optional_split(
            row,
            cache=context.resources.cache,
            feature_cache=context.resources.feature_cache,
            benchmark_cache=context.resources.benchmark,
            candidate_count=len(adapted),
            scoring_config=context.scoring.resolved_scoring_config,
            split=context.resolved_split,
        )
        for row in adapted
    ]
    report_candidates = _runner._report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=len(adapted),
        resolved_split=context.resolved_split,
        scoring=context.scoring,
    )
    report_candidates = _runner._sorted_report_candidates(
        report_candidates, scoring=context.scoring
    )
    return _runner._candidate_research_report_payload(
        base_tf=context.base_tf,
        normalized_timeframes=context.normalized_timeframes,
        universe=context.universe,
        resolved_split=context.resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=1.0,
        scoring=context.scoring,
        data_sources=_copy_data_sources(context.resources.data_sources),
        report_candidates=report_candidates,
    )


def _strict_group_payloads_for_combo(
    *,
    inc: int,
    auto: int,
    combo_dir: Path,
    validation_split: dict[str, str],
    incumbent_context: SharedStrictResearchContext,
    autoresearch_context: SharedStrictResearchContext,
    incumbent_portfolio_payload: dict[str, Any],
    autoresearch_portfolio_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    incumbent_candidates = _validation._apply_group_leverage(
        copy.deepcopy(incumbent_context.source_candidates),
        leverage=inc,
    )
    autoresearch_candidates = _validation._apply_group_leverage(
        copy.deepcopy(autoresearch_context.source_candidates),
        leverage=auto,
    )

    incumbent_report = _run_report_from_context(
        context=incumbent_context, candidates=incumbent_candidates
    )
    incumbent_rows = _VALIDATION_CORE._saved_weight_rows(
        list(incumbent_report.get("candidates") or []),
        [
            dict(row)
            for row in list(incumbent_portfolio_payload.get("weights") or [])
            if isinstance(row, dict)
        ],
    )
    incumbent_eval = _VALIDATION_CORE.evaluate_saved_weight_portfolio(incumbent_rows)
    incumbent_payload = _validation._build_group_portfolio_payload(
        label="incumbent",
        source_payload=incumbent_portfolio_payload,
        source_path=Path(
            str(
                incumbent_portfolio_payload.get("source_portfolio_path")
                or _validation.resolve_current_optimization_path()
            )
        ),
        leverage=inc,
        strict_report=incumbent_report,
        refreshed_rows=incumbent_rows,
        eval_payload=incumbent_eval,
    )
    incumbent_written = _validation._write_payload(
        payload=incumbent_payload,
        output_dir=combo_dir / "strict_incumbent_portfolio_current",
        stem="strict_incumbent_portfolio",
        markdown=f"# strict incumbent portfolio\n\n- leverage: `{inc}x`\n",
    )

    autoresearch_report = _run_report_from_context(
        context=autoresearch_context, candidates=autoresearch_candidates
    )
    autoresearch_rows = _VALIDATION_CORE._saved_weight_rows(
        list(autoresearch_report.get("candidates") or []),
        [
            dict(row)
            for row in list(autoresearch_portfolio_payload.get("weights") or [])
            if isinstance(row, dict)
        ],
    )
    autoresearch_eval = _VALIDATION_CORE.evaluate_saved_weight_portfolio(autoresearch_rows)
    autoresearch_payload = _validation._build_group_portfolio_payload(
        label="autoresearch_55_45",
        source_payload=autoresearch_portfolio_payload,
        source_path=Path(
            str(
                autoresearch_portfolio_payload.get("source_portfolio_path")
                or _validation._market._resolve_autoresearch_default_path()
            )
        ),
        leverage=auto,
        strict_report=autoresearch_report,
        refreshed_rows=autoresearch_rows,
        eval_payload=autoresearch_eval,
    )
    autoresearch_written = _validation._write_payload(
        payload=autoresearch_payload,
        output_dir=combo_dir / "strict_autoresearch_portfolio_current",
        stem="strict_autoresearch_portfolio",
        markdown=f"# strict autoresearch 55/45 portfolio\n\n- leverage: `{auto}x`\n",
    )

    blend_payload = _validation._build_blend_payload(
        incumbent_payload=incumbent_payload,
        autoresearch_payload=autoresearch_payload,
        blend_weight=float(_validation.DEFAULT_BLEND_WEIGHT),
    )
    _validation._write_payload(
        payload=blend_payload,
        output_dir=combo_dir / "strict_blend_portfolio_current",
        stem="strict_grouped_blend_portfolio",
        markdown=(
            "# strict grouped blend portfolio\n\n"
            f"- incumbent weight: `{float(_validation.DEFAULT_BLEND_WEIGHT):.2%}`\n"
            f"- autoresearch weight: `{1.0 - float(_validation.DEFAULT_BLEND_WEIGHT):.2%}`\n"
        ),
    )
    return incumbent_written, autoresearch_written, blend_payload


def run_batch_retune(
    *,
    incumbent_leverages: list[int],
    autoresearch_leverages: list[int],
    output_dir: Path,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
    shared_cache_dir: Path | None,
    refresh_shared_cache: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{_utc_now_iso()}] preparing strict retune inputs", flush=True)

    incumbent_bundle_path = _validation.DEFAULT_INCUMBENT_BUNDLE
    incumbent_portfolio_path = _validation.resolve_current_optimization_path()
    autoresearch_portfolio_path = _validation._market._resolve_autoresearch_default_path()
    leverage_tuning_path = _validation.DEFAULT_LEVERAGE_TUNING_PATH
    decision_path = _validation.DEFAULT_DECISION_PATH

    incumbent_bundle = _validation._path_payload(incumbent_bundle_path)
    incumbent_portfolio_payload = _validation._path_payload(incumbent_portfolio_path)
    incumbent_portfolio_payload["source_portfolio_path"] = str(
        Path(incumbent_portfolio_path).resolve()
    )
    autoresearch_portfolio_payload = _validation._path_payload(autoresearch_portfolio_path)
    autoresearch_portfolio_payload["source_portfolio_path"] = str(
        Path(autoresearch_portfolio_path).resolve()
    )
    leverage_payload = _validation._path_payload(leverage_tuning_path)
    decision_payload = _validation._path_payload(decision_path)

    common_oos_end = min(
        _validation._portfolio_oos_end(incumbent_portfolio_payload),
        _validation._portfolio_oos_end(autoresearch_portfolio_payload),
    )
    validation_split = _VALIDATION_CORE.build_validation_split(common_oos_end)

    incumbent_source_candidates = _validation._resolve_portfolio_candidates(
        bundle_payload=incumbent_bundle,
        portfolio_payload=incumbent_portfolio_payload,
    )
    autoresearch_source_candidates = _validation._resolve_portfolio_candidates(
        portfolio_payload=autoresearch_portfolio_payload,
    )
    print(f"[{_utc_now_iso()}] loading shared strict research resources", flush=True)
    shared_started = perf_counter()
    shared_resources, shared_resource_source, shared_resource_cache_key = _prepare_shared_resources(
        candidates=[*incumbent_source_candidates, *autoresearch_source_candidates],
        split=validation_split,
        cache_dir=None if shared_cache_dir is None else Path(shared_cache_dir).resolve(),
        refresh_cache=bool(refresh_shared_cache),
    )
    shared_elapsed = perf_counter() - shared_started
    source_label = "disk-cache-hit" if shared_resource_source == "disk_cache" else "fresh-load"
    print(
        f"[{_utc_now_iso()}] shared strict research resources ready ({source_label}, {shared_elapsed:.2f}s)",
        flush=True,
    )
    del shared_resources
    _release_process_memory()

    rows: list[dict[str, Any]] = []
    for inc in incumbent_leverages:
        for auto in autoresearch_leverages:
            combo_dir = output_dir / f"inc_{inc}_auto_{auto}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            print(f"[{_utc_now_iso()}] combo start inc={inc} auto={auto}", flush=True)
            combo_resources: SharedStrictResources | None = None
            incumbent_context: SharedStrictResearchContext | None = None
            autoresearch_context: SharedStrictResearchContext | None = None
            try:
                combo_resources, _, _ = _prepare_shared_resources(
                    candidates=[*incumbent_source_candidates, *autoresearch_source_candidates],
                    split=validation_split,
                    cache_dir=None
                    if shared_cache_dir is None
                    else Path(shared_cache_dir).resolve(),
                    refresh_cache=False,
                )
                incumbent_context = _prepare_shared_context(
                    candidates=incumbent_source_candidates,
                    split=validation_split,
                    resources=combo_resources,
                )
                autoresearch_context = _prepare_shared_context(
                    candidates=autoresearch_source_candidates,
                    split=validation_split,
                    resources=combo_resources,
                )
                incumbent_written, autoresearch_written, blend_payload = (
                    _strict_group_payloads_for_combo(
                        inc=inc,
                        auto=auto,
                        combo_dir=combo_dir,
                        validation_split=validation_split,
                        incumbent_context=incumbent_context,
                        autoresearch_context=autoresearch_context,
                        incumbent_portfolio_payload=incumbent_portfolio_payload,
                        autoresearch_portfolio_payload=autoresearch_portfolio_payload,
                    )
                )
                incumbent_context = None
                autoresearch_context = None
                combo_resources = None
                _release_process_memory()
                strict_market = _validation._market.run_group_market_regime_judgement(
                    incumbent_path=incumbent_written["json"],
                    autoresearch_path=autoresearch_written["json"],
                    output_dir=combo_dir / "strict_market_regime_judgement_current",
                    horizon_days=max(1, int(_validation.DEFAULT_HORIZON_DAYS)),
                    soft_rss_bytes=max(1, int(soft_rss_bytes)),
                    hard_rss_bytes=max(1, int(hard_rss_bytes)),
                )
                strict_allocator = _validation._three_way.run_three_way_market_regime_allocator(
                    incumbent_path=incumbent_written["json"],
                    blend_path=(
                        combo_dir
                        / "strict_blend_portfolio_current"
                        / "strict_grouped_blend_portfolio_latest.json"
                    ),
                    autoresearch_path=autoresearch_written["json"],
                    market_judgement_path=strict_market["latest_json_path"],
                    output_dir=combo_dir / "strict_three_way_market_regime_allocator_current",
                    soft_rss_bytes=max(1, int(soft_rss_bytes)),
                    hard_rss_bytes=max(1, int(hard_rss_bytes)),
                )
                promoted_candidate = _validation._resolve_current_promoted_candidate(
                    decision_payload
                )
                strict_allocator_payload = _validation._apply_allocator_state_leverage_to_payload(
                    allocator_payload=dict(strict_allocator["payload"]),
                    leverage_by_state={
                        "incumbent": int(inc),
                        "blend_85_15": int(
                            (leverage_payload.get("best_result") or {})
                            .get("leverage_by_state", {})
                            .get("blend_85_15")
                            or 1
                        ),
                        "autoresearch_55_45": int(auto),
                    },
                )
                strict_allocator_payload["artifact_path"] = str(
                    Path(strict_allocator["latest_json_path"]).resolve()
                )
                comparison = _validation._comparison_block(
                    dict(strict_allocator_payload.get("split_metrics") or {}),
                    {
                        "train": dict(promoted_candidate.get("train") or {}),
                        "val": dict(promoted_candidate.get("val") or {}),
                        "oos": dict(promoted_candidate.get("oos") or {}),
                    },
                )
                final_payload = {
                    "artifact_kind": "grouped_allocator_strict_leverage_validation",
                    "generated_at": _utc_now_iso(),
                    "selection_basis": "strict_candidate_level_leverage_validation_before_allocator_promotion",
                    "validation_split": validation_split,
                    "input_paths": {
                        "incumbent_bundle": str(Path(incumbent_bundle_path).resolve()),
                        "incumbent_portfolio": str(Path(incumbent_portfolio_path).resolve()),
                        "autoresearch_portfolio": str(Path(autoresearch_portfolio_path).resolve()),
                        "leverage_tuning": str(Path(leverage_tuning_path).resolve()),
                        "decision": str(Path(decision_path).resolve()),
                    },
                    "leverage_by_state": {
                        "incumbent": int(inc),
                        "blend_85_15": int(
                            (leverage_payload.get("best_result") or {})
                            .get("leverage_by_state", {})
                            .get("blend_85_15")
                            or 1
                        ),
                        "autoresearch_55_45": int(auto),
                    },
                    "strict_artifact_paths": {
                        "incumbent": str(Path(incumbent_written["json"]).resolve()),
                        "autoresearch_55_45": str(Path(autoresearch_written["json"]).resolve()),
                        "blend_85_15": str(
                            (
                                combo_dir
                                / "strict_blend_portfolio_current"
                                / "strict_grouped_blend_portfolio_latest.json"
                            ).resolve()
                        ),
                        "market_judgement": str(Path(strict_market["latest_json_path"]).resolve()),
                        "allocator": str(Path(strict_allocator["latest_json_path"]).resolve()),
                    },
                    "strict_group_metrics": {
                        "incumbent": dict(
                            _validation._path_payload(incumbent_written["json"]).get(
                                "portfolio_metrics"
                            )
                            or {}
                        ),
                        "autoresearch_55_45": dict(
                            _validation._path_payload(autoresearch_written["json"]).get(
                                "portfolio_metrics"
                            )
                            or {}
                        ),
                        "blend_85_15": dict(blend_payload.get("portfolio_metrics") or {}),
                        "allocator": dict(strict_allocator_payload.get("split_metrics") or {}),
                    },
                    "current_promoted_challenger": promoted_candidate,
                    "strict_allocator": strict_allocator_payload,
                    "comparison_vs_promoted_challenger": comparison,
                    "notes": [
                        "Batch retune reused strict research resources via a persisted shared disk cache.",
                        "Shared market-data/feature caches preserve exactness while avoiding repeated cold loads.",
                        "Allocator state returns were re-levered using leverage_by_state after strict sleeve reconstruction.",
                    ],
                }
                written = _validation._write_payload(
                    payload=final_payload,
                    output_dir=combo_dir,
                    stem="grouped_allocator_strict_leverage_validation",
                    markdown=_validation._build_markdown(final_payload),
                )
                rows.append(
                    {
                        "incumbent_leverage": int(inc),
                        "autoresearch_leverage": int(auto),
                        "status": "ok",
                        "elapsed_seconds": perf_counter() - started,
                        "strict_allocator": final_payload.get("strict_allocator", {}).get(
                            "split_metrics"
                        ),
                        "state_leverage_validation": final_payload.get("strict_allocator", {}).get(
                            "state_leverage_validation"
                        ),
                        "comparison_vs_promoted_challenger": final_payload.get(
                            "comparison_vs_promoted_challenger"
                        ),
                        "report_path": str(Path(written["json"]).resolve()),
                    }
                )
                print(f"[{_utc_now_iso()}] combo done inc={inc} auto={auto}", flush=True)
            except Exception as exc:  # pragma: no cover - summarized for operators
                incumbent_context = None
                autoresearch_context = None
                combo_resources = None
                _release_process_memory()
                rows.append(
                    {
                        "incumbent_leverage": int(inc),
                        "autoresearch_leverage": int(auto),
                        "status": "error",
                        "elapsed_seconds": perf_counter() - started,
                        "error": str(exc),
                    }
                )
                print(f"[{_utc_now_iso()}] combo error inc={inc} auto={auto}: {exc}", flush=True)
    rows.sort(
        key=lambda row: (
            row.get("status") == "ok",
            float(
                ((row.get("strict_allocator") or {}).get("oos") or {}).get("total_return") or 0.0
            ),
            float(((row.get("strict_allocator") or {}).get("oos") or {}).get("sharpe") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "artifact_kind": "grouped_allocator_strict_leverage_batch_retune",
        "generated_at": _utc_now_iso(),
        "incumbent_leverages": list(incumbent_leverages),
        "autoresearch_leverages": list(autoresearch_leverages),
        "shared_resource_cache": {
            "cache_dir": str(Path(shared_cache_dir).resolve())
            if shared_cache_dir is not None
            else None,
            "refresh_requested": bool(refresh_shared_cache),
            "resource_source": shared_resource_source,
            "cache_key": shared_resource_cache_key,
            "preload_elapsed_seconds": shared_elapsed,
        },
        "rows": rows,
    }
    summary_json = output_dir / "summary_latest.json"
    summary_md = output_dir / "summary_latest.md"
    _write_json(summary_json, payload)
    summary_md.write_text(_build_markdown(rows), encoding="utf-8")
    return {"payload": payload, "summary_json": summary_json, "summary_md": summary_md}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-leverages", default="1,3,5,7,9")
    parser.add_argument("--autoresearch-leverages", default="1,2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--soft-rss-bytes", type=int, default=int(_validation.DEFAULT_SOFT_RSS_BYTES)
    )
    parser.add_argument(
        "--hard-rss-bytes", type=int, default=int(_validation.DEFAULT_HARD_RSS_BYTES)
    )
    parser.add_argument("--shared-cache-dir", type=Path, default=DEFAULT_SHARED_RESOURCE_CACHE_DIR)
    parser.add_argument("--refresh-shared-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_batch_retune(
        incumbent_leverages=_parse_leverage_list(args.incumbent_leverages),
        autoresearch_leverages=_parse_leverage_list(args.autoresearch_leverages),
        output_dir=Path(args.output_dir).resolve(),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
        shared_cache_dir=None
        if args.shared_cache_dir is None
        else Path(args.shared_cache_dir).resolve(),
        refresh_shared_cache=bool(args.refresh_shared_cache),
    )
    print(Path(report["summary_json"]).resolve())
    print(Path(report["summary_md"]).resolve())


if __name__ == "__main__":
    main()
