from __future__ import annotations

from lumina_quant.strategy_factory import research_stage_support


def test_build_research_resource_loader_preserves_injected_dependencies() -> None:
    def _split_window_bounds(split):
        return split

    def _datetime_to_iso_z(value):
        return value

    def _load_bundle_cache(**kwargs):
        return kwargs

    def _load_feature_cache(**kwargs):
        return kwargs

    def _benchmark_cache(cache, timeframes):
        return {"cache": cache, "timeframes": list(timeframes)}

    def _canonicalize_symbol_list(items):
        return list(items)

    loader = research_stage_support.build_research_resource_loader(
        split_window_bounds=_split_window_bounds,
        datetime_to_iso_z=_datetime_to_iso_z,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=_benchmark_cache,
        canonicalize_symbol_list=_canonicalize_symbol_list,
    )

    assert loader.split_window_bounds is _split_window_bounds
    assert loader.datetime_to_iso_z is _datetime_to_iso_z
    assert loader.load_bundle_cache is _load_bundle_cache
    assert loader.load_feature_cache is _load_feature_cache
    assert loader.benchmark_cache is _benchmark_cache
    assert loader.canonicalize_symbol_list is _canonicalize_symbol_list


def test_load_research_run_resources_delegates_to_loader() -> None:
    captured: dict[str, object] = {}
    progress_callback = object()

    class _Loader:
        def load(self, **kwargs):
            captured.update(kwargs)
            return (
                {"cache": 1},
                {"parquet": ["BTC/USDT@1m"]},
                {"features": 1},
                {"1m": {"close": []}},
            )

    result = research_stage_support.load_research_run_resources(
        loader=_Loader(),
        adapted=[{"candidate_id": "cand-1"}],
        normalized_timeframes=["1m"],
        universe=["BTC/USDT"],
        resolved_split={"mode": "default"},
        data_mode="raw_first",
        allow_csv_fallback=False,
        allow_synthetic_fallback=True,
        min_bundle_bars=240,
        market_data_settings={"market_data_parquet_path": "var/data/runtime"},
        progress_callback=progress_callback,
    )

    assert captured == {
        "adapted": [{"candidate_id": "cand-1"}],
        "normalized_timeframes": ["1m"],
        "universe": ["BTC/USDT"],
        "resolved_split": {"mode": "default"},
        "data_mode": "raw_first",
        "allow_csv_fallback": False,
        "allow_synthetic_fallback": True,
        "min_bundle_bars": 240,
        "market_data_settings": {"market_data_parquet_path": "var/data/runtime"},
        "progress_callback": progress_callback,
    }
    assert result == (
        {"cache": 1},
        {"parquet": ["BTC/USDT@1m"]},
        {"features": 1},
        {"1m": {"close": []}},
    )


def test_evaluate_candidate_with_optional_split_uses_selector_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}
    cache = {("BTC/USDT", "1m"): object()}
    feature_cache = {"BTC/USDT": object()}
    benchmark_cache = {"1m": object()}

    class _Selector:
        def __init__(self, *, evaluate_candidate, stage1_prefilter_score):
            captured["evaluate_candidate"] = evaluate_candidate
            captured["stage1_prefilter_score"] = stage1_prefilter_score

        def evaluate_candidate_with_optional_split(self, candidate, **kwargs):
            captured["candidate"] = candidate
            captured["kwargs"] = kwargs
            return {"candidate_id": candidate["candidate_id"], "pass": True}

    monkeypatch.setattr(research_stage_support, "ResearchStageSelector", _Selector)

    result = research_stage_support.evaluate_candidate_with_optional_split(
        evaluate_candidate="evaluate-fn",
        stage1_prefilter_score="stage1-fn",
        candidate={"candidate_id": "cand-1"},
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=None,
        benchmark_cache=benchmark_cache,
        candidate_count=2,
        scoring_config={"return_weight": 10.0},
        split={"mode": "default"},
    )

    assert captured["evaluate_candidate"] == "evaluate-fn"
    assert captured["stage1_prefilter_score"] == "stage1-fn"
    assert captured["candidate"] == {"candidate_id": "cand-1"}
    assert captured["kwargs"] == {
        "cache": cache,
        "feature_cache": feature_cache,
        "aligned_cache": None,
        "benchmark_cache": benchmark_cache,
        "candidate_count": 2,
        "scoring_config": {"return_weight": 10.0},
        "split": {"mode": "default"},
    }
    assert result == {"candidate_id": "cand-1", "pass": True}


def test_select_stage2_results_uses_selector_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}
    cache = {("BTC/USDT", "1m"): object()}
    feature_cache = {"BTC/USDT": object()}
    benchmark = {"1m": object()}

    class _Selector:
        def __init__(self, *, evaluate_candidate, stage1_prefilter_score):
            captured["evaluate_candidate"] = evaluate_candidate
            captured["stage1_prefilter_score"] = stage1_prefilter_score

        def select_stage2_results(self, **kwargs):
            captured["kwargs"] = kwargs
            return [{"candidate_id": "cand-2"}]

    monkeypatch.setattr(research_stage_support, "ResearchStageSelector", _Selector)

    scoring = type(
        "_Scoring",
        (),
        {
            "resolved_scoring_config": {"return_weight": 10.0},
            "keep_ratio_applied": 0.5,
            "stage1_weights": {"sharpe_weight": 2.0},
            "stage1_error_score": -1_000_000.0,
        },
    )()

    result = research_stage_support.select_stage2_results(
        evaluate_candidate="evaluate-fn",
        stage1_prefilter_score="stage1-fn",
        adapted=[{"candidate_id": "cand-1"}, {"candidate_id": "cand-2"}],
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=None,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split={"mode": "default"},
    )

    assert captured["evaluate_candidate"] == "evaluate-fn"
    assert captured["stage1_prefilter_score"] == "stage1-fn"
    assert captured["kwargs"] == {
        "adapted": [{"candidate_id": "cand-1"}, {"candidate_id": "cand-2"}],
        "cache": cache,
        "feature_cache": feature_cache,
        "aligned_cache": None,
        "benchmark": benchmark,
        "scoring": scoring,
        "resolved_split": {"mode": "default"},
        "progress_callback": None,
    }
    assert result == [{"candidate_id": "cand-2"}]
