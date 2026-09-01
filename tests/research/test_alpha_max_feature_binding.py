from __future__ import annotations

from dataclasses import dataclass

import pytest

import lumina_quant.research.alpha_max_engine_runner as runner
import lumina_quant.research.alpha_max_evidence as evidence
from lumina_quant.data.feature_points import SealedFeatureFile
from lumina_quant.research.alpha_max_evidence import (
    AlphaMaxOrderedFundingLookup,
    FeatureRootSpec,
)


@dataclass(frozen=True)
class _Entry:
    relative_path: str
    byte_count: int = 17
    mode: int = 0o600
    mtime_ns: int = 123
    sha256: str = "a" * 64


class _RootSeal:
    post_init_calls = 0

    def __init__(self, spec: FeatureRootSpec) -> None:
        self.root_kind = "feature"
        self.root_id = spec.root_id
        self.path = spec.path
        self.exchange = spec.exchange
        self.start_utc = spec.start_utc
        self.end_utc = spec.end_utc
        self.inventory_sha256 = spec.inventory_sha256
        self.content_sha256 = spec.content_sha256
        self.entries = (
            _Entry(
                "feature_points/exchange=binance/"
                f"symbol=BTCUSDT/date={spec.start_utc:%Y-%m-%d}/part-0.parquet"
            ),
        )

    def __post_init__(self) -> None:
        type(self).post_init_calls += 1


class _FeatureLookup:
    instances = []

    def __init__(
        self,
        *,
        db_path,
        exchange,
        start_date,
        end_date,
        sealed_files=None,
    ) -> None:
        self.db_path = db_path
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.sealed_files = sealed_files
        type(self).instances.append(self)


def _spec(tmp_path, root_id: str) -> FeatureRootSpec:
    path = tmp_path / root_id
    path.mkdir()
    start, end = evidence._ROOT_INTERVALS[root_id]
    return FeatureRootSpec(
        root_id=root_id,
        path=str(path.resolve()),
        exchange="binance",
        start_utc=start,
        end_utc=end,
        inventory_sha256="b" * 64,
        content_sha256="c" * 64,
    )


def test_ordered_lookup_exactly_revalidates_and_freezes_sealed_file_bindings(
    tmp_path, monkeypatch
) -> None:
    specs = (_spec(tmp_path, "purge"), _spec(tmp_path, "validation"))
    seals = tuple(_RootSeal(spec) for spec in specs)
    _FeatureLookup.instances = []
    _RootSeal.post_init_calls = 0
    monkeypatch.setattr(evidence, "AlphaMaxRootSeal", _RootSeal)
    monkeypatch.setattr(evidence, "FeaturePointLookup", _FeatureLookup)

    lookup = AlphaMaxOrderedFundingLookup(specs, root_seals=seals)

    assert lookup.root_seals == seals
    assert _RootSeal.post_init_calls == 2
    assert len(_FeatureLookup.instances) == 2
    assert all(
        type(instance.sealed_files) is tuple
        and all(type(entry) is SealedFeatureFile for entry in instance.sealed_files)
        for instance in _FeatureLookup.instances
    )
    with pytest.raises(AttributeError, match="immutable"):
        lookup._root_seals = ()


def test_ordered_lookup_rejects_seal_to_spec_identity_mismatch(tmp_path, monkeypatch) -> None:
    specs = (_spec(tmp_path, "purge"), _spec(tmp_path, "validation"))
    seals = tuple(_RootSeal(spec) for spec in specs)
    seals[1].path = specs[0].path
    monkeypatch.setattr(evidence, "AlphaMaxRootSeal", _RootSeal)

    with pytest.raises(ValueError, match="feature_root_seal_spec_mismatch"):
        AlphaMaxOrderedFundingLookup(specs, root_seals=seals)


def test_runner_passes_the_exact_feature_seals_into_ordered_lookup(tmp_path, monkeypatch) -> None:
    specs = (_spec(tmp_path, "purge"), _spec(tmp_path, "validation"))
    seals = (object(), object())
    by_identity = dict(zip(map(id, seals), specs, strict=True))
    captured = {}

    class _OrderedLookup:
        def __init__(self, root_specs, *, root_seals) -> None:
            captured["root_specs"] = root_specs
            captured["root_seals"] = root_seals

    monkeypatch.setattr(
        runner,
        "_alpha_max_feature_spec",
        lambda seal: by_identity[id(seal)],
    )
    monkeypatch.setattr(runner, "AlphaMaxOrderedFundingLookup", _OrderedLookup)

    result = runner._alpha_max_ordered_lookup(
        {
            ("purge", "feature"): seals[0],
            ("validation", "feature"): seals[1],
        },
        ("purge", "validation"),
    )

    assert type(result) is _OrderedLookup
    assert captured == {"root_specs": specs, "root_seals": seals}
