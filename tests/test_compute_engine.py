import lumina_quant.compute_engine as compute_engine
import polars as pl
import pytest


class _CompletedProcess:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_detect_nvidia_gpu_returns_false_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(compute_engine.shutil, "which", lambda name: None)

    ok, reason = compute_engine.detect_nvidia_gpu()

    assert ok is False
    assert "nvidia-smi not found" in reason


def test_detect_nvidia_gpu_returns_true_on_nvidia_smi_success(monkeypatch):
    monkeypatch.setattr(compute_engine.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        compute_engine.subprocess,
        "run",
        lambda *args, **kwargs: _CompletedProcess(
            returncode=0,
            stdout="GPU 0: Test Device\nGPU 1: Test Device",
        ),
    )

    ok, reason = compute_engine.detect_nvidia_gpu()

    assert ok is True
    assert "detected 2 GPU(s)" in reason


def test_cpu_mode_forces_cpu_without_gpu_probe(monkeypatch):
    def _should_not_probe(*, device, smoke_test):
        raise AssertionError("GPU probe should not run in cpu mode")

    monkeypatch.setattr(compute_engine, "polars_gpu_available", _should_not_probe)

    selection = compute_engine.select_engine(mode="cpu", device="gpu:3", verbose="1")

    assert selection.requested_mode == "cpu"
    assert selection.resolved_engine == "cpu"
    assert selection.device == 3
    assert selection.verbose is True
    assert selection.reason == "LQ_GPU_MODE=cpu"


def test_auto_mode_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(
        compute_engine,
        "polars_gpu_available",
        lambda *, device, smoke_test: (False, "gpu runtime unavailable"),
    )

    selection = compute_engine.select_engine(mode="auto", device="cuda:1")

    assert selection.requested_mode == "auto"
    assert selection.resolved_engine == "cpu"
    assert selection.device == 1
    assert "auto fallback to CPU" in selection.reason


def test_auto_mode_selects_gpu_when_available(monkeypatch):
    monkeypatch.setattr(
        compute_engine,
        "polars_gpu_available",
        lambda *, device, smoke_test: (True, f"gpu smoke test passed on device={device}"),
    )

    selection = compute_engine.select_engine(mode="auto", device="1")

    assert selection.resolved_engine == "gpu"
    assert selection.device == 1
    assert "gpu smoke test passed" in selection.reason


def test_forced_gpu_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        compute_engine,
        "polars_gpu_available",
        lambda *, device, smoke_test: (False, "nvidia-smi not found"),
    )

    with pytest.raises(compute_engine.GPUNotAvailableError, match="requires GPU"):
        compute_engine.select_engine(mode="forced-gpu", device="0")


def test_gpu_mode_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        compute_engine,
        "polars_gpu_available",
        lambda *, device, smoke_test: (False, "gpu smoke test failed"),
    )

    with pytest.raises(compute_engine.GPUNotAvailableError, match="LQ_GPU_MODE=gpu"):
        compute_engine.select_engine(mode="gpu", device="0")


def test_select_engine_verbose_fallback_logs_reason(monkeypatch, capsys):
    monkeypatch.setattr(
        compute_engine,
        "polars_gpu_available",
        lambda *, device, smoke_test: (False, "simulated failure"),
    )

    selection = compute_engine.select_engine(mode="auto", verbose=True)
    captured = capsys.readouterr()

    assert selection.resolved_engine == "cpu"
    assert "auto fallback to CPU" in selection.reason
    assert "auto fallback to CPU" in captured.out


def test_parse_gpu_device_accepts_prefix_and_numeric_tokens():
    assert compute_engine._parse_gpu_device(None) is None
    assert compute_engine._parse_gpu_device(2) == 2
    assert compute_engine._parse_gpu_device("3") == 3
    assert compute_engine._parse_gpu_device("cuda:4") == 4
    assert compute_engine._parse_gpu_device("gpu:5") == 5


def test_parse_gpu_device_rejects_invalid_values():
    with pytest.raises(ValueError):
        compute_engine._parse_gpu_device("abc")

    with pytest.raises(ValueError):
        compute_engine._parse_gpu_device("cuda:x")


def test_parse_gpu_vram_accepts_numeric_values():
    assert compute_engine._parse_gpu_vram_gb(None) == 0.0
    assert compute_engine._parse_gpu_vram_gb("0") == 0.0
    assert compute_engine._parse_gpu_vram_gb("6.5") == 6.5


def test_parse_gpu_vram_rejects_invalid_values():
    with pytest.raises(ValueError):
        compute_engine._parse_gpu_vram_gb("abc")
    with pytest.raises(ValueError):
        compute_engine._parse_gpu_vram_gb("-1")


def test_collect_uses_provided_engine_without_reselecting(monkeypatch):
    class _FakeLazyFrame:
        def __init__(self):
            self.engine = None

        def collect(self, *, engine):
            self.engine = engine
            return {"engine": engine}

    lazy_frame = _FakeLazyFrame()
    engine = compute_engine.ComputeEngine(
        requested_mode="cpu",
        resolved_engine="cpu",
        device=None,
        verbose=False,
        reason="test",
    )

    output = compute_engine.collect(lazy_frame, engine=engine)

    assert output == {"engine": "streaming"}
    assert lazy_frame.engine == "streaming"


def test_resolve_compute_engine_aliases_select_engine(monkeypatch):
    sentinel = compute_engine.ComputeEngine(
        requested_mode="cpu",
        resolved_engine="cpu",
        device=None,
        verbose=False,
        reason="alias",
    )
    monkeypatch.setattr(compute_engine, "select_engine", lambda **kwargs: sentinel)

    resolved = compute_engine.resolve_compute_engine(mode="auto")

    assert resolved is sentinel


def test_gpu_collect_falls_back_to_cpu_on_oom_when_not_forced(monkeypatch):
    class _FakeLazyFrame:
        def __init__(self):
            self.engines = []

        def collect(self, *, engine):
            self.engines.append(engine)
            if engine != "streaming":
                raise RuntimeError("CUDA out of memory")
            return {"engine": engine}

    monkeypatch.setattr(compute_engine, "_build_gpu_engine", lambda *, device: f"gpu:{device}")

    lazy = _FakeLazyFrame()
    engine = compute_engine.ComputeEngine(
        requested_mode="auto",
        resolved_engine="gpu",
        device=0,
        verbose=False,
        reason="test",
    )

    out = engine.collect(lazy)

    assert out == {"engine": "streaming"}
    assert lazy.engines == ["gpu:0", "streaming"]


def test_gpu_collect_keeps_oom_when_forced(monkeypatch):
    class _FakeLazyFrame:
        def collect(self, *, engine):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(compute_engine, "_build_gpu_engine", lambda *, device: f"gpu:{device}")

    lazy = _FakeLazyFrame()
    engine = compute_engine.ComputeEngine(
        requested_mode="forced-gpu",
        resolved_engine="gpu",
        device=0,
        verbose=False,
        reason="test",
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        engine.collect(lazy)


def test_compare_cpu_gpu_determinism_report(monkeypatch):
    class _FakeEngine:
        def __init__(self, frame: pl.DataFrame):
            self._frame = frame

        def collect(self, lazy_frame):
            _ = lazy_frame
            return self._frame

    cpu_frame = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    gpu_frame = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    def _fake_select_engine(*, mode, **kwargs):
        _ = kwargs
        if mode == "cpu":
            return _FakeEngine(cpu_frame)
        if mode == "gpu":
            return _FakeEngine(gpu_frame)
        raise AssertionError(f"unexpected mode={mode}")

    monkeypatch.setattr(compute_engine, "select_engine", _fake_select_engine)

    report = compute_engine.compare_cpu_gpu_determinism(None, tolerance=1e-12)

    assert report["within_tolerance"] is True
    assert report["cpu_rows"] == 2
    assert report["gpu_rows"] == 2
    assert report["max_abs_diff"]["x"] == 0.0
