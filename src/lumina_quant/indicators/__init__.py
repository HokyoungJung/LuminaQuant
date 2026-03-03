"""Public-safe indicator exports with optional private extension overlay."""

from __future__ import annotations

from importlib import import_module

from .sample_public_indicator import sample_momentum as sample_momentum

__all__ = ["sample_momentum"]


def _load_private_indicator_exports() -> dict[str, object]:
    try:
        private_module = import_module("lumina_quant_private.indicators")
    except Exception:
        return {}

    names = getattr(private_module, "__all__", None)
    if not isinstance(names, list) or not names:
        names = [name for name in dir(private_module) if not name.startswith("_")]

    exports: dict[str, object] = {}
    for name in names:
        try:
            exports[str(name)] = getattr(private_module, str(name))
        except Exception:
            continue
    return exports


_PRIVATE_EXPORTS = _load_private_indicator_exports()
for _name, _obj in _PRIVATE_EXPORTS.items():
    globals()[_name] = _obj
    if _name not in __all__:
        __all__.append(_name)

__all__ = sorted(set(__all__))
