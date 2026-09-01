from __future__ import annotations

import importlib

import pytest
from lumina_quant.infra import (
    AuditStore,
    JsonLogFormatter,
    NotificationManager,
    StateManager,
    setup_logging,
)
from lumina_quant.utils.audit_store import AuditStore as CanonicalAuditStore
from lumina_quant.utils.logging_utils import JsonLogFormatter as CanonicalJsonLogFormatter
from lumina_quant.utils.logging_utils import setup_logging as canonical_setup_logging
from lumina_quant.utils.notification import NotificationManager as CanonicalNotificationManager
from lumina_quant.utils.persistence import StateManager as CanonicalStateManager


def test_infra_exports_use_canonical_runtime_modules():
    assert AuditStore is CanonicalAuditStore
    assert JsonLogFormatter is CanonicalJsonLogFormatter
    assert NotificationManager is CanonicalNotificationManager
    assert StateManager is CanonicalStateManager
    assert setup_logging is canonical_setup_logging


@pytest.mark.parametrize(
    "module_name",
    (
        "lumina_quant.infra.audit",
        "lumina_quant.infra.logging",
        "lumina_quant.infra.notification",
        "lumina_quant.infra.persistence",
    ),
)
def test_legacy_infra_alias_modules_removed(module_name: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
