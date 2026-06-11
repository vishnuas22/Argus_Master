"""Plugin registry. Each evidence module is a separate file in modules/
exporting MODULE = <class>. Import failures register a visible
FailedImportModule (fail-closed). Stub module only when ARGUS_ENABLE_STUB=1.
"""
import importlib
import logging
import os
from typing import List

from base import EvidenceModule, FailedImportModule

logger = logging.getLogger("argus.registry")

MODULE_FILES = ["metadata", "compression", "spectral", "realness", "perturbation"]

_cache = None


def _load():
    entries = []
    names = list(MODULE_FILES)
    if os.environ.get("ARGUS_ENABLE_STUB") == "1":
        names.append("stub")
    for name in names:
        try:
            mod = importlib.import_module(f"modules.{name}")
            entries.append(("cls", mod.MODULE))
        except Exception as exc:  # noqa: BLE001 — fail-closed, stay visible
            logger.warning("module file %s failed to import: %s", name, exc)
            entries.append(("failed", (name, str(exc))))
    return entries


def get_modules(force_reload: bool = False) -> List[EvidenceModule]:
    global _cache
    if _cache is None or force_reload:
        _cache = _load()
    instances = []
    for kind, payload in _cache:
        if kind == "cls":
            instances.append(payload())
        else:
            name, err = payload
            instances.append(FailedImportModule(name, err))
    return instances


def registered_versions() -> dict:
    return {m.module_id: m.version for m in get_modules()}
