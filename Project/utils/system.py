"""System-level helpers for runtime tracking and resource measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

try:  # optional dependency guard
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


RUNTIME_PATH = Path("reports/runtime.json")


def capture_resource_snapshot() -> Dict[str, float]:
    """Return lightweight resource metrics for the current process."""
    if psutil is None:
        return {}
    proc = psutil.Process()  # type: ignore[attr-defined]
    with proc.oneshot():  # type: ignore[attr-defined]
        mem = proc.memory_info()
        cpu = proc.cpu_times()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
        "cpu_user_sec": cpu.user,
        "cpu_system_sec": cpu.system,
    }


def merge_runtime_sections(sections: Mapping[str, Iterable[Mapping[str, Any]]], *, path: Path = RUNTIME_PATH) -> None:
    """Merge one or more runtime payloads into the persisted JSON report."""
    materialised: Dict[str, List[Dict[str, Any]]] = {}
    for key, entries in sections.items():
        if not entries:
            continue
        materialised[key] = [dict(item) for item in entries]
    if not materialised:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}
    for key, rows in materialised.items():
        bucket = existing.setdefault(key, [])
        if not isinstance(bucket, list):
            bucket = []
        bucket.extend(rows)
        existing[key] = bucket
    path.write_text(json.dumps(existing, indent=2))


__all__ = ["capture_resource_snapshot", "merge_runtime_sections", "RUNTIME_PATH"]
