"""Capture garbage collection: cap a suite directory by count and/or age.

:func:`evict` runs after a successful disk write, on the **suite** directory just written to. It
caps it by **unit**: a flat ``*.json`` capture, or a whole conversation folder
(``conv_<16 hex>/``, see :mod:`evalshift.sinks._layout`) timed by its newest file. It does a single
:func:`os.scandir` pass over the suite directory (plus one per conversation folder) and orders by
**filesystem mtime** (not the envelope's ``created_at``) — stat only, no JSON parsing, so it stays
cheap even when ``max_captures`` is large. Two rules combine (union):

* **count** — walking units newest first, the newest unit keeps its newest ``max_captures`` files;
  older units are kept whole while the running count stays within ``max_captures``, and the first
  that does not fit is evicted with every older unit;
* **TTL** — evict any ``*.json`` file whose age (``now_ts - mtime``) exceeds
  ``capture_ttl_seconds``, so an idle conversation goes whole and a live one loses only its old
  turns.

A conversation folder left empty is removed. Other subdirectories are never entered. Each unlink
is individually swallowed, and the whole scan is guarded, so eviction never raises into the host.
Limitation (``docs/DECISIONS.md`` D-6): tools that rewrite mtimes (``cp -p``, rsync,
tar-extract) can perturb eviction order.

Stdlib only (D-deps).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from evalshift.safety import logger
from evalshift.sinks._layout import is_conversation_dir


@dataclass(frozen=True, slots=True)
class _Unit:
    """What GC counts together: one flat capture, or a conversation folder.

    ``files`` are ``(mtime, path)`` pairs, oldest first; the unit's time is its newest file's.
    """

    files: tuple[tuple[float, Path], ...]
    folder: Path | None

    @property
    def newest(self) -> float:
        return self.files[-1][0]


def _scan(directory: Path) -> list[_Unit]:
    units: list[_Unit] = []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith(".json"):
            units.append(_Unit(((entry.stat().st_mtime, Path(entry.path)),), None))
        elif entry.is_dir() and is_conversation_dir(entry.name):
            files = sorted(
                (turn.stat().st_mtime, Path(turn.path))
                for turn in os.scandir(entry.path)
                if turn.is_file() and turn.name.endswith(".json")
            )
            if files:
                units.append(_Unit(tuple(files), Path(entry.path)))
    return sorted(units, key=lambda u: (u.newest, u.files[0][1]))  # oldest first


def evict(
    directory: Path,
    *,
    max_captures: int | None,
    capture_ttl_seconds: float | None,
    now_ts: float,
) -> list[Path]:
    """Evict aged / overflow captures from the suite ``directory``; return the files removed.

    Count rule, newest unit first: the newest unit keeps its newest ``max_captures`` files (all
    of a normal conversation; the latest turns of one that never ends), then older units are
    kept whole while the count stays within ``max_captures`` -- the first that does not fit
    goes, with everything older. TTL rule: every file older than ``capture_ttl_seconds`` goes,
    so an idle conversation goes whole and a live one loses only its old turns. A folder left
    empty is removed. No-op (returns ``[]``) when neither threshold is set or the directory is
    unreadable; never raises into the caller.
    """
    if max_captures is None and capture_ttl_seconds is None:
        return []
    try:
        units = _scan(directory)
    except OSError:
        logger.debug("evalshift: gc scan of %s failed (skipped)", directory, exc_info=True)
        return []

    doomed: set[Path] = set()
    if capture_ttl_seconds is not None:
        doomed.update(
            path
            for unit in units
            for mtime, path in unit.files
            if now_ts - mtime > capture_ttl_seconds
        )
    if max_captures is not None:
        kept = 0
        full = False
        for index in range(len(units) - 1, -1, -1):  # newest first
            files = units[index].files
            if index == len(units) - 1:
                cut = max(len(files) - max_captures, 0)  # never-ending: keep the newest turns
                doomed.update(path for _, path in files[:cut])
                kept = len(files) - cut
            elif not full and kept + len(files) <= max_captures:
                kept += len(files)
            else:
                full = True
                doomed.update(path for _, path in files)

    removed: list[Path] = []
    for unit in units:  # oldest-first, for a stable removal order
        hit = False
        for _, path in unit.files:
            if path not in doomed:
                continue
            hit = True
            try:
                path.unlink(missing_ok=True)
                removed.append(path)
            except OSError:
                logger.debug("evalshift: gc unlink of %s failed (skipped)", path, exc_info=True)
        if hit and unit.folder is not None:
            try:
                unit.folder.rmdir()  # fails, harmlessly, while turns remain
            except OSError:
                logger.debug("evalshift: gc kept %s (not empty)", unit.folder)
    return removed


__all__ = ["evict"]
