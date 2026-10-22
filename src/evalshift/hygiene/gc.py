"""Capture garbage collection: cap a suite directory by count and/or age.

:func:`evict` runs after a successful disk write, on the **suite** directory just written to. It
caps it by **unit**: a flat ``*.json`` capture, or a whole conversation folder
(``conv_<16 hex>/``, see :mod:`evalshift.sinks._layout`) timed by its newest file. It does a single
:func:`os.scandir` pass over the suite directory (plus one per conversation folder) and orders by
**filesystem mtime** (not the envelope's ``created_at``) — stat only, no JSON parsing, so it stays
cheap even when ``max_captures`` is large. Two rules apply, TTL first:

* **TTL** — evict any ``*.json`` file whose age (``now_ts - mtime``) exceeds
  ``capture_ttl_seconds``, so an idle conversation goes whole and a live one loses only its old
  turns;
* **count** — on the files the TTL leaves: order units newest first by their newest surviving
  file and the files within a unit newest first, then keep the first ``max_captures`` files in
  that order. The newest units are kept whole while they fit, the unit at the boundary keeps its
  newest turns that fit, and every older unit goes whole (so the newest unit, if it is larger
  than the cap, keeps its newest ``max_captures`` turns). A flat-only suite keeps its newest
  ``max_captures``.

A conversation folder left empty is removed. Other subdirectories are never entered. An entry
that cannot be read (a flat file's stat, a folder's listing) is skipped -- never counted, never
removed -- so one unreadable or vanished folder cannot stop the rest of the GC. Each unlink is
individually swallowed, and the suite-directory scan is guarded, so eviction never raises into
the host.
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


def _scan_unit(entry: os.DirEntry[str]) -> _Unit | None:
    """The unit ``entry`` is, or ``None`` for anything GC leaves alone (or a folder now empty)."""
    if entry.is_file() and entry.name.endswith(".json"):
        return _Unit(((entry.stat().st_mtime, Path(entry.path)),), None)
    if entry.is_dir() and is_conversation_dir(entry.name):
        with os.scandir(entry.path) as turns:
            files = sorted(
                (turn.stat().st_mtime, Path(turn.path))
                for turn in turns
                if turn.is_file() and turn.name.endswith(".json")
            )
        return _Unit(tuple(files), Path(entry.path)) if files else None
    return None


def _scan(directory: Path) -> list[_Unit]:
    units: list[_Unit] = []
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                unit = _scan_unit(entry)
            except OSError:  # unreadable or vanished: skipping it only lowers the count
                logger.debug("evalshift: gc skipped %s", entry.path, exc_info=True)
                continue
            if unit is not None:
                units.append(unit)
    return sorted(units, key=lambda u: (u.newest, u.files[0][1]))  # oldest first


def evict(
    directory: Path,
    *,
    max_captures: int | None,
    capture_ttl_seconds: float | None,
    now_ts: float,
) -> list[Path]:
    """Evict aged / overflow captures from the suite ``directory``; return the files removed.

    TTL rule first: every file older than ``capture_ttl_seconds`` goes, so an idle conversation
    goes whole and a live one loses only its old turns. Count rule on what the TTL leaves: units
    ordered newest first (by their newest surviving file), files within a unit newest first, and
    the first ``max_captures`` files in that order are kept -- the newest units whole while they
    fit, the unit at the boundary its newest turns that fit, every older unit none. A folder left
    empty is removed. An entry that cannot be read is skipped (never counted, never removed). No-op
    (returns ``[]``) when neither threshold is set or the directory is unreadable; never raises
    into the caller.
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
        survivors = [
            files for unit in units if (files := tuple(f for f in unit.files if f[1] not in doomed))
        ]
        survivors.sort(key=lambda files: (files[-1][0], files[0][1]), reverse=True)  # newest first
        kept = 0
        for files in survivors:
            for _, path in reversed(files):  # newest turn first
                if kept < max_captures:
                    kept += 1
                else:
                    doomed.add(path)

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
