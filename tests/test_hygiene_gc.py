"""Unit tests for capture garbage collection (Phase 6)."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import pytest

from evalshift.hygiene import gc


def _make(directory: Path, name: str, *, mtime: float) -> Path:
    path = directory / name
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_count_eviction_keeps_newest_n(tmp_path: Path) -> None:
    files = [_make(tmp_path, f"cap_{i}.json", mtime=100.0 + i) for i in range(5)]
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=None, now_ts=1000.0)
    assert set(removed) == set(files[:3])  # three oldest gone
    survivors = set(tmp_path.glob("*.json"))
    assert survivors == set(files[3:])  # two newest kept


def test_max_captures_one_keeps_only_newest(tmp_path: Path) -> None:
    old = _make(tmp_path, "cap_old.json", mtime=100.0)
    new = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [old]
    assert new.exists()


def test_ttl_eviction_removes_aged_files(tmp_path: Path) -> None:
    fresh = _make(tmp_path, "cap_fresh.json", mtime=990.0)
    stale = _make(tmp_path, "cap_stale.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=60.0, now_ts=1000.0)
    assert removed == [stale]  # age 900 > ttl 60
    assert fresh.exists()  # age 10 < ttl 60


def test_count_and_ttl_rules_union(tmp_path: Path) -> None:
    files = [_make(tmp_path, f"cap_{i}.json", mtime=m) for i, m in enumerate([100.0, 980.0, 995.0])]
    # max_captures=2 would drop the oldest (files[0]); ttl=60 also drops files[0] (age 900).
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=60.0, now_ts=1000.0)
    assert set(removed) == {files[0]}
    assert {files[1], files[2]} == set(tmp_path.glob("*.json"))


def test_only_json_files_are_touched(tmp_path: Path) -> None:
    keep_txt = _make(tmp_path, "notes.txt", mtime=100.0)
    old_json = _make(tmp_path, "cap_old.json", mtime=100.0)
    new_json = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [old_json]
    assert keep_txt.exists()  # non-json never considered
    assert new_json.exists()


def test_no_thresholds_evicts_nothing(tmp_path: Path) -> None:
    _make(tmp_path, "cap_a.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == []
    assert list(tmp_path.glob("*.json"))


def test_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=60.0, now_ts=1000.0) == []


def test_missing_dir_does_not_raise(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert gc.evict(missing, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0) == []


def test_gc_never_touches_a_sibling_toolsets_directory(tmp_path: Path) -> None:
    """Regression for D-toolset: ``evict`` is called with ``<base>/captures/<suite>/`` (see
    ``sinks/hygiene.py``), a sibling of ``<base>/toolsets/`` -- never that directory itself or an
    ancestor of it -- and ``evict`` never recurses into subdirectories (module docstring). Proves
    both halves structurally: pointing ``evict`` at a captures/<suite> dir leaves a toolsets/ dir
    one level up completely untouched, even when it contains files that would sort as "oldest".
    """
    base = tmp_path
    captures_suite_dir = base / "captures" / "demo"
    captures_suite_dir.mkdir(parents=True)
    toolsets_dir = base / "toolsets"
    toolsets_dir.mkdir()

    sidecar = _make(toolsets_dir, "deadbeef.json", mtime=1.0)  # oldest possible mtime
    _make(captures_suite_dir, "cap_a.json", mtime=100.0)
    newer = _make(captures_suite_dir, "cap_b.json", mtime=200.0)

    removed = gc.evict(captures_suite_dir, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)

    assert removed == [captures_suite_dir / "cap_a.json"]
    assert newer.exists()
    assert sidecar.exists()  # never scanned: not in captures_suite_dir, not recursed into
    assert list(toolsets_dir.glob("*.json")) == [sidecar]


def test_unlink_failure_on_one_file_does_not_abort_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _make(tmp_path, "cap_a.json", mtime=100.0)  # oldest
    b = _make(tmp_path, "cap_b.json", mtime=110.0)
    _make(tmp_path, "cap_c.json", mtime=200.0)  # newest, kept

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "cap_a.json":
            raise PermissionError("locked")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    removed = gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [b]  # b removed despite a failing
    assert a.exists()  # the failing one survives, no exception raised


CONV_A = "conv_" + "a" * 16
CONV_B = "conv_" + "b" * 16
CONV_C = "conv_" + "c" * 16


def _turns(directory: Path, folder: str, *mtimes: float) -> list[Path]:
    conv = directory / folder
    conv.mkdir(parents=True, exist_ok=True)
    return [_make(conv, f"cap_{folder[-2:]}_{i}.json", mtime=m) for i, m in enumerate(mtimes)]


def test_the_boundary_conversation_keeps_its_newest_turns_and_older_ones_go_whole(
    tmp_path: Path,
) -> None:
    oldest = _turns(tmp_path, CONV_C, 50.0, 60.0)
    boundary = _turns(tmp_path, CONV_A, 100.0, 101.0, 102.0)
    new = _turns(tmp_path, CONV_B, 200.0, 201.0)
    removed = gc.evict(tmp_path, max_captures=4, capture_ttl_seconds=None, now_ts=1000.0)
    # B is kept whole (2), A keeps the 2 newest turns that still fit, C goes whole.
    assert set(removed) == {*oldest, boundary[0]}
    assert not (tmp_path / CONV_C).exists()
    assert boundary[1].exists() and boundary[2].exists()
    assert all(p.exists() for p in new)


def test_a_conversation_larger_than_the_cap_keeps_only_its_newest_turns(tmp_path: Path) -> None:
    # A conversation that never ends (a static decorator id, a user id): disk stays bounded.
    live = _turns(tmp_path, CONV_A, 300.0, 301.0, 302.0)
    older = _make(tmp_path, "cap_flat.json", mtime=100.0)
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [older, live[0]]
    assert live[1].exists() and live[2].exists()


def test_ttl_evicts_every_old_file_so_an_idle_conversation_goes_whole(tmp_path: Path) -> None:
    idle = _turns(tmp_path, CONV_A, 100.0, 110.0)
    active = _turns(tmp_path, CONV_B, 100.0, 990.0)  # first turn old, latest recent
    removed = gc.evict(tmp_path, max_captures=None, capture_ttl_seconds=500.0, now_ts=1000.0)
    assert set(removed) == {*idle, active[0]}
    assert not (tmp_path / CONV_A).exists()
    assert active[1].exists()


def test_flat_and_conversation_units_share_one_cap_oldest_first(tmp_path: Path) -> None:
    flat_old = _make(tmp_path, "cap_old.json", mtime=50.0)
    conv = _turns(tmp_path, CONV_A, 100.0, 150.0)
    flat_new = _make(tmp_path, "cap_new.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=3, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [flat_old]
    assert flat_new.exists() and all(p.exists() for p in conv)


def test_a_folder_that_is_not_a_conversation_is_never_touched(tmp_path: Path) -> None:
    other = tmp_path / "nested"
    other.mkdir()
    keep = _make(other, "cap_x.json", mtime=1.0)
    _make(tmp_path, "cap_new.json", mtime=200.0)
    gc.evict(tmp_path, max_captures=1, capture_ttl_seconds=1.0, now_ts=1000.0)
    assert keep.exists()


def test_the_count_is_of_what_the_ttl_leaves(tmp_path: Path) -> None:
    # A's turn at 100 is past the TTL; counting it would push A out beside the flat capture.
    a = _turns(tmp_path, CONV_A, 100.0, 900.0)
    flat = _make(tmp_path, "cap_flat.json", mtime=950.0)
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=500.0, now_ts=1000.0)
    assert removed == [a[0]]
    assert a[1].exists() and flat.exists()


def test_one_newer_capture_does_not_evict_a_conversation_at_the_cap(tmp_path: Path) -> None:
    turns = _turns(tmp_path, CONV_A, *(100.0 + i for i in range(5)))
    flat = _make(tmp_path, "cap_flat.json", mtime=200.0)
    removed = gc.evict(tmp_path, max_captures=5, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [turns[0]]  # the conversation keeps its newest cap - 1 turns
    assert flat.exists() and all(p.exists() for p in turns[1:])


def _scandir_failing_for(name: str, *, vanish: bool = False) -> Any:
    real = os.scandir

    def fake(path: Any = ".") -> Any:
        if isinstance(path, (str, os.PathLike)) and Path(path).name == name:
            if not vanish:
                raise PermissionError(f"denied: {path}")
            shutil.rmtree(path)  # gone between the suite scan and this one
        return real(path)

    return fake


@pytest.mark.parametrize("vanish", [False, True])
def test_an_unreadable_or_vanished_folder_does_not_stop_the_rest_of_the_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vanish: bool
) -> None:
    blocked = _turns(tmp_path, CONV_B, 10.0, 20.0)  # oldest; skipped, never counted
    old = _make(tmp_path, "cap_old.json", mtime=100.0)
    conv = _turns(tmp_path, CONV_A, 150.0, 160.0)
    new = _make(tmp_path, "cap_new.json", mtime=200.0)
    monkeypatch.setattr(os, "scandir", _scandir_failing_for(CONV_B, vanish=vanish))
    removed = gc.evict(tmp_path, max_captures=2, capture_ttl_seconds=None, now_ts=1000.0)
    assert removed == [old, conv[0]]
    assert new.exists() and conv[1].exists()
    assert all(p.exists() for p in blocked) is not vanish


def _keep_newest_after_ttl(
    files: list[tuple[float, Path]], cap: int | None, ttl: float | None, now: float
) -> set[Path]:
    alive = sorted(f for f in files if ttl is None or now - f[0] <= ttl)
    return {path for _, path in (alive if cap is None else alive[max(len(alive) - cap, 0) :])}


def test_flat_only_suites_keep_the_newest_n_after_the_ttl(tmp_path: Path) -> None:
    rng = random.Random(20261021)
    for case in range(300):
        directory = tmp_path / f"case_{case}"
        directory.mkdir()
        # Integer mtimes from a small range force ties, so the tie order is checked too.
        files = [
            (m, _make(directory, f"cap_{i:03d}.json", mtime=m))
            for i, m in enumerate(float(rng.randint(0, 50)) for _ in range(rng.randint(0, 25)))
        ]
        cap = rng.choice([None, *range(1, 30)])
        ttl = rng.choice([None, float(rng.randint(0, 60))])
        if cap is None and ttl is None:
            cap = rng.randint(1, 30)
        now = 50.0
        expected = _keep_newest_after_ttl(files, cap, ttl, now)
        removed = gc.evict(directory, max_captures=cap, capture_ttl_seconds=ttl, now_ts=now)
        survivors = set(directory.glob("*.json"))
        assert survivors == expected, (case, cap, ttl)
        assert set(removed) == {p for _, p in files} - expected, (case, cap, ttl)
