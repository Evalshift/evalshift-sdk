"""Unit tests for the capture sampling gate (Phase 6)."""

from __future__ import annotations

import pytest

from evalshift.hygiene.sample import conversation_draw, should_capture


def test_none_rate_captures_everything() -> None:
    assert should_capture(None) is True


def test_rate_at_or_above_one_captures_everything() -> None:
    assert should_capture(1.0) is True
    assert should_capture(2.0) is True


def test_rate_at_or_below_zero_captures_nothing() -> None:
    assert should_capture(0.0) is False
    assert should_capture(-0.5) is False


def test_midrate_keeps_when_draw_below_rate() -> None:
    assert should_capture(0.5, rng=lambda: 0.4) is True


def test_midrate_drops_when_draw_at_or_above_rate() -> None:
    assert should_capture(0.5, rng=lambda: 0.6) is False
    # Boundary: draw == rate drops (strict <), keeping the 0.0 case consistent.
    assert should_capture(0.5, rng=lambda: 0.5) is False


@pytest.mark.parametrize(
    ("key", "draw"),
    [("c1", 0.5987745500564885), ("planner:session-1", 0.49559010417828814)],
)
def test_a_conversations_draw_is_a_stable_hash(key: str, draw: float) -> None:
    assert conversation_draw(key) == draw


def test_a_conversation_key_ignores_the_random_draw() -> None:
    assert should_capture(0.59, rng=lambda: 0.0, key="c1") is False  # 0.5988 >= 0.59
    assert should_capture(0.60, rng=lambda: 0.99, key="c1") is True  # 0.5988 < 0.60


def test_rate_bounds_still_win_over_a_key() -> None:
    assert should_capture(None, key="c1") is True
    assert should_capture(1.0, key="c1") is True
    assert should_capture(0.0, key="c1") is False
