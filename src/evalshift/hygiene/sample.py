"""Capture sampling: decide whether to record this agent run.

The decision is made **once, at agent entry** (before any span tree is built), so an unsampled
run degrades to the same near-pure pass-through as the gate-off path — no tree, no redaction, no
serialization. With a conversation id the draw is a hash of it, so a conversation is captured
whole or not at all, in every process. The draw source ``rng`` is injectable so tests are
deterministic.

Stdlib only (D-deps).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable


def conversation_draw(key: str) -> float:
    """A stable draw in ``[0, 1)`` for ``key``: 64 bits of ``sha256("evalshift-sample:" + key)``.

    Every turn of a conversation, in every process, gets the same draw, so a sample rate keeps or
    drops a conversation whole.
    """
    digest = hashlib.sha256(f"evalshift-sample:{key}".encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def should_capture(
    rate: float | None, *, rng: Callable[[], float] | None = None, key: str | None = None
) -> bool:
    """True iff this run should be captured under ``rate`` (a probability in ``[0, 1]``).

    ``None`` or ``rate >= 1`` always captures; ``rate <= 0`` never captures. Otherwise, with a
    ``key`` (a conversation id) capture iff :func:`conversation_draw` of it falls below ``rate``;
    without one, iff a uniform draw ``rng()`` does (strict ``<``, so ``rate == draw`` drops).

    ``rng`` defaults to :func:`random.random`, resolved at call time so tests can monkeypatch it.
    """
    if rate is None or rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    if key is not None:
        return conversation_draw(key) < rate
    draw = rng if rng is not None else random.random
    return draw() < rate


__all__ = ["conversation_draw", "should_capture"]
