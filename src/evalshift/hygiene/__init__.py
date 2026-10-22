"""Capture hygiene: sampling, dedup, and garbage collection (Phase 6).

These controls keep a busy agent's capture firehose self-managing. Dedup and a ``max_captures``
cap of 200 per suite directory are **on by default**; ``sample_rate`` and ``capture_ttl`` are
opt-in (``docs/DECISIONS.md`` D-6a). ``EVALSHIFT_MAX_CAPTURES=0 EVALSHIFT_DEDUP=off`` restores
unbounded capture.

* :mod:`evalshift.hygiene.sample` — capture a fraction of agent runs, decided once per agent
  invocation; per conversation when an ``agent_session`` is given a ``conversation_id``.
* :mod:`evalshift.hygiene.dedup` — collapse captures with identical inputs (per-process registry).
* :mod:`evalshift.hygiene.gc` — cap a suite directory by count / age, by unit: a capture or a
  whole conversation folder, evicting the oldest.

Stdlib only (D-deps).
"""

from __future__ import annotations
