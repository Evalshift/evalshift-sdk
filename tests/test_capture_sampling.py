"""End-to-end capture sampling across all four agent entry points (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure, record_model_call
from evalshift.capture import state
from tests.conftest import CaptureReader


def test_sample_rate_zero_writes_nothing_sync(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle(query: str) -> str:
        return f"ok:{query}"

    assert handle("hi") == "ok:hi"  # host result unaffected
    assert read_captures("samp") == []


async def test_sample_rate_zero_writes_nothing_async(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)

    @capture.agent(suite="samp", redact=False, tools=[])
    async def handle(query: str) -> str:
        return f"ok:{query}"

    assert await handle("hi") == "ok:hi"
    assert read_captures("samp") == []


def test_sample_rate_zero_session_yields_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)
    with capture.agent_session(suite="samp", redact=False, tools=[]) as tree:
        assert tree is None
    assert read_captures("samp") == []


async def test_sample_rate_zero_async_session_yields_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    configure(sample_rate=0.0)
    async with capture.agent_session_async(suite="samp", redact=False, tools=[]) as tree:
        assert tree is None
    assert read_captures("samp") == []


def test_sample_rate_one_writes_one(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=1.0)

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1


def test_mid_rate_keeps_when_draw_below(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("random.random", lambda: 0.1)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1


def test_mid_rate_drops_when_draw_above(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("random.random", lambda: 0.9)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle() -> str:
        return "ok"

    handle()
    assert read_captures("samp") == []


def test_unsampled_run_is_true_passthrough(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=0.0)
    observed: dict[str, object] = {}

    @capture.tool
    def mytool() -> str:
        observed["tree"] = state.current_tree()
        return "t"

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle() -> str:
        record_model_call(model_id="m", tools=[])
        return mytool()

    assert handle() == "t"
    assert observed["tree"] is None  # no session opened -> tools/model calls no-op
    assert read_captures("samp") == []


def test_sampling_fault_still_captures(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> float:
        raise RuntimeError("rng broke")

    monkeypatch.setattr("random.random", boom)
    configure(sample_rate=0.5)

    @capture.agent(suite="samp", redact=False, tools=[])
    def handle() -> str:
        return "ok"

    handle()
    assert len(read_captures("samp")) == 1


def test_a_conversation_is_sampled_whole(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rate 0.55: "planner:session-1" draws 0.4956 (kept), "c1" draws 0.5988 (dropped) -- on every
    # turn, whatever random.random says.
    configure(sample_rate=0.55)
    draws = iter([0.0, 0.99, 0.0, 0.99, 0.0, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))
    for conversation in ("planner:session-1", "c1"):
        for turn in range(3):
            with capture.agent_session(
                suite="s",
                redact=False,
                agent_input=f"{conversation} {turn}",
                conversation_id=conversation,
                turn_index=turn,
                tools=[],
            ):
                record_model_call(model_id="m", tools=[])
    kept = read_captures("s")
    assert {c["conversation_id"] for c in kept} == {"planner:session-1"}
    assert sorted(c["turn_index"] for c in kept) == [0, 1, 2]


async def test_an_async_conversation_is_sampled_whole(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The async twin of the test above: same rate, same ids, same random draws.
    configure(sample_rate=0.55)
    draws = iter([0.0, 0.99, 0.0, 0.99, 0.0, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))
    for conversation in ("planner:session-1", "c1"):
        for turn in range(3):
            async with capture.agent_session_async(
                suite="s",
                redact=False,
                agent_input=f"{conversation} {turn}",
                conversation_id=conversation,
                turn_index=turn,
                tools=[],
            ):
                record_model_call(model_id="m", tools=[])
    kept = read_captures("s")
    assert {c["conversation_id"] for c in kept} == {"planner:session-1"}
    assert sorted(c["turn_index"] for c in kept) == [0, 1, 2]


def test_the_decorators_static_conversation_id_samples_at_random(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The decorator's id is fixed at decoration time: hashing it would keep every run or none.
    configure(sample_rate=0.55)
    draws = iter([0.0, 0.99, 0.0, 0.99])
    monkeypatch.setattr("random.random", lambda: next(draws))

    @capture.agent(suite="s", redact=False, tools=[], conversation_id="c1")
    def run(question: str) -> str:
        record_model_call(model_id="m", tools=[])
        return question

    for i in range(4):
        run(f"q{i}")
    assert len(read_captures("s")) == 2  # the two runs whose random draw was 0.0
