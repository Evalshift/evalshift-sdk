"""Conversation folders: where a capture lands, on disk and in a store (0.7.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalshift import capture, configure, record_model_call
from evalshift.capture.span import SpanTree
from evalshift.sinks._layout import (
    conversation_dir,
    is_conversation_dir,
    resolve_layout,
    writing_flat,
)
from evalshift.sinks.file import FileSink
from evalshift.sinks.object_store import ObjectStoreSink
from evalshift.stores.memory import MemoryStore
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import build_capture
from tests.conftest import CaptureReader


def _envelope(conversation_id: str | None = "c1") -> CaptureEnvelope:
    return build_capture(
        SpanTree(),
        suite="support_agent",
        agent_input="hi",
        capture_id="cap_abc",
        conversation_id=conversation_id,
        turn_index=0 if conversation_id is not None else None,
    )


@pytest.mark.parametrize(
    ("conversation_id", "folder"),
    [
        # Pinned in evalshift-cli's tests too: both sides read and write these exact names.
        ("c1", "conv_d0f631ca1ddba8db"),
        ("planner:session-1", "conv_e1961a13f5bd6642"),
        ("", "conv_e3b0c44298fc1c14"),
    ],
)
def test_conversation_dir_is_a_stable_hash_of_the_id(conversation_id: str, folder: str) -> None:
    assert conversation_dir(conversation_id) == folder
    assert is_conversation_dir(folder)


@pytest.mark.parametrize(
    "name", ["nested", "conv_D0F631CA1DDBA8DB", "conv_d0f631ca", "conv_d0f631ca1ddba8db0"]
)
def test_only_the_sdks_own_folder_names_count(name: str) -> None:
    assert not is_conversation_dir(name)


def test_file_sink_files_a_conversation_turn_in_its_folder(tmp_path: Path) -> None:
    path = FileSink(base=tmp_path).write(_envelope())
    folder = tmp_path / "captures" / "support_agent" / "conv_d0f631ca1ddba8db"
    assert path == folder / "cap_abc.json"
    assert path is not None
    assert path.exists()


def test_file_sink_keeps_a_standalone_capture_flat(tmp_path: Path) -> None:
    path = FileSink(base=tmp_path).write(_envelope(None))
    assert path == tmp_path / "captures" / "support_agent" / "cap_abc.json"


def test_object_store_keys_mirror_the_local_layout() -> None:
    store = MemoryStore()
    sink = ObjectStoreSink(store, background=False)
    sink.write(_envelope())
    sink.write(_envelope(None))
    assert set(store.objects) == {
        "captures/support_agent/conv_d0f631ca1ddba8db/cap_abc.json",
        "captures/support_agent/cap_abc.json",
    }


def test_flat_layout_env_restores_the_old_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", " FLAT ")
    assert FileSink(base=tmp_path).write(_envelope()) == (
        tmp_path / "captures" / "support_agent" / "cap_abc.json"
    )
    store = MemoryStore()
    ObjectStoreSink(store, background=False).write(_envelope())
    assert set(store.objects) == {"captures/support_agent/cap_abc.json"}


def test_an_explicit_layout_wins_over_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", "flat")
    path = FileSink(base=tmp_path, layout="conversation").write(_envelope())
    assert path is not None
    assert path.parent.name == "conv_d0f631ca1ddba8db"


@pytest.mark.parametrize("raw", [None, "", "conversation", "Conversation", "nested", "0"])
def test_anything_but_flat_means_conversation(
    raw: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if raw is None:
        monkeypatch.delenv("EVALSHIFT_CAPTURE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("EVALSHIFT_CAPTURE_LAYOUT", raw)
    assert resolve_layout() == "conversation"


def test_a_session_turn_lands_in_its_conversation_folder(
    capturing: Path, read_captures: CaptureReader
) -> None:
    for turn in range(2):
        with capture.agent_session(
            suite="planner",
            redact=False,
            agent_input=f"turn {turn}",
            conversation_id="planner:session-1",
            turn_index=turn,
            tools=[],
        ):
            record_model_call(model_id="claude-opus-4-8", tools=[])
    folder = capturing / "captures" / "planner" / "conv_e1961a13f5bd6642"
    assert len(list(folder.glob("*.json"))) == 2
    assert sorted(c["turn_index"] for c in read_captures("planner")) == [0, 1]


# R10: a decorator's conversation_id is fixed at decoration time, so it names no conversation;
# its captures stay flat (with the id still in the envelope).


def test_writing_flat_overrides_the_layout_only_inside_the_block() -> None:
    with writing_flat():
        assert resolve_layout("conversation") == "flat"
    assert resolve_layout("conversation") == "conversation"


def test_decorator_capture_stays_flat_with_its_conversation_id(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="planner", redact=False, conversation_id="c1", turn_index=0, tools=[])
    def handle(query: str) -> str:
        record_model_call(model_id="claude-opus-4-8", tools=[])
        return query

    handle("hi")
    suite_dir = capturing / "captures" / "planner"
    assert [p.parent for p in suite_dir.rglob("*.json")] == [suite_dir]
    [cap] = read_captures("planner")
    assert cap["conversation_id"] == "c1"
    assert cap["turn_index"] == 0


async def test_async_decorator_capture_stays_flat(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="planner", redact=False, conversation_id="c1", turn_index=0, tools=[])
    async def handle(query: str) -> str:
        record_model_call(model_id="claude-opus-4-8", tools=[])
        return query

    await handle("hi")
    suite_dir = capturing / "captures" / "planner"
    assert [p.parent for p in suite_dir.rglob("*.json")] == [suite_dir]
    [cap] = read_captures("planner")
    assert cap["conversation_id"] == "c1"


def test_decorator_capture_key_is_flat_in_a_store(capturing: Path) -> None:
    store = MemoryStore()
    configure(sink=ObjectStoreSink(store, background=False))

    @capture.agent(suite="planner", redact=False, conversation_id="c1", turn_index=0, tools=[])
    def handle() -> str:
        record_model_call(model_id="claude-opus-4-8", tools=[])
        return "ok"

    handle()
    [key] = [k for k in store.objects if k.startswith("captures/")]
    assert key.startswith("captures/planner/cap_")
    assert key.count("/") == 2


def test_a_session_after_a_decorator_write_still_gets_its_folder(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="planner", redact=False, conversation_id="c1", turn_index=0, tools=[])
    def handle() -> str:
        record_model_call(model_id="claude-opus-4-8", tools=[])
        return "ok"

    handle()
    with capture.agent_session(
        suite="planner",
        redact=False,
        agent_input="turn 1",
        conversation_id="c1",
        turn_index=1,
        tools=[],
    ):
        record_model_call(model_id="claude-opus-4-8", tools=[])
    suite_dir = capturing / "captures" / "planner"
    assert len(list(suite_dir.glob("*.json"))) == 1
    assert len(list((suite_dir / "conv_d0f631ca1ddba8db").glob("*.json"))) == 1
    assert sorted(c["turn_index"] for c in read_captures("planner")) == [0, 1]
