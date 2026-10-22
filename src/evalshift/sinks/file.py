"""Basic on-disk sink: one ``cap_<id>.json`` per capture.

Path layout is ``<base>/captures/<suite>/<capture_id>.json``, or
``<base>/captures/<suite>/conv_<16 hex>/<capture_id>.json`` for a conversation's turns (see
:mod:`evalshift.sinks._layout`). ``base`` resolves at write time:
an explicit constructor argument wins, else the ``EVALSHIFT_DIR`` env var, else ``.evalshift``
relative to the current working directory. There is deliberately **no** repo-root walk — the SDK
must not assume it runs inside a checkout (containers, Lambda). Resolution order lives in
:mod:`evalshift.sinks._paths`, shared with :class:`~evalshift.sinks.toolset.ToolsetSink`.

Graceful degradation lives in the sink itself: a filesystem ``OSError`` (read-only mount, disk
full, permission denied) is swallowed and logged at ``debug``; the capture is dropped and
``write`` returns ``None`` rather than crashing the host agent (the api layer's
:func:`evalshift.safety.fail_open` stays as a second line of defense). The ``suite`` segment is
sanitized against path traversal so a stray ``../`` cannot escape ``<base>/captures``.

Stdlib only (D-deps).
"""

from __future__ import annotations

import os
from pathlib import Path

from evalshift.safety import logger
from evalshift.sinks._layout import CaptureLayout, conversation_dir, resolve_layout
from evalshift.sinks._paths import resolve_base
from evalshift.trace.models import CaptureEnvelope
from evalshift.trace.serialize import capture_filename, dumps


def _safe_segment(name: str) -> str:
    """Reduce ``name`` to a single safe path segment (no separators, no ``..`` escape)."""
    cleaned = name.replace("\\", "/").replace("/", "_")
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip(". ")
    return cleaned or "_"


def capture_relpath(envelope: CaptureEnvelope, *, layout: CaptureLayout) -> str:
    """``<safe_suite>/[conv_<16 hex>/]<capture_id>.json`` -- the same under ``captures/`` on
    disk and in a store (see :mod:`evalshift.sinks._layout`)."""
    suite = _safe_segment(envelope.suite)
    name = capture_filename(envelope.capture_id)
    if layout == "conversation" and envelope.conversation_id is not None:
        return f"{suite}/{conversation_dir(envelope.conversation_id)}/{name}"
    return f"{suite}/{name}"


class FileSink:
    """Write capture envelopes to ``<base>/captures/<suite>/<capture_id>.json``, or
    ``<base>/captures/<suite>/conv_<16 hex>/<capture_id>.json`` for a conversation's turns
    (see :mod:`evalshift.sinks._layout`)."""

    def __init__(
        self, base: str | os.PathLike[str] | None = None, *, layout: CaptureLayout | None = None
    ) -> None:
        self._base = base
        self._layout = layout

    @property
    def base(self) -> str | os.PathLike[str] | None:
        """The raw ``base`` constructor argument (``None`` means "use the default resolution").

        Exposed so other on-disk bookkeeping that must land under the same root as this sink's
        captures -- currently just :class:`~evalshift.sinks.toolset.ToolsetSink`'s sidecar, via
        :func:`evalshift.config._toolset_base` -- can match it exactly, instead of independently
        re-resolving ``EVALSHIFT_DIR``/CWD and silently landing somewhere else when this sink was
        constructed with an explicit ``base=``.
        """
        return self._base

    def _resolve_base(self) -> Path:
        return resolve_base(self._base)

    def suite_dir(self, envelope: CaptureEnvelope) -> Path:
        """``<base>/captures/<safe_suite>`` -- what GC caps, whichever folder a write landed in."""
        return (self._resolve_base() / "captures" / _safe_segment(envelope.suite)).absolute()

    def write(self, envelope: CaptureEnvelope) -> Path | None:
        """Serialize ``envelope`` and write it; create parent dirs; return the path written.

        The returned path is absolute (the default ``.evalshift`` base is anchored to the CWD)
        so callers get an unambiguous location regardless of how ``base`` was specified. Returns
        ``None`` when the write degrades on a filesystem ``OSError`` (read-only / disk full).
        """
        path = (
            self._resolve_base()
            / "captures"
            / capture_relpath(envelope, layout=resolve_layout(self._layout))
        ).absolute()
        target_dir = path.parent
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(dumps(envelope), encoding="utf-8")
        except OSError:
            logger.debug("evalshift: capture write to %s failed (dropped)", path, exc_info=True)
            return None
        return path


__all__ = ["FileSink"]
