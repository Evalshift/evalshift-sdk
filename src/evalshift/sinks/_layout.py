"""Where a capture lives under ``captures/``: in its conversation's folder, or flat.

A capture with a ``conversation_id`` goes to ``<safe_suite>/conv_<16 hex>/<capture_id>.json``,
where the hex is the start of ``sha256(conversation_id)``: fixed length, lowercase, safe on every
filesystem, and never the raw id. Every other capture goes to ``<safe_suite>/<capture_id>.json``.
The layout is the same under ``<base>/captures/`` and under a store's prefix, so a bucket stays a
mirror of ``.evalshift/``. ``EVALSHIFT_CAPTURE_LAYOUT=flat`` (or ``layout="flat"`` on a sink)
writes every capture flat, as before 0.7.0. The evalshift CLI reads both layouts from 1.6.0.
Captures from the ``@capture.agent`` decorator always stay flat (:func:`writing_flat`): its
``conversation_id`` is fixed when the function is decorated, so it names no conversation.

Stdlib only (D-deps).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

CaptureLayout = Literal["conversation", "flat"]

#: Env var selecting the layout; only ``flat`` (any case, stripped) changes it.
LAYOUT_ENV = "EVALSHIFT_CAPTURE_LAYOUT"

_CONVERSATION_DIR = re.compile(r"conv_[0-9a-f]{16}")

_FLAT_WRITE: ContextVar[bool] = ContextVar("evalshift_flat_write", default=False)


def conversation_dir(conversation_id: str) -> str:
    """The folder a conversation's turns live in: ``conv_`` + 16 hex digits of its sha256.

    ``str()`` first: the envelope types the id as ``str``, but nothing enforces it, and a stray
    int must not turn into a dropped capture.
    """
    return "conv_" + hashlib.sha256(str(conversation_id).encode("utf-8")).hexdigest()[:16]


def is_conversation_dir(name: str) -> bool:
    """Whether ``name`` is a folder :func:`conversation_dir` produces."""
    return _CONVERSATION_DIR.fullmatch(name) is not None


@contextmanager
def writing_flat() -> Iterator[None]:
    """Write the captures inside this block flat, whatever the layout says.

    The ``@capture.agent`` decorator wraps its sink write in this: its ``conversation_id`` is
    fixed at decoration time, so it names no conversation and its runs never share a folder.
    """
    token = _FLAT_WRITE.set(True)
    try:
        yield
    finally:
        _FLAT_WRITE.reset(token)


def resolve_layout(explicit: CaptureLayout | None = None) -> CaptureLayout:
    """The layout to write: ``flat`` inside :func:`writing_flat`, else ``explicit`` if given,
    else ``flat`` when the env var says so.

    Anything other than ``flat`` -- unset, empty, ``conversation`` or a typo -- means
    ``conversation``, the default, so a bad value can never stop captures.
    """
    if _FLAT_WRITE.get():
        return "flat"
    if explicit is not None:
        return explicit
    return "flat" if os.environ.get(LAYOUT_ENV, "").strip().lower() == "flat" else "conversation"


__all__ = [
    "LAYOUT_ENV",
    "CaptureLayout",
    "conversation_dir",
    "is_conversation_dir",
    "resolve_layout",
    "writing_flat",
]
