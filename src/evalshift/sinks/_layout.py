"""Where a capture lives under ``captures/``: in its conversation's folder, or flat.

A capture with a ``conversation_id`` goes to ``<safe_suite>/conv_<16 hex>/<capture_id>.json``,
where the hex is the start of ``sha256(conversation_id)``: fixed length, lowercase, safe on every
filesystem, and never the raw id. Every other capture goes to ``<safe_suite>/<capture_id>.json``.
The layout is the same under ``<base>/captures/`` and under a store's prefix, so a bucket stays a
mirror of ``.evalshift/``. ``EVALSHIFT_CAPTURE_LAYOUT=flat`` (or ``layout="flat"`` on a sink)
writes every capture flat, as before 0.7.0. The evalshift CLI reads both layouts from 1.6.0.
Captures from the ``@capture.agent`` decorator always stay flat (:func:`writing_flat`): its
``conversation_id`` is fixed when the function is decorated, so it names no conversation.

**Co-install guard.** An older evalshift CLI cannot see conversation folders, and a CLI 1.5.x
install can pull this SDK in without the user choosing it. So when neither a sink's ``layout=``
nor the env var picks a layout, and an evalshift CLI older than 1.6 is installed in this
environment, captures are written flat and one warning is logged per process.
``EVALSHIFT_CAPTURE_LAYOUT=conversation`` opts in anyway. A CLI in another environment (CI pinned
to 1.5, say) cannot be detected.

Stdlib only (D-deps).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import metadata
from typing import Literal

CaptureLayout = Literal["conversation", "flat"]

#: Env var selecting the layout: ``flat`` or ``conversation`` (any case, stripped).
LAYOUT_ENV = "EVALSHIFT_CAPTURE_LAYOUT"

#: The first evalshift CLI (``major``, ``minor``) that reads conversation folders.
_FIRST_FOLDER_CLI = (1, 6)

#: The leading ``major.minor`` of a version string; ``1.6.0rc1`` and ``1.6.0.dev0`` count as 1.6.
_VERSION = re.compile(r"(\d+)\.(\d+)")

#: The SDK's logger (the one :mod:`evalshift.safety` uses), by name: this module is stdlib-only.
_log = logging.getLogger("evalshift")

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


def _cli_version() -> str:
    """The installed evalshift CLI's version; raises ``PackageNotFoundError`` when it is absent.

    The CLI distribution is ``evalshift`` (this SDK's is ``evalshift-sdk``).
    """
    return metadata.version("evalshift")


@functools.cache
def _guard_layout() -> CaptureLayout:
    """``flat`` when an evalshift CLI older than 1.6 is installed alongside, else ``conversation``.

    Computed once per process (``cache_clear()`` resets it), so the one warning is logged once.
    Anything unreadable -- no CLI, an odd version string, a failing lookup -- means no guard.
    """
    try:
        version = _cli_version()
        match = _VERSION.match(version)
        if match is None or (int(match[1]), int(match[2])) >= _FIRST_FOLDER_CLI:
            return "conversation"
        _log.warning(
            "evalshift: the installed evalshift CLI %s can't read conversation folders, so "
            "captures are written flat; upgrade the CLI to 1.6.0 or set "
            "EVALSHIFT_CAPTURE_LAYOUT=conversation",
            version,
        )
        return "flat"
    except Exception:  # never let the guard stop a capture
        return "conversation"


def resolve_layout(explicit: CaptureLayout | None = None) -> CaptureLayout:
    """The layout to write, first match wins.

    1. ``flat`` inside :func:`writing_flat`;
    2. ``explicit``, when a sink was given ``layout=``;
    3. the env var, when it says ``flat`` or ``conversation`` (any case, stripped) -- so
       ``conversation`` is the explicit opt-in that overrides the guard;
    4. else (unset, empty or a typo) the co-install guard: ``flat`` when an evalshift CLI older
       than 1.6 is installed in this environment (one warning per process), else
       ``conversation``.

    Never raises, so a bad value can never stop captures. A sink must call this synchronously
    inside ``write()``: :func:`writing_flat` is a context variable, so a call made later (on a
    background thread, say) would not see it.
    """
    if _FLAT_WRITE.get():
        return "flat"
    if explicit is not None:
        return explicit
    raw = os.environ.get(LAYOUT_ENV, "").strip().lower()
    if raw == "flat":
        return "flat"
    if raw == "conversation":
        return "conversation"
    return _guard_layout()


__all__ = [
    "LAYOUT_ENV",
    "CaptureLayout",
    "conversation_dir",
    "is_conversation_dir",
    "resolve_layout",
    "writing_flat",
]
