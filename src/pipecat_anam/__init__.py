#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam Pipecat plugin: in-process AnamVideoService + optional Daily-egress AnamTransport."""

from typing import Any

from pipecat_anam.video import AnamVideoService

__all__ = [
    "ANAM_AVATAR_USER_NAME",
    "AnamParams",
    "AnamTransport",
    "AnamVideoService",
]

# AnamTransport pulls in pipecat-ai[daily] + daily-python.
# Defer that import until a user actually asks for one of these names so
# non-Daily users (e.g. SmallWebRTC) can keep using AnamVideoService without
# the extra deps. See PEP 562.
_LAZY_TRANSPORT_ATTRS = frozenset({"ANAM_AVATAR_USER_NAME", "AnamParams", "AnamTransport"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_TRANSPORT_ATTRS:
        try:
            from pipecat_anam import transport
        except ImportError as exc:
            raise ImportError(
                f"pipecat_anam.{name} requires the Daily transport extras. "
                "Install with: pip install 'pipecat-ai[daily]'"
            ) from exc
        return getattr(transport, name)
    raise AttributeError(f"module 'pipecat_anam' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
