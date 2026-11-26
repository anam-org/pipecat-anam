#
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run the real anam SDK objects with only the network patched out.

``AnamClient``, ``Session``, ``StreamingClient`` and ``SignallingClient`` are the real
classes. Only the API session start and the signalling/WebRTC connect are replaced, so
the SDK's own end path (``_terminate`` → teardown → ``CONNECTION_CLOSED`` listeners)
runs against the real pipecat-anam listeners. No socket or peer connection is opened.
"""

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from anam import AnamClient
from anam._api import CoreApiClient
from anam._signalling import SignallingClient
from anam._streaming import StreamingClient
from anam.errors import SessionError
from anam.types import ConnectionClosedCode, SessionInfo


def make_session_info() -> SessionInfo:
    return SessionInfo(
        session_id="real-sdk-session",
        engine_host="engine.test.invalid",
        engine_protocol="https",
        signalling_endpoint="/ws",
        heartbeat_interval_seconds=5,
        max_reconnection_attempts=5,
    )


@contextlib.contextmanager
def patch_sdk_network(
    session_info: SessionInfo | None = None,
    *,
    end_during_connect: bool = False,
    ready_during_connect: bool = False,
) -> Iterator[None]:
    """Patch the API session start and ``StreamingClient.connect``.

    The fake ``connect`` builds the real ``SignallingClient`` exactly as the SDK does but
    never opens it, then marks the streaming client started.

    ``end_during_connect`` ends the session inside ``connect`` the way the SDK does when
    the signalling fails during startup: it closes and raises ``SessionError``.
    ``ready_during_connect`` delivers ``SESSION_READY`` before ``connect`` returns.
    """
    info = session_info or make_session_info()

    async def start_session(self: CoreApiClient, *args: Any, **kwargs: Any) -> SessionInfo:
        return info

    async def connect(self: StreamingClient, timeout: float = 30.0) -> None:
        self._signalling_client = SignallingClient(
            session_info=self._session_info,
            on_message=self._handle_signal_message,
            on_open=self._on_signalling_open,
            on_end=self._terminate,
            on_talk_stream_delivery_failed=self._report_failure,
        )
        if end_during_connect:
            self._terminate(ConnectionClosedCode.SIGNALLING_FAILURE, "x")
            await self.close()
            raise SessionError("Session ended while connecting")
        if ready_during_connect:
            assert self._on_session_ready is not None
            await self._on_session_ready()
            for _ in range(5):
                await asyncio.sleep(0)
        self._started = True

    with (
        patch.object(CoreApiClient, "start_session", start_session),
        patch.object(StreamingClient, "connect", connect),
    ):
        yield


def end_remotely(
    client: AnamClient, code: ConnectionClosedCode, reason: str | None
) -> asyncio.Task[None]:
    """End the session through the SDK's own end path and return its end task."""
    streaming_client = client._streaming_client
    assert streaming_client is not None
    streaming_client._terminate(code, reason)
    end_task = streaming_client._end_task
    assert end_task is not None
    return end_task
