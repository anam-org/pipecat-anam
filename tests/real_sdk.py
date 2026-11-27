#
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import anam._signalling
from anam import AnamClient
from anam._api import CoreApiClient
from anam._signalling import SignalAction, SignallingClient
from anam._streaming import StreamingClient
from anam.types import ConnectionClosedCode, SessionInfo
from websockets.protocol import State


def make_session_info() -> SessionInfo:
    return SessionInfo(
        session_id="real-sdk-session",
        engine_host="engine.test.invalid",
        engine_protocol="https",
        signalling_endpoint="/ws",
        heartbeat_interval_seconds=5,
        max_reconnection_attempts=5,
    )


def _stub_start_session(info: SessionInfo) -> contextlib.AbstractContextManager[Any]:
    async def start_session(self: CoreApiClient, *args: Any, **kwargs: Any) -> SessionInfo:
        return info

    return patch.object(CoreApiClient, "start_session", start_session)


@contextlib.contextmanager
def patch_sdk_network(
    session_info: SessionInfo | None = None, *, open_signalling: bool = False
) -> Iterator[None]:
    async def connect(self: StreamingClient, timeout: float = 30.0) -> None:
        # No on_open: it would start the peer connection, which these tests do not fake.
        self._signalling_client = SignallingClient(
            session_info=self._session_info,
            on_message=self._handle_signal_message,
            on_end=self._terminate,
            on_talk_stream_delivery_failed=self._report_failure,
        )
        if open_signalling:
            await self._signalling_client.connect()
        self._started = True

    with (
        _stub_start_session(session_info or make_session_info()),
        patch.object(StreamingClient, "connect", connect),
    ):
        yield


async def end_on_signalling_open(self: StreamingClient) -> None:
    self._terminate(ConnectionClosedCode.SIGNALLING_FAILURE, "x")


async def ready_on_signalling_open(self: StreamingClient) -> None:
    await self._handle_signal_message({"actionType": SignalAction.SESSION_READY.value})
    # Session-ready and ICE connected are separate events, and connect waits between them.
    await asyncio.sleep(0)
    self._mark_connected()


@contextlib.contextmanager
def real_sdk_connect(
    on_signalling_open: Callable[[StreamingClient], Awaitable[None]],
) -> Iterator[None]:
    """Run the real StreamingClient.connect; on_signalling_open stands in for the WebRTC offer."""
    with (
        _stub_start_session(make_session_info()),
        fake_signalling_sockets(),
        patch.object(StreamingClient, "_on_signalling_open", on_signalling_open),
    ):
        yield


def end_remotely(
    client: AnamClient, code: ConnectionClosedCode, reason: str | None
) -> asyncio.Task[None]:
    streaming_client = client._streaming_client
    assert streaming_client is not None
    streaming_client._terminate(code, reason)
    end_task = streaming_client._end_task
    assert end_task is not None
    return end_task


class FakeSocket:
    """A websockets ClientConnection stand-in that records what is sent and closes on drop()."""

    def __init__(self) -> None:
        self.state = State.OPEN
        self.close_code: int | None = None
        self.close_reason = ""
        self.sent: list[dict[str, Any]] = []
        self.transport = SimpleNamespace(abort=self.drop)
        self._closed = asyncio.Event()

    def __aiter__(self) -> "FakeSocket":
        return self

    async def __anext__(self) -> str:
        await self._closed.wait()
        raise StopAsyncIteration

    async def send(self, message: str) -> None:
        assert self.state is State.OPEN
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        self.drop(1000)

    def drop(self, code: int = 1006) -> None:
        if self.state is State.CLOSED:
            return
        self.state = State.CLOSED
        self.close_code = code
        self._closed.set()

    def sent_actions(self, action: str) -> list[dict[str, Any]]:
        return [message for message in self.sent if message.get("actionType") == action]


@contextlib.contextmanager
def fake_signalling_sockets() -> Iterator[list[FakeSocket]]:
    """Replace the SDK's websockets connect; each connect opens a new FakeSocket."""
    sockets: list[FakeSocket] = []

    async def connect(url: str, **kwargs: Any) -> FakeSocket:
        socket = FakeSocket()
        sockets.append(socket)
        return socket

    with (
        patch.object(anam._signalling, "connect", connect),
        patch.object(SignallingClient, "RECONNECT_BACKOFF_SECONDS", 0.01),
    ):
        yield sockets
