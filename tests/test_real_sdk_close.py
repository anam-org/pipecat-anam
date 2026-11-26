#
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Close paths of ``AnamTransport`` and ``AnamVideoService`` on the real anam SDK.

anam 0.13 makes ``Session.close()`` wait for the ``CONNECTION_CLOSED`` listeners, so a
listener that waits for a task calling ``close()`` deadlocks. These tests run the real
SDK end path (see ``real_sdk``) and bound every SDK end or close with a timeout, so a
deadlock fails the test instead of hanging the suite.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anam import AnamClient, AnamEvent, Session
from anam.types import ConnectionClosedCode
from conftest import (
    _blocking_frames,
    _Harness,
    _make_transport,
    _make_video_service,
    make_client,
)
from pipecat.frames.frames import StartFrame, TTSAudioRawFrame, TTSStartedFrame
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.filters.identity_filter import IdentityFilter
from real_sdk import end_remotely, patch_sdk_network

from pipecat_anam import AnamVideoService
from pipecat_anam.transport import AnamTransportClient


async def _connected_client(
    client: AnamTransportClient | None = None,
) -> tuple[AnamTransportClient, Session, AnamClient]:
    client = client or make_client()
    with patch_sdk_network():
        await asyncio.wait_for(client._anam_connect(), 5)
    sdk_session = client.session
    assert isinstance(sdk_session, Session)
    assert sdk_session.is_active
    return client, sdk_session, sdk_session._client


# --- Negative control -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_sdk_harness_detects_listener_waiting_on_closer() -> None:
    """A listener that awaits a task calling ``close()`` hangs the SDK end path.

    Proves the harness reaches the SDK's listener wait, so the other tests would catch
    such a deadlock in pipecat-anam.
    """
    _, sdk_session, sdk_client = await _connected_client()
    close_now = asyncio.Event()

    async def close_session() -> None:
        await close_now.wait()
        await sdk_session.close()

    # Created outside listener context: a task a listener starts inherits the SDK's
    # notification context, and its close() would skip the wait this test needs.
    worker = asyncio.create_task(close_session())

    async def wait_for_closer(code: str, reason: str | None) -> None:
        close_now.set()
        await worker

    sdk_client.add_listener(AnamEvent.CONNECTION_CLOSED, wait_for_closer)
    end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
    try:
        # shield: a plain wait_for would cancel end_task and, through it, the worker.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(end_task), 1)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await asyncio.wait_for(end_task, 5)


async def _until_closed_locally(sdk_client: AnamClient) -> None:
    """Wait until the SDK has latched a local close (NORMAL) as the ending cause."""
    streaming_client = sdk_client._streaming_client
    assert streaming_client is not None

    async def poll() -> None:
        while streaming_client._end is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 5)
    assert streaming_client._end == (ConnectionClosedCode.NORMAL, None)


# --- AnamTransportClient --------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_sdk_signalling_failure_reports_one_error() -> None:
    client, _, sdk_client = await _connected_client()

    end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "retries used up")
    await asyncio.wait_for(end_task, 5)

    client._on_error.assert_awaited_once_with(
        "Anam session closed unexpectedly: signalling_failure (retries used up)"
    )
    assert client.session is None


@pytest.mark.asyncio
async def test_real_sdk_normal_remote_end_is_not_an_error() -> None:
    client, _, sdk_client = await _connected_client()

    await asyncio.wait_for(end_remotely(sdk_client, ConnectionClosedCode.NORMAL, None), 5)

    client._on_error.assert_not_awaited()
    assert client.session is None


@pytest.mark.asyncio
async def test_real_sdk_stop_completes() -> None:
    client, sdk_session, _ = await _connected_client()

    await asyncio.wait_for(client.stop(), 5)

    client._on_error.assert_not_awaited()
    assert sdk_session.is_active is False
    assert client.stop_called is True


@pytest.mark.parametrize("order", ["remote_first", "local_first"])
@pytest.mark.asyncio
async def test_real_sdk_stop_races_remote_end(order: str) -> None:
    """The SDK keeps the first cause, so it alone decides whether an error is reported."""
    client, _, sdk_client = await _connected_client()

    if order == "remote_first":
        end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
        stop_task = asyncio.create_task(client.stop())
    else:
        stop_task = asyncio.create_task(client.stop())
        await _until_closed_locally(sdk_client)
        end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
    await asyncio.wait_for(asyncio.gather(stop_task, end_task), 5)

    if order == "remote_first":
        client._on_error.assert_awaited_once_with(
            "Anam session closed unexpectedly: signalling_failure (x)"
        )
    else:
        client._on_error.assert_not_awaited()
    assert client.session is None


@pytest.mark.asyncio
async def test_real_sdk_end_during_connect_reports_start_failure() -> None:
    """The SDK sends no CONNECTION_CLOSED for a session that never started."""
    client = make_client()
    client._daily_client = MagicMock(leave=AsyncMock())
    client._daily_start_join = AsyncMock()

    with patch_sdk_network(end_during_connect=True):
        await asyncio.wait_for(client.start(StartFrame()), 5)

    client._on_error.assert_awaited_once()
    [message] = client._on_error.await_args.args
    assert (
        message == "AnamTransport failed to start: [session_error] Session ended while connecting"
    )
    assert client.session is None


@pytest.mark.asyncio
async def test_real_sdk_transport_fatal_cascade() -> None:
    """``on_error`` → ``_on_fatal_error`` → ``client.stop()`` runs inside the SDK listener."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    client, _, sdk_client = await _connected_client(transport._client)
    daily = MagicMock(leave=AsyncMock())
    client._daily_client = daily

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CANCEL)
    await harness.start()

    end_task = end_remotely(sdk_client, ConnectionClosedCode.SERVER_CLOSED, "boom")
    await asyncio.wait_for(end_task, 5)
    await harness.wait_finished()

    assert client.stop_called is True
    assert client.session is None
    daily.leave.assert_awaited_once()
    [error] = harness.errors
    assert "Anam session closed unexpectedly: server_closed" in error.error


@pytest.mark.asyncio
async def test_real_sdk_agent_audio_after_end_is_dropped() -> None:
    """A late TTS chunk after the session ended does not raise into the pipeline."""
    client, _, sdk_client = await _connected_client()
    client._audio_out_sample_rate = 24000
    client._create_agent_audio_stream()
    stream = client.agent_audio_stream
    assert stream is not None

    end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
    await asyncio.wait_for(end_task, 5)

    await asyncio.wait_for(stream.send_audio_chunk(b"\x00\x00" * 480), 5)
    await asyncio.wait_for(stream.end_sequence(), 5)


# --- AnamVideoService -----------------------------------------------------------------


@dataclass
class _Consumer:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    def frames(self) -> Callable[[Session], AsyncIterator[Any]]:
        def factory(_session: Session) -> AsyncIterator[Any]:
            self.started.set()
            return _blocking_frames(self.cancelled)

        return factory


@dataclass
class _RealSdkService:
    service: AnamVideoService
    harness: _Harness
    sdk_client: AnamClient
    sdk_session: Session
    video: _Consumer
    audio: _Consumer
    tasks: list[asyncio.Task] = field(default_factory=list)

    def listeners_attached(self) -> bool:
        callbacks = self.sdk_client._event_callbacks
        return (
            self.service._on_session_ready in callbacks[AnamEvent.SESSION_READY]
            or self.service._on_connection_closed in callbacks[AnamEvent.CONNECTION_CLOSED]
        )


@contextlib.asynccontextmanager
async def _real_sdk_service(**network: bool) -> AsyncIterator[_RealSdkService]:
    """Start an AnamVideoService through its own connect path on the real SDK.

    The pipeline runs under CONTINUE, so whatever the test observes was done by the
    service and not by the worker tearing it down. Media iterators block forever.
    """
    video, audio = _Consumer(), _Consumer()
    service = _make_video_service()
    harness = _Harness(service, ProcessorUnusablePolicy.CONTINUE)
    # A test that calls service.cleanup() directly also stops the service's frame tasks,
    # so the worker's CancelFrame never reaches the end of the pipeline at teardown.
    harness.worker._cancel_timeout_secs = 0.5
    with (
        patch_sdk_network(**network),
        patch.object(Session, "video_frames", video.frames()),
        patch.object(Session, "audio_frames", audio.frames()),
    ):
        await harness.start()
        await asyncio.wait_for(video.started.wait(), 5)
        await asyncio.wait_for(audio.started.wait(), 5)
        sdk_client = service._client
        sdk_session = service._anam_session
        assert sdk_client is not None
        assert isinstance(sdk_session, Session)
        real = _RealSdkService(service, harness, sdk_client, sdk_session, video, audio)
        assert service._video_task is not None and service._audio_task is not None
        real.tasks = [service._video_task, service._audio_task]
        try:
            yield real
        finally:
            if harness.is_running():
                await harness.cancel()


async def _ready(real: _RealSdkService) -> None:
    await asyncio.wait_for(real.sdk_client._handle_session_ready(), 5)
    assert real.service._session_ready_event.is_set()


async def _settle() -> None:
    await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_real_sdk_video_signalling_failure_reports_one_error() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)

        end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
        await asyncio.wait_for(end_task, 5)
        await real.harness.wait_errored()
        await _settle()

        [error] = real.harness.errors
        assert "Anam connection closed: signalling_failure" in error.error
        assert not real.listeners_attached()
        assert all(task.done() for task in real.tasks)
        assert real.video.cancelled.is_set() and real.audio.cancelled.is_set()


@pytest.mark.asyncio
async def test_real_sdk_video_normal_remote_end_is_not_an_error() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)

        await asyncio.wait_for(end_remotely(real.sdk_client, ConnectionClosedCode.NORMAL, None), 5)
        await _settle()

        assert real.harness.errors == []


@pytest.mark.asyncio
async def test_real_sdk_video_cleanup_completes() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)

        await asyncio.wait_for(real.service.cleanup(), 5)
        await _settle()

        assert real.harness.errors == []
        assert real.sdk_session.is_active is False


@pytest.mark.parametrize("order", ["remote_first", "local_first"])
@pytest.mark.asyncio
async def test_real_sdk_video_cleanup_races_remote_end(order: str) -> None:
    async with _real_sdk_service() as real:
        await _ready(real)

        if order == "remote_first":
            end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
            cleanup_task = asyncio.create_task(real.service.cleanup())
        else:
            cleanup_task = asyncio.create_task(real.service.cleanup())
            await _until_closed_locally(real.sdk_client)
            end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
        await asyncio.wait_for(asyncio.gather(cleanup_task, end_task), 5)
        await _settle()

        if order == "remote_first":
            [error] = real.harness.errors
            assert "Anam connection closed: signalling_failure" in error.error
        else:
            assert real.harness.errors == []


@pytest.mark.asyncio
async def test_real_sdk_video_interruption_after_abnormal_end() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)
        end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
        await asyncio.wait_for(end_task, 5)

        await asyncio.wait_for(real.service._handle_interruption(), 5)


@pytest.mark.asyncio
async def test_real_sdk_video_early_session_ready_is_not_lost() -> None:
    """SESSION_READY can arrive while connect_async() is still running; it fires only once."""
    async with _real_sdk_service(ready_during_connect=True) as real:
        service = real.service
        assert service._session_ready_event.is_set()
        assert service._send_task is not None and not service._send_task.done()
        assert real.harness.errors == []

        stream = service._agent_audio_stream
        assert stream is not None
        sent = asyncio.Event()
        stream.send_audio_chunk = AsyncMock(side_effect=lambda _audio: sent.set())
        await real.harness.worker.queue_frames(
            [
                TTSStartedFrame(context_id="c1"),
                TTSAudioRawFrame(
                    audio=b"\0\0" * 160, sample_rate=24000, num_channels=1, context_id="c1"
                ),
            ]
        )
        await asyncio.wait_for(sent.wait(), 5)
        stream.send_audio_chunk.assert_awaited_once_with(b"\0\0" * 160)


@pytest.mark.asyncio
async def test_real_sdk_video_interruption_after_normal_end() -> None:
    """After a normal remote end the service keeps the ended session; interrupting must not
    wait for its data channel (the SDK would wait 10 s and raise)."""
    async with _real_sdk_service() as real:
        await _ready(real)
        await asyncio.wait_for(end_remotely(real.sdk_client, ConnectionClosedCode.NORMAL, None), 5)

        await asyncio.wait_for(real.service._handle_interruption(), 1)

        assert real.harness.errors == []
