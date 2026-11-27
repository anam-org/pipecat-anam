#
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anam import AnamClient, AnamEvent, Session
from anam._signalling import SignallingClient
from anam._streaming import StreamingClient
from anam.types import ConnectionClosedCode
from conftest import (
    _blocking_frames,
    _Harness,
    _make_video_service,
    _wait_until,
    make_client,
)
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_transport import TransportParams
from real_sdk import (
    FakeSocket,
    end_remotely,
    fake_signalling_sockets,
    patch_sdk_network,
    real_sdk_connect,
)

from pipecat_anam import AnamVideoService
from pipecat_anam.transport import AnamOutputTransport, AnamTransportClient


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


async def _until_closed_locally(sdk_client: AnamClient) -> None:
    streaming_client = sdk_client._streaming_client
    assert streaming_client is not None

    async def poll() -> None:
        while streaming_client._end is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 5)
    assert streaming_client._end == (ConnectionClosedCode.NORMAL, None)


@pytest.mark.asyncio
async def test_real_sdk_signalling_failure_reports_one_error() -> None:
    client, _, sdk_client = await _connected_client()

    end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "retries used up")
    await asyncio.wait_for(end_task, 5)

    client._on_error.assert_awaited_once_with(
        "Anam session closed unexpectedly: signalling_failure (retries used up)"
    )
    assert client.session is None


@pytest.mark.parametrize("order", ["remote_first", "local_first"])
@pytest.mark.asyncio
async def test_real_sdk_stop_races_remote_end(order: str) -> None:
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


def _tts_frames(context_id: str = "c1") -> list[Frame]:
    return [
        TTSStartedFrame(context_id=context_id),
        TTSAudioRawFrame(
            audio=b"\0\0" * 160, sample_rate=24000, num_channels=1, context_id=context_id
        ),
    ]


@contextlib.contextmanager
def _error_logs() -> Iterator[list[str]]:
    errors: list[str] = []
    sink = logger.add(lambda message: errors.append(str(message)), level="ERROR")
    try:
        yield errors
    finally:
        logger.remove(sink)


def _streaming_of(session: Session) -> StreamingClient:
    streaming_client = session._client._streaming_client
    assert streaming_client is not None
    return streaming_client


def _signalling_of(session: Session) -> SignallingClient:
    signalling = _streaming_of(session)._signalling_client
    assert signalling is not None
    return signalling


async def _drop_and_reconnect(sockets: list[FakeSocket], session: Session) -> FakeSocket:
    signalling = _signalling_of(session)
    [first] = sockets
    first.drop()
    await _wait_until(lambda: len(sockets) == 2 and signalling._ws is sockets[1], 5)
    return sockets[1]


def _output_harness(client: AnamTransportClient) -> tuple[_Harness, MagicMock]:
    daily = MagicMock(
        leave=AsyncMock(), cleanup=AsyncMock(), send_message=AsyncMock(return_value=None)
    )
    client._daily_client = daily
    harness = _Harness(
        AnamOutputTransport(client, TransportParams()),
        ProcessorUnusablePolicy.CONTINUE,
        cancel_timeout_secs=0.5,
    )
    return harness, daily


@pytest.mark.asyncio
async def test_real_sdk_transport_recovers_a_dropped_signalling_socket() -> None:
    client = make_client()
    harness, daily = _output_harness(client)
    with fake_signalling_sockets() as sockets:
        with patch_sdk_network(open_signalling=True):
            await asyncio.wait_for(client._anam_connect(), 5)
        session = client.session
        assert session is not None
        await harness.start()
        try:
            client._create_agent_audio_stream()

            new_socket = await _drop_and_reconnect(sockets, session)
            await harness.worker.queue_frames(_tts_frames())
            await _wait_until(lambda: bool(new_socket.sent_actions("agentaudioinput")), 5)

            assert sockets[0].sent_actions("agentaudioinput") == []
            assert session.is_active
            assert client.session is session
            client._on_error.assert_not_awaited()
            daily.leave.assert_not_awaited()
            assert harness.errors == []
        finally:
            if harness.is_running():
                await harness.cancel()


@pytest.mark.asyncio
async def test_real_sdk_transport_interruption_while_session_is_ending() -> None:
    client, sdk_session, sdk_client = await _connected_client()
    client._audio_out_sample_rate = 24000
    client._create_agent_audio_stream()
    output = AnamOutputTransport(client, TransportParams())
    signalling = _signalling_of(sdk_session)
    release_teardown = asyncio.Event()
    signalling.close = AsyncMock(side_effect=release_teardown.wait)

    end_task = end_remotely(sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
    try:
        await asyncio.wait_for(_until_awaited(signalling.close), 5)
        assert client.session is sdk_session and sdk_session.is_active

        with _error_logs() as errors:
            await asyncio.wait_for(output._on_interruption(), 1)

        assert errors == []
    finally:
        release_teardown.set()
        await asyncio.wait_for(end_task, 5)


@pytest.mark.asyncio
async def test_real_sdk_transport_interruption_during_local_stop_skips_interrupt() -> None:
    client, sdk_session, _ = await _connected_client()
    client._daily_client = MagicMock(leave=AsyncMock())
    output = AnamOutputTransport(client, TransportParams())
    signalling = _signalling_of(sdk_session)
    release_teardown = asyncio.Event()
    signalling.close = AsyncMock(side_effect=release_teardown.wait)

    stop_task = asyncio.create_task(client.stop())
    try:
        await asyncio.wait_for(_until_awaited(signalling.close), 5)
        assert client.session is sdk_session and not sdk_session.is_active

        with patch.object(Session, "interrupt", autospec=True) as interrupt:
            await asyncio.wait_for(output._on_interruption(), 1)

        interrupt.assert_not_called()
    finally:
        release_teardown.set()
        await asyncio.wait_for(stop_task, 5)


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
async def _real_sdk_service(
    network: contextlib.AbstractContextManager[Any] | None = None,
) -> AsyncIterator[_RealSdkService]:
    video, audio = _Consumer(), _Consumer()
    service = _make_video_service()
    harness = _Harness(service, ProcessorUnusablePolicy.CONTINUE, cancel_timeout_secs=0.5)
    with (
        network or patch_sdk_network(),
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
    assert real.service._send_ready_event.is_set()


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


@pytest.mark.parametrize("order", ["remote_first", "local_first"])
@pytest.mark.asyncio
async def test_real_sdk_video_cleanup_races_remote_end(order: str) -> None:
    async with _real_sdk_service() as real:
        await _ready(real)

        if order == "remote_first":
            shutdown_client = real.service._shutdown_client

            async def shutdown_after_the_error() -> None:
                await asyncio.wait_for(real.harness.errored.wait(), 2)
                await shutdown_client()

            with patch.object(real.service, "_shutdown_client", shutdown_after_the_error):
                end_task = end_remotely(
                    real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x"
                )
                cleanup_task = asyncio.create_task(real.service.cleanup())
                await asyncio.wait_for(asyncio.gather(cleanup_task, end_task), 5)
        else:
            cleanup_task = asyncio.create_task(real.service.cleanup())
            await _until_closed_locally(real.sdk_client)
            end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
            await asyncio.wait_for(asyncio.gather(cleanup_task, end_task), 5)
        await _settle()

        assert cleanup_task.done() and end_task.done()
        assert real.sdk_session.is_active is False
        if order == "remote_first":
            [error] = real.harness.errors
            assert "Anam connection closed: signalling_failure" in error.error
        else:
            assert real.harness.errors == []


@pytest.mark.asyncio
async def test_real_sdk_video_interruption_while_session_is_ending() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)
        signalling = _signalling_of(real.sdk_session)
        release_teardown = asyncio.Event()
        signalling.close = AsyncMock(side_effect=release_teardown.wait)

        end_task = end_remotely(real.sdk_client, ConnectionClosedCode.SIGNALLING_FAILURE, "x")
        try:
            await asyncio.wait_for(_until_awaited(signalling.close), 5)
            assert real.sdk_session.is_active

            with _error_logs() as errors:
                await asyncio.wait_for(real.service._handle_interruption(), 1)

            assert errors == []
            assert real.harness.errors == []
            assert real.service._send_task is not None and not real.service._send_task.done()
        finally:
            release_teardown.set()
            await asyncio.wait_for(end_task, 5)


@pytest.mark.asyncio
async def test_real_sdk_video_interrupt_timeout_on_a_live_session_is_an_error() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)
        _streaming_of(real.sdk_session).wait_for_data_channel = AsyncMock(return_value=False)

        with _error_logs() as errors:
            await asyncio.wait_for(real.service._handle_interruption(), 1)

        [error] = errors
        assert "Anam interrupt failed: [session_error] Data channel did not open in time" in error
        assert real.service._send_task is not None and not real.service._send_task.done()


@pytest.mark.asyncio
async def test_real_sdk_video_forwards_the_interruption_after_a_failed_interrupt() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)
        _streaming_of(real.sdk_session).wait_for_data_channel = AsyncMock(return_value=False)
        forwarded = asyncio.Event()
        push_frame = real.service.push_frame

        async def spy(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            await push_frame(frame, direction)
            if isinstance(frame, InterruptionFrame) and direction is FrameDirection.DOWNSTREAM:
                forwarded.set()

        real.service.push_frame = spy  # type: ignore[method-assign]
        with _error_logs() as errors:
            await real.harness.worker.queue_frames([InterruptionFrame()])
            await asyncio.wait_for(forwarded.wait(), 5)

        assert any("Data channel did not open in time" in error for error in errors)
        assert real.harness.errors == []


async def _until_awaited(mock: AsyncMock) -> None:
    while not mock.await_count:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_real_sdk_video_early_session_ready_is_not_lost() -> None:
    async with _real_sdk_service(real_sdk_connect()) as real:
        service = real.service
        assert service._send_ready_event.is_set()
        assert service._send_task is not None and not service._send_task.done()
        assert real.harness.errors == []

        stream = service._agent_audio_stream
        assert stream is not None
        sent = asyncio.Event()
        stream.send_audio_chunk = AsyncMock(side_effect=lambda _audio: sent.set())
        await real.harness.worker.queue_frames(_tts_frames())
        await asyncio.wait_for(sent.wait(), 5)
        stream.send_audio_chunk.assert_awaited_once_with(b"\0\0" * 160)


@pytest.mark.asyncio
async def test_real_sdk_video_interruption_after_normal_end() -> None:
    async with _real_sdk_service() as real:
        await _ready(real)
        await asyncio.wait_for(end_remotely(real.sdk_client, ConnectionClosedCode.NORMAL, None), 5)

        with patch.object(Session, "interrupt", autospec=True) as interrupt:
            await asyncio.wait_for(real.service._handle_interruption(), 1)

        interrupt.assert_not_called()
        assert real.service._anam_session is real.sdk_session
        assert real.harness.errors == []


@pytest.mark.asyncio
async def test_real_sdk_video_recovers_a_dropped_signalling_socket() -> None:
    with fake_signalling_sockets() as sockets:
        async with _real_sdk_service(patch_sdk_network(open_signalling=True)) as real:
            await _ready(real)

            new_socket = await _drop_and_reconnect(sockets, real.sdk_session)
            await real.harness.worker.queue_frames(_tts_frames())
            await _wait_until(lambda: bool(new_socket.sent_actions("agentaudioinput")), 5)

            assert sockets[0].sent_actions("agentaudioinput") == []
            assert real.sdk_session.is_active
            assert real.service._anam_session is real.sdk_session
            assert real.harness.errors == []
