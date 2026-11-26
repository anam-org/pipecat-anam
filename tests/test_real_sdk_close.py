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
from unittest.mock import AsyncMock, MagicMock

import pytest
from anam import AnamClient, AnamEvent, Session
from anam.types import ConnectionClosedCode
from conftest import _Harness, _make_transport, make_client
from pipecat.frames.frames import StartFrame
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.filters.identity_filter import IdentityFilter
from real_sdk import end_remotely, patch_sdk_network

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
