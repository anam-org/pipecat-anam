#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how :class:`AnamTransport` and :class:`AnamVideoService` report
unrecoverable errors.

Pipecat 1.8.0 deprecated fatal errors (``push_error(fatal=True)`` and
``ErrorFrame(fatal=True)``) and removes them in 2.0.0. Both now mark the failing
processor unusable instead, and the ``PipelineWorker``'s ``processor_unusable_policy``
decides whether the pipeline is cancelled, ended, or kept running.

Tests that depend on the worker's reaction run a real ``PipelineWorker`` under a
``WorkerRunner``. ``on_pipeline_error`` fires for every ``ErrorFrame``, so
``len(harness.errors)`` is how "reported once" is checked. Some tests record every
warning, so a regression to the deprecated API fails the test. No live Daily/Anam
connection is used.

Event handlers registered on a Pipecat object run in their own task, so tests that
assert on an ``on_error`` handler let the loop turn before checking it.
"""

import asyncio
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from anam import AnamClient, AnamEvent, ConnectionClosedCode
from conftest import (
    _avatar,
    _blocking_frames,
    _connected_service,
    _deprecations,
    _FakeSession,
    _fast_ready_timeout,
    _Harness,
    _human,
    _make_transport,
    _make_video_service,
    _raising_frames,
    make_client,
)
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame
from pipecat.pipeline.worker import ProcessorUnusablePolicy
from pipecat.processors.filters.identity_filter import IdentityFilter

from pipecat_anam import AnamTransport, AnamVideoService


async def _settle() -> None:
    """Let event-handler tasks scheduled by ``_call_event_handler`` run."""
    await asyncio.sleep(0.05)


def _record_calls(transport: AnamTransport, calls: list[str]) -> AsyncMock:
    """Record the order of stop, on_error dispatch and push_error on ``transport``.

    ``on_error`` is recorded when it is dispatched, not when its handler task runs.
    Returns the ``push_error`` spy; the real ``push_error`` still runs, so the
    ErrorFrame still reaches the worker.
    """
    transport._client.stop = AsyncMock(side_effect=lambda: calls.append("stop"))

    real_call_event_handler = transport._call_event_handler

    async def call_event_handler(event_name, *args, **kwargs):
        if event_name == "on_error":
            calls.append("on_error")
        await real_call_event_handler(event_name, *args, **kwargs)

    transport._call_event_handler = call_event_handler

    real_push_error = transport._input.push_error

    async def push_error(*args, **kwargs):
        calls.append("push_error")
        await real_push_error(*args, **kwargs)

    spy = AsyncMock(side_effect=push_error)
    transport._input.push_error = spy
    return spy


def _probe_at_push_error(
    service: AnamVideoService, probe: Callable[[], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run ``probe`` each time ``service`` pushes an error, before the error is pushed.

    Under CANCEL the pipeline's own teardown cancels every task afterwards, so cleanup
    done by ``_report_unrecoverable_error`` itself is only observable at this point.
    """
    snapshots: list[dict[str, Any]] = []
    real_push_error = service.push_error

    async def push_error(*args, **kwargs):
        snapshots.append(probe())
        await real_push_error(*args, **kwargs)

    service.push_error = push_error
    return snapshots


# --- AnamTransport._on_fatal_error ----------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_cancels_pipeline_with_cancel_policy() -> None:
    """An unrecoverable transport error marks the input unusable and CANCEL stops the pipeline."""
    transport = _make_transport()
    # Stand in for the Daily-backed input transport: only push_error is exercised.
    transport._input = IdentityFilter(name="anam-input")
    on_error = AsyncMock()
    transport.add_event_handler("on_error", on_error)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        harness = _Harness(transport._input, ProcessorUnusablePolicy.CANCEL)
        await harness.start()

        await transport._on_fatal_error("Anam avatar participant left Daily room unexpectedly")
        await harness.wait_finished()

    assert _deprecations(caught) == []
    assert harness.worker.has_finished()
    assert transport._input.is_usable is False
    on_error.assert_awaited_once_with(
        transport, "Anam avatar participant left Daily room unexpectedly"
    )
    [error] = harness.errors
    assert error.processor is transport._input
    assert error.fatal is False
    assert "left Daily room unexpectedly" in error.error


@pytest.mark.asyncio
async def test_transport_error_keeps_pipeline_running_with_default_policy() -> None:
    """Pipecat's default CONTINUE policy reports the error but leaves the pipeline running."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        harness = _Harness(transport._input, ProcessorUnusablePolicy.CONTINUE)
        await harness.start()

        await transport._on_fatal_error("Daily room disconnected unexpectedly")
        await harness.wait_errored()
        await asyncio.sleep(0.2)
        still_running = harness.is_running()

        await harness.cancel()

    assert _deprecations(caught) == []
    assert still_running
    assert transport._input.is_usable is False
    [error] = harness.errors
    assert error.fatal is False


@pytest.mark.asyncio
async def test_transport_fatal_error_stops_client_before_reporting() -> None:
    """The session is released and Daily is left before on_error and the ErrorFrame."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    calls: list[str] = []
    push_error = _record_calls(transport, calls)

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CONTINUE)
    await harness.start()

    await transport._on_fatal_error("boom")
    await harness.wait_errored()
    await harness.cancel()

    assert calls == ["stop", "on_error", "push_error"]
    push_error.assert_awaited_once_with(error_msg="boom", force_treat_as_permanent=True)


@pytest.mark.asyncio
async def test_transport_fatal_error_reported_once() -> None:
    """A second fatal error after the first is a no-op."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    transport._client.stop = AsyncMock()
    on_error = AsyncMock()
    transport.add_event_handler("on_error", on_error)

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CONTINUE)
    await harness.start()

    await transport._on_fatal_error("first")
    await transport._on_fatal_error("second")
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    await harness.cancel()

    assert transport._client.stop.await_count == 1
    on_error.assert_awaited_once_with(transport, "first")
    assert len(harness.errors) == 1
    assert transport._unrecoverable_error_reported is True


@pytest.mark.asyncio
async def test_transport_fatal_error_concurrent_sources_report_once() -> None:
    """Daily leave, avatar leave and Anam close arriving together give one report."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")

    async def yielding_stop() -> None:
        # Yield so the three reports interleave.
        await asyncio.sleep(0)

    transport._client.stop = AsyncMock(side_effect=yielding_stop)
    on_error = AsyncMock()
    transport.add_event_handler("on_error", on_error)

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CONTINUE)
    await harness.start()

    await asyncio.gather(
        transport._on_fatal_error("a"),
        transport._on_fatal_error("b"),
        transport._on_fatal_error("c"),
    )
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    await harness.cancel()

    assert transport._client.stop.await_count == 1
    assert len(harness.errors) == 1
    on_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_transport_fatal_error_cascade_through_real_client() -> None:
    """An Anam close cascades through the real client callbacks into exactly one error.

    The client's ``on_error`` is wired to ``transport._on_fatal_error`` by the
    constructor. Leaving Daily makes Daily fire ``on_left``, and the avatar then leaves
    the room; both happen after ``stop()`` and must not report again.
    """
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    client = transport._client
    session = _FakeSession()
    client._session = session

    async def leave() -> None:
        await client._on_left()

    daily = MagicMock(leave=AsyncMock(side_effect=leave))
    client._daily_client = daily

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CANCEL)
    await harness.start()

    await client._on_connection_closed("server_closed", "boom")
    await client._on_left()
    await transport._on_participant_disconnected(_avatar(), "left")
    await harness.wait_finished()

    assert harness.worker.has_finished()
    [error] = harness.errors
    assert "Anam session closed unexpectedly: server_closed" in error.error
    daily.leave.assert_awaited_once()
    # The SDK closes the session before emitting CONNECTION_CLOSED, and the client
    # drops its reference before reporting, so stop() has no session to close.
    session.close.assert_not_awaited()
    assert client.session is None
    assert client.stop_called is True


@pytest.mark.asyncio
async def test_transport_fatal_error_falls_back_to_output() -> None:
    """Without an input transport the error is pushed from the output transport."""
    transport = _make_transport()
    transport._output = IdentityFilter(name="anam-output")
    transport._client.stop = AsyncMock()

    harness = _Harness(transport._output, ProcessorUnusablePolicy.CANCEL)
    await harness.start()

    await transport._on_fatal_error("boom")
    await harness.wait_finished()

    [error] = harness.errors
    assert error.processor is transport._output
    assert transport._output.is_usable is False
    transport._client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_transport_fatal_error_without_processors_still_stops_client() -> None:
    """With neither input nor output, the session is still released and nothing raises."""
    transport = _make_transport()
    transport._client.stop = AsyncMock()
    on_error = AsyncMock()
    transport.add_event_handler("on_error", on_error)

    await transport._on_fatal_error("boom")
    await _settle()

    transport._client.stop.assert_awaited_once()
    on_error.assert_awaited_once_with(transport, "boom")


@pytest.mark.asyncio
async def test_transport_fatal_error_handler_exception_still_pushes_error() -> None:
    """A raising on_error handler does not block the ErrorFrame."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    transport._client.stop = AsyncMock()
    on_error = AsyncMock(side_effect=RuntimeError("handler"))
    transport.add_event_handler("on_error", on_error)

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CANCEL)
    await harness.start()

    await transport._on_fatal_error("boom")
    await harness.wait_finished()

    on_error.assert_awaited_once()
    assert len(harness.errors) == 1
    assert transport._input.is_usable is False


@pytest.mark.asyncio
async def test_transport_fatal_error_when_stop_raises() -> None:
    """Documents current behaviour when ``_client.stop()`` itself raises.

    The exception propagates, nothing is reported, and the reported flag stays set, so
    the error is lost for good. The real ``stop()`` swallows close and leave errors, so
    this only matters if that changes.
    """
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    transport._input.push_error = AsyncMock()
    transport._client.stop = AsyncMock(side_effect=RuntimeError("stop"))
    on_error = AsyncMock()
    transport.add_event_handler("on_error", on_error)

    with pytest.raises(RuntimeError, match="stop"):
        await transport._on_fatal_error("boom")
    await _settle()

    transport._input.push_error.assert_not_awaited()
    on_error.assert_not_awaited()
    assert transport._unrecoverable_error_reported is True


# --- Avatar participant left ----------------------------------------------------------


@pytest.mark.asyncio
async def test_avatar_left_unexpectedly_reports_fatal_error() -> None:
    """The avatar leaving while the transport is running is unrecoverable."""
    transport = _make_transport()
    transport._input = IdentityFilter(name="anam-input")
    transport._client.stop = AsyncMock()
    on_error = AsyncMock()
    on_avatar_disconnected = AsyncMock()
    transport.add_event_handler("on_error", on_error)
    transport.add_event_handler("on_avatar_disconnected", on_avatar_disconnected)
    participant = _avatar()

    harness = _Harness(transport._input, ProcessorUnusablePolicy.CANCEL)
    await harness.start()

    await transport._on_participant_disconnected(participant, "networkTimeout")
    await harness.wait_finished()

    on_avatar_disconnected.assert_awaited_once_with(transport, participant, "networkTimeout")
    on_error.assert_awaited_once_with(
        transport, "Anam avatar participant left Daily room unexpectedly: networkTimeout"
    )
    transport._client.stop.assert_awaited_once()
    [error] = harness.errors
    assert error.processor is transport._input
    assert transport._input.is_usable is False


@pytest.mark.asyncio
async def test_avatar_left_after_stop_is_not_an_error() -> None:
    """The avatar leaving during a user-initiated shutdown is not reported."""
    transport = _make_transport()
    await transport._client.stop()
    on_error = AsyncMock()
    on_avatar_disconnected = AsyncMock()
    transport.add_event_handler("on_error", on_error)
    transport.add_event_handler("on_avatar_disconnected", on_avatar_disconnected)
    participant = _avatar()

    with patch.object(transport, "_on_fatal_error", AsyncMock()) as on_fatal_error:
        await transport._on_participant_disconnected(participant, "leftCall")
        await _settle()

    on_avatar_disconnected.assert_awaited_once_with(transport, participant, "leftCall")
    on_fatal_error.assert_not_awaited()
    on_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_participant_left_is_not_an_error() -> None:
    """A non-avatar participant leaving only fires on_client_disconnected."""
    transport = _make_transport()
    transport._client.stop = AsyncMock()
    on_client_disconnected = AsyncMock()
    on_avatar_disconnected = AsyncMock()
    transport.add_event_handler("on_client_disconnected", on_client_disconnected)
    transport.add_event_handler("on_avatar_disconnected", on_avatar_disconnected)
    participant = _human()

    with patch.object(transport, "_on_fatal_error", AsyncMock()) as on_fatal_error:
        await transport._on_participant_disconnected(participant, "leftCall")
        await _settle()

    on_client_disconnected.assert_awaited_once_with(transport, participant)
    on_avatar_disconnected.assert_not_awaited()
    on_fatal_error.assert_not_awaited()
    transport._client.stop.assert_not_awaited()


# --- AnamTransportClient callbacks that route to on_error ----------------------------


@pytest.mark.asyncio
async def test_client_normal_connection_close_is_not_an_error() -> None:
    """A graceful Anam close drops the session without reporting an error."""
    client = make_client()
    client._session = _FakeSession()

    await client._on_connection_closed("normal", None)

    client._on_error.assert_not_awaited()
    assert client.session is None


@pytest.mark.parametrize(
    "code", ["server_closed", "webrtc_failure", "signalling_failure", "timeout", "error"]
)
@pytest.mark.asyncio
async def test_client_abnormal_connection_close_reports_error(code: str) -> None:
    """Every non-normal Anam close code is reported through on_error."""
    assert ConnectionClosedCode(code) is not ConnectionClosedCode.NORMAL
    client = make_client()
    client._session = _FakeSession()

    await client._on_connection_closed(code, "r")

    client._on_error.assert_awaited_once_with(f"Anam session closed unexpectedly: {code} (r)")
    assert client.session is None


@pytest.mark.asyncio
async def test_client_daily_left_after_stop_is_not_an_error() -> None:
    """Leaving Daily as part of stop() is not reported."""
    client = make_client()

    await client.stop()
    await client._on_left()

    client._on_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_daily_left_without_stop_reports_error() -> None:
    """Leaving Daily without stop() (room ended, kicked, ...) is reported."""
    client = make_client()

    await client._on_left()

    client._on_error.assert_awaited_once()
    [message] = client._on_error.await_args.args
    assert "Daily room disconnected unexpectedly" in message


# --- AnamVideoService._report_unrecoverable_error -----------------------------------


@pytest.mark.asyncio
async def test_video_service_connect_failure_detaches_listeners_and_cancels_send_task() -> None:
    """A failed Anam connect cleans up, marks the service unusable and CANCEL stops the pipeline."""
    service = _make_video_service()
    connect_tasks: list[asyncio.Task] = []

    async def connect_async(*args, **kwargs):
        connect_tasks.append(asyncio.current_task())
        raise ConnectionError("connection refused")

    with (
        patch.object(AnamClient, "connect_async", connect_async),
        patch.object(AnamClient, "remove_listener") as remove_listener,
        warnings.catch_warnings(record=True) as caught,
    ):
        snapshots = _probe_at_push_error(
            service,
            lambda: {
                "removed": list(remove_listener.call_args_list),
                "send_task": service._send_task,
                "agent_audio_stream": service._agent_audio_stream,
            },
        )
        warnings.simplefilter("always")
        harness = _Harness(service, ProcessorUnusablePolicy.CANCEL)
        await harness.start()
        await harness.wait_finished()

    assert _deprecations(caught) == []
    assert harness.worker.has_finished()
    assert service.is_usable is False
    [snapshot] = snapshots
    assert call(AnamEvent.SESSION_READY, service._on_session_ready) in snapshot["removed"]
    assert call(AnamEvent.CONNECTION_CLOSED, service._on_connection_closed) in snapshot["removed"]
    assert snapshot["send_task"] is None
    assert snapshot["agent_audio_stream"] is None
    # The connect task reported the error itself, so it finishes normally.
    [connect_task] = connect_tasks
    assert connect_task.done()
    assert not connect_task.cancelled()
    [error] = harness.errors
    assert error.processor is service
    assert error.fatal is False
    assert isinstance(error.exception, ConnectionError)
    assert "Error connecting to Anam" in error.error


@pytest.mark.asyncio
async def test_video_service_connect_failure_keeps_pipeline_running_with_default_policy() -> None:
    """Under CONTINUE a failed connect is reported once and the pipeline keeps running."""
    service = _make_video_service()

    with patch.object(
        AnamClient,
        "connect_async",
        AsyncMock(side_effect=ConnectionError("connection refused")),
    ):
        harness = _Harness(service, ProcessorUnusablePolicy.CONTINUE)
        await harness.start()
        await harness.wait_errored()
        await asyncio.sleep(0.2)
        still_running = harness.is_running()
        await harness.cancel()

    assert still_running
    assert service.is_usable is False
    [error] = harness.errors
    assert error.fatal is False


@pytest.mark.asyncio
async def test_video_service_session_ready_timeout_cancels_consumers() -> None:
    """The session-ready watchdog firing cancels the already-running consumer tasks."""
    video_cancelled = asyncio.Event()
    audio_cancelled = asyncio.Event()
    session = _FakeSession(
        video_frames=lambda: _blocking_frames(video_cancelled),
        audio_frames=lambda: _blocking_frames(audio_cancelled),
    )
    service = _make_video_service()
    snapshots = _probe_at_push_error(
        service,
        lambda: {
            "video_cancelled": video_cancelled.is_set(),
            "audio_cancelled": audio_cancelled.is_set(),
            "video_task": service._video_task,
            "audio_task": service._audio_task,
            "close_count": session.close.await_count,
        },
    )

    with _fast_ready_timeout():
        _, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session, service)
        await harness.wait_finished()

    [error] = harness.errors
    assert error.error == "Anam session connection timed out."
    assert isinstance(error.exception, TimeoutError)
    assert snapshots == [
        {
            "video_cancelled": True,
            "audio_cancelled": True,
            "video_task": None,
            "audio_task": None,
            "close_count": 1,
        }
    ]
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_service_video_consumer_error_is_permanent() -> None:
    """A video consumer failure marks the service unusable and cancels the other tasks."""
    audio_cancelled = asyncio.Event()
    session = _FakeSession(
        video_frames=lambda: _raising_frames(RuntimeError("decode")),
        audio_frames=lambda: _blocking_frames(audio_cancelled),
    )
    service = _make_video_service()
    snapshots = _probe_at_push_error(
        service,
        lambda: {
            "audio_cancelled": audio_cancelled.is_set(),
            "connect_task": service._connect_task,
            "connect_task_cancelled": session.connect_task.cancelled(),
            "close_count": session.close.await_count,
        },
    )

    _, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session, service)
    await harness.wait_finished()

    [error] = harness.errors
    assert "Anam error consuming video frames: decode" in error.error
    assert isinstance(error.exception, RuntimeError)
    assert error.fatal is False
    assert service.is_usable is False
    assert snapshots == [
        {
            "audio_cancelled": True,
            "connect_task": None,
            "connect_task_cancelled": True,
            "close_count": 1,
        }
    ]
    # The reporting task is left to finish on its own instead of cancelling itself.
    assert session.video_task.done()
    assert not session.video_task.cancelled()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_service_video_consumer_error_keeps_pipeline_running_with_default_policy() -> (
    None
):
    """Under CONTINUE a video consumer failure is reported once and the pipeline keeps running."""
    session = _FakeSession(video_frames=lambda: _raising_frames(RuntimeError("decode")))

    service, harness = await _connected_service(ProcessorUnusablePolicy.CONTINUE, session)
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    still_running = harness.is_running()
    await harness.cancel()

    assert still_running
    assert service.is_usable is False
    assert len(harness.errors) == 1


@pytest.mark.asyncio
async def test_video_service_audio_consumer_error_is_permanent() -> None:
    """An audio consumer failure marks the service unusable and cancels the other tasks."""
    video_cancelled = asyncio.Event()
    session = _FakeSession(
        video_frames=lambda: _blocking_frames(video_cancelled),
        audio_frames=lambda: _raising_frames(RuntimeError("decode")),
    )
    service = _make_video_service()
    snapshots = _probe_at_push_error(
        service,
        lambda: {
            "video_cancelled": video_cancelled.is_set(),
            "connect_task": service._connect_task,
            "connect_task_cancelled": session.connect_task.cancelled(),
            "close_count": session.close.await_count,
        },
    )

    _, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session, service)
    await harness.wait_finished()

    [error] = harness.errors
    assert "Anam error consuming audio frames" in error.error
    assert isinstance(error.exception, RuntimeError)
    assert service.is_usable is False
    assert snapshots == [
        {
            "video_cancelled": True,
            "connect_task": None,
            "connect_task_cancelled": True,
            "close_count": 1,
        }
    ]
    assert session.audio_task.done()
    assert not session.audio_task.cancelled()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_service_abnormal_connection_closed_is_permanent() -> None:
    """An abnormal close reported from outside the service's tasks cancels all of them."""
    video_cancelled = asyncio.Event()
    audio_cancelled = asyncio.Event()
    session = _FakeSession(
        video_frames=lambda: _blocking_frames(video_cancelled),
        audio_frames=lambda: _blocking_frames(audio_cancelled),
    )
    service = _make_video_service()

    _, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session, service)
    with patch.object(AnamClient, "remove_listener") as remove_listener:
        snapshots = _probe_at_push_error(
            service,
            lambda: {
                "video_cancelled": video_cancelled.is_set(),
                "audio_cancelled": audio_cancelled.is_set(),
                "connect_task_cancelled": session.connect_task.cancelled(),
                "tasks": (service._connect_task, service._video_task, service._audio_task),
                "removed": list(remove_listener.call_args_list),
            },
        )
        await service._on_connection_closed("webrtc_failure", "ice failed")
    await harness.wait_finished()

    [error] = harness.errors
    assert error.error == "Anam connection closed: webrtc_failure - ice failed"
    assert error.exception is None
    assert service.is_usable is False
    [snapshot] = snapshots
    assert snapshot["video_cancelled"] is True
    assert snapshot["audio_cancelled"] is True
    assert snapshot["connect_task_cancelled"] is True
    assert snapshot["tasks"] == (None, None, None)
    assert call(AnamEvent.SESSION_READY, service._on_session_ready) in snapshot["removed"]
    assert call(AnamEvent.CONNECTION_CLOSED, service._on_connection_closed) in snapshot["removed"]
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_video_service_abnormal_close_without_reason() -> None:
    """Without a reason the message has no ``" - "`` suffix."""
    session = _FakeSession()

    service, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session)
    await service._on_connection_closed("server_closed", None)
    await harness.wait_finished()

    [error] = harness.errors
    assert error.error == "Anam connection closed: server_closed"


@pytest.mark.asyncio
async def test_video_service_normal_connection_closed_is_not_an_error() -> None:
    """A graceful close reports nothing; the pipeline survives even under CANCEL."""
    session = _FakeSession()

    service, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session)
    await service._on_connection_closed("normal", None)
    await asyncio.sleep(0.2)
    still_running = harness.is_running()
    await harness.cancel()

    assert still_running
    assert harness.errors == []
    assert service.is_usable is True
    assert service._unrecoverable_error_reported is False


@pytest.mark.asyncio
async def test_video_service_unrecoverable_error_reported_once() -> None:
    """Several failure sources give one report and one session close."""
    session = _FakeSession()

    service, harness = await _connected_service(ProcessorUnusablePolicy.CONTINUE, session)
    await service._on_connection_closed("error", "x")
    await service._on_connection_closed("timeout", "y")
    await service._report_unrecoverable_error("z")
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    await harness.cancel()

    [error] = harness.errors
    assert "error - x" in error.error
    assert session.close.await_count == 1


@pytest.mark.asyncio
async def test_video_service_concurrent_consumer_failures_report_once() -> None:
    """Both consumers failing in the same loop tick give one report."""
    session = _FakeSession(
        video_frames=lambda: _raising_frames(RuntimeError("video"), yield_first=True),
        audio_frames=lambda: _raising_frames(RuntimeError("audio"), yield_first=True),
    )

    service, harness = await _connected_service(ProcessorUnusablePolicy.CONTINUE, session)
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    await harness.cancel()

    [error] = harness.errors
    assert error.error in (
        "Anam error consuming video frames: video",
        "Anam error consuming audio frames: audio",
    )
    assert service.is_usable is False
    assert session.close.await_count <= 1


@pytest.mark.asyncio
async def test_video_service_session_close_failure_still_reports_error() -> None:
    """A failing session close is logged and the permanent error is still pushed."""
    session = _FakeSession()
    session.close = AsyncMock(side_effect=RuntimeError("close"))

    service, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session)
    await service._on_connection_closed("error", "x")
    await harness.wait_finished()

    session.close.assert_awaited_once()
    assert len(harness.errors) == 1
    assert service.is_usable is False


# --- TTS send error stays non-permanent -----------------------------------------------


@pytest.mark.asyncio
async def test_video_service_tts_send_error_is_not_permanent() -> None:
    """A failed audio chunk send is reported but keeps the service usable."""
    session = _FakeSession()
    session.audio_stream.send_audio_chunk = AsyncMock(side_effect=RuntimeError("ws"))

    service, harness = await _connected_service(ProcessorUnusablePolicy.CANCEL, session)
    await service._on_session_ready()
    await harness.worker.queue_frames(
        [
            TTSStartedFrame(context_id="c1"),
            TTSAudioRawFrame(
                audio=b"\0\0" * 160, sample_rate=24000, num_channels=1, context_id="c1"
            ),
        ]
    )
    await harness.wait_errored()
    await asyncio.sleep(0.2)
    still_running = harness.is_running()
    # Checked before cancelling: the pipeline's own shutdown closes the session.
    session.close.assert_not_awaited()
    await harness.cancel()

    [error] = harness.errors
    assert "Anam audio send error: ws" in error.error
    assert still_running
    assert service.is_usable is True
    assert service._unrecoverable_error_reported is False
