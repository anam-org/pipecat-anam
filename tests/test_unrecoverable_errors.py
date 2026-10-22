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

Each test runs a real ``PipelineWorker`` under a ``WorkerRunner`` and records every
warning, so a regression to the deprecated API fails the test. No live Daily/Anam
connection is used.
"""

import asyncio
import warnings
from unittest.mock import AsyncMock, patch

import pytest
from anam import AnamClient, PersonaConfig
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_anam import AnamTransport, AnamVideoService

TIMEOUT_SECS = 10


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


def _deprecations(caught: list[warnings.WarningMessage]) -> list[str]:
    # The stdlib `audioop` warning comes from Pipecat's audio utils, not from this package.
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "audioop" not in str(w.message)
    ]


class _Harness:
    """Run a pipeline in a PipelineWorker and collect the errors it reports."""

    def __init__(self, processor: FrameProcessor, policy: ProcessorUnusablePolicy):
        self.worker = PipelineWorker(
            Pipeline([processor]),
            params=PipelineParams(),
            processor_unusable_policy=policy,
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self.errors: list[ErrorFrame] = []
        self.started = asyncio.Event()
        self.errored = asyncio.Event()
        self._run_task: asyncio.Task | None = None

        @self.worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame):
            self.started.set()

        @self.worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            self.errors.append(frame)
            self.errored.set()

    async def start(self) -> None:
        await self.runner.add_workers(self.worker)
        self._run_task = asyncio.create_task(self.runner.run())
        await asyncio.wait_for(self.started.wait(), TIMEOUT_SECS)

    async def wait_finished(self) -> None:
        assert self._run_task is not None
        await asyncio.wait_for(self._run_task, TIMEOUT_SECS)

    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()


def _make_transport() -> AnamTransport:
    return AnamTransport(
        api_key="test-api-key",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
    )


# --- AnamTransport --------------------------------------------------------------------


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
        await asyncio.wait_for(harness.errored.wait(), TIMEOUT_SECS)
        await asyncio.sleep(0.2)
        still_running = harness.is_running()

        await harness.worker.cancel()
        await harness.wait_finished()

    assert _deprecations(caught) == []
    assert still_running
    assert transport._input.is_usable is False
    [error] = harness.errors
    assert error.fatal is False


# --- AnamVideoService -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_video_service_connect_failure_cancels_pipeline_with_cancel_policy() -> None:
    """A failed Anam connect marks the service unusable and CANCEL stops the pipeline."""
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())

    with (
        patch.object(
            AnamClient,
            "connect_async",
            AsyncMock(side_effect=ConnectionError("connection refused")),
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        harness = _Harness(service, ProcessorUnusablePolicy.CANCEL)
        await harness.start()
        await harness.wait_finished()

    assert _deprecations(caught) == []
    assert harness.worker.has_finished()
    assert service.is_usable is False
    [error] = harness.errors
    assert error.processor is service
    assert error.fatal is False
    assert isinstance(error.exception, ConnectionError)
    assert "Error connecting to Anam" in error.error
