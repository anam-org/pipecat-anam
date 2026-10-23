#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared helpers for the pipecat-anam test suite.

No live Daily/Anam connection is used anywhere: sessions, Daily clients and frame
iterators are fakes, and pipelines run in a real ``PipelineWorker`` only where the
test is about how the worker reacts to an error.
"""

import asyncio
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from anam import AnamClient, PersonaConfig
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_anam import AnamTransport, AnamVideoService
from pipecat_anam.transport import ANAM_AVATAR_USER_NAME, AnamParams, AnamTransportClient

TIMEOUT_SECS = 10

# The session-ready watchdog in AnamVideoService._connect_session is hardcoded to 30 s.
_SESSION_READY_TIMEOUT_SECS = 30


def make_client() -> AnamTransportClient:
    return AnamTransportClient(
        bot_name="test-bot",
        daily_room_url="https://example.daily.co/test",
        daily_bot_token=None,
        daily_avatar_token=None,
        daily_avatar_user_name=None,
        api_key="test-api-key",
        persona_config=PersonaConfig(),
        api_base_url="https://api.anam.ai",
        api_version="v1",
        ice_servers=None,
        video_width=None,
        video_height=None,
        show_ai_avatar_disclosure=None,
        region=None,
        region_policy=None,
        params=AnamParams(),
        on_connected=AsyncMock(),
        on_participant_connected=AsyncMock(),
        on_participant_disconnected=AsyncMock(),
        on_error=AsyncMock(),
    )


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


def _make_transport() -> AnamTransport:
    return AnamTransport(
        api_key="test-api-key",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
    )


def _deprecations(caught: list[warnings.WarningMessage]) -> list[str]:
    # The stdlib `audioop` warning comes from Pipecat's audio utils, not from this package.
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "audioop" not in str(w.message)
    ]


def _avatar() -> dict[str, Any]:
    """A Daily participant dict for the Anam avatar."""
    return {"id": "a", "info": {"userName": ANAM_AVATAR_USER_NAME}}


def _human() -> dict[str, Any]:
    """A Daily participant dict for an end user."""
    return {"id": "h", "info": {"userName": "alice"}}


async def _blocking_frames(cancelled: asyncio.Event) -> AsyncIterator[Any]:
    """Frame iterator that never yields. Sets ``cancelled`` once its consumer is cancelled."""
    try:
        await asyncio.Event().wait()
        yield
    finally:
        cancelled.set()


async def _raising_frames(exc: Exception, *, yield_first: bool = False) -> AsyncIterator[Any]:
    """Frame iterator that raises ``exc`` on the first ``__anext__``.

    ``yield_first`` gives the event loop one turn first, so two consumers can fail in
    the same loop tick.
    """
    if yield_first:
        await asyncio.sleep(0)
    raise exc
    yield


class _FakeSession:
    """Stand-in for :class:`anam.Session`.

    ``video_frames`` and ``audio_frames`` are zero-argument factories returning the
    async iterators the service consumes; both block forever by default. The task each
    iterator was requested from is recorded, so a test can check whether the consumer
    task was cancelled even after the service has dropped its reference.
    """

    def __init__(
        self,
        *,
        video_frames: Callable[[], AsyncIterator[Any]] | None = None,
        audio_frames: Callable[[], AsyncIterator[Any]] | None = None,
    ) -> None:
        self.region = None
        self.session_id = "s-1"
        self.is_active = True
        self.close = AsyncMock(side_effect=self._mark_closed)
        self.interrupt = AsyncMock()
        self.send_director_note_cue = AsyncMock()
        self.audio_stream = MagicMock(send_audio_chunk=AsyncMock(), end_sequence=AsyncMock())
        self.create_agent_audio_input_stream = MagicMock(return_value=self.audio_stream)
        self._video_frames = video_frames or (lambda: _blocking_frames(asyncio.Event()))
        self._audio_frames = audio_frames or (lambda: _blocking_frames(asyncio.Event()))
        self.connect_task: asyncio.Task | None = None
        self.video_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None

    async def _mark_closed(self) -> None:
        self.is_active = False

    def video_frames(self) -> AsyncIterator[Any]:
        self.video_task = asyncio.current_task()
        return self._video_frames()

    def audio_frames(self) -> AsyncIterator[Any]:
        self.audio_task = asyncio.current_task()
        return self._audio_frames()


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

    async def wait_errored(self) -> None:
        await asyncio.wait_for(self.errored.wait(), TIMEOUT_SECS)

    async def wait_finished(self) -> None:
        assert self._run_task is not None
        await asyncio.wait_for(self._run_task, TIMEOUT_SECS)

    async def cancel(self) -> None:
        await self.worker.cancel()
        await self.wait_finished()

    def is_running(self) -> bool:
        return self._run_task is not None and not self._run_task.done()


async def _wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT_SECS) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _make_video_service() -> AnamVideoService:
    return AnamVideoService(api_key="k", persona_config=_passthrough_persona())


async def _connected_service(
    policy: ProcessorUnusablePolicy,
    session: _FakeSession,
    service: AnamVideoService | None = None,
) -> tuple[AnamVideoService, _Harness]:
    """Start an AnamVideoService whose ``connect_async`` returns ``session``.

    Returns once the frame consumer tasks exist (or an error was already reported).
    The session-ready event is never set, so the connect task is left waiting on its
    watchdog. Pass ``service`` to instrument it before it starts.
    """
    service = service or _make_video_service()

    async def connect_async(_client: AnamClient, *args: Any, **kwargs: Any) -> _FakeSession:
        session.connect_task = asyncio.current_task()
        return session

    with patch.object(AnamClient, "connect_async", connect_async):
        harness = _Harness(service, policy)
        await harness.start()
        await _wait_until(lambda: service._video_task is not None or harness.errored.is_set())
    return service, harness


@contextmanager
def _fast_ready_timeout(seconds: float = 0.05) -> Iterator[None]:
    """Shorten AnamVideoService's 30 s session-ready watchdog to ``seconds``.

    Only waits with exactly that timeout are shortened, so the harness's own
    ``TIMEOUT_SECS`` waits are unaffected.
    """
    real_wait_for = asyncio.wait_for

    async def wait_for(fut: Any, timeout: float | None = None) -> Any:
        if timeout == _SESSION_READY_TIMEOUT_SECS:
            timeout = seconds
        return await real_wait_for(fut, timeout)

    with patch("asyncio.wait_for", wait_for):
        yield
