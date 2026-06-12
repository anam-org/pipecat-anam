#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam transport for Pipecat with direct egress into a customer-supplied Daily room.

:class:`AnamTransport` joins a Daily room as the Pipecat bot, tells Anam to publish and avatar
into the same room, and forwards TTS + interrupts over Anam's WebSocket.

The caller brings the Daily room URL and meeting tokens; the transport does not mint either.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any, Optional

from anam import (
    AgentAudioInputConfig,
    AgentAudioInputStream,
    AnamClient,
    AnamEvent,
    ClientOptions,
    ConnectionClosedCode,
    EgressDailyOptions,
    EgressOptions,
    PersonaConfig,
    Session,
    SessionOptions,
)
from daily.daily import AudioData
from loguru import logger

from pipecat.frames.frames import (
    BotConnectedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import (
    DailyCallbacks,
    DailyParams,
    DailyTransportClient,
)


# Default display name the Anam avatar joins Daily with.
ANAM_AVATAR_USER_NAME = "anam-avatar"

# Default Pipecat bot display name in the Daily room.
PIPECAT_BOT_NAME = "Pipecat"

# Opt-in subscription policy: don't subscribe to any remote tracks by default.
# capture_participant_audio() then explicitly subscribes to end-user mics only.
_OPT_IN_SUBSCRIPTION_PROFILE = {
    "base": {
        "media": {
            "camera": "unsubscribed",
            "screenVideo": "unsubscribed",
            "microphone": "unsubscribed",
        }
    }
}

# Late-audio grace period after TTSStoppedFrame before we send `end_sequence`.
END_OF_UTTERANCE_TIMEOUT = 0.35  # seconds

# Hard ceiling for detecting the avatar joined the Daily room.
AVATAR_CONNECT_TIMEOUT = 30.0  # seconds


class AnamParams(DailyParams):
    """Configuration parameters for the Anam transport.

    Remote microphones are never subscribed by default. User mics are
    subscribed only when audio capture starts for the input track.
    """

    audio_in_enabled: bool = True
    # Per-participant renderers are required for selective user-mic capture.
    audio_in_user_tracks: bool = True
    # Disables BaseOutputTransport as we use the out-of-band Anam SDK to ingest TTS.
    audio_out_enabled: bool = False
    # Anam publishes the avatar's A/V; Pipecat must not compete on those tracks.
    microphone_out_enabled: bool = False
    camera_out_enabled: bool = False


class AnamTransportClient:
    """Owns one Daily room participant + one Anam SDK session.

    - :meth:`setup` wires up the Daily client.
    - :meth:`start` connects to Anam, joins the Daily room in parallel, then waits for the avatar.
    - :meth:`stop` leaves Daily and closes the Anam session.
    """

    def __init__(
        self,
        *,
        bot_name: str,
        daily_room_url: str,
        daily_bot_token: Optional[str],
        daily_avatar_token: Optional[str],
        daily_avatar_user_name: Optional[str],
        api_key: str,
        persona_config: PersonaConfig,
        api_base_url: str,
        api_version: str,
        ice_servers: Optional[list[dict]],
        params: AnamParams,
        on_connected: Callable[[Mapping[str, Any]], Awaitable[None]],
        on_participant_connected: Callable[[Mapping[str, Any]], Awaitable[None]],
        on_participant_disconnected: Callable[[Mapping[str, Any], str], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]],
    ) -> None:
        self._bot_name = bot_name
        self._daily_room_url = daily_room_url
        self._daily_bot_token = daily_bot_token
        self._daily_avatar_token = daily_avatar_token
        self._daily_avatar_user_name = daily_avatar_user_name or ANAM_AVATAR_USER_NAME
        self._api_key = api_key
        self._persona_config = persona_config
        self._api_base_url = api_base_url
        self._api_version = api_version
        self._ice_servers = ice_servers
        self._params = params
        self._on_connected = on_connected
        self._on_participant_connected = on_participant_connected
        self._on_participant_disconnected = on_participant_disconnected
        self._on_error = on_error

        self._daily_client: Optional[DailyTransportClient] = None
        self._session: Optional[Session] = None
        self._agent_audio_stream: Optional[AgentAudioInputStream] = None
        # Set when the avatar participant joins Daily (name match).
        self._avatar_connected_event = asyncio.Event()
        # Distinguishes intentional shutdown from an unexpected disconnect.
        self._stop_called: bool = False

    async def setup(self, setup: FrameProcessorSetup) -> None:
        """Wire up the Daily client."""
        if self._daily_client is not None:
            return
        logger.debug("AnamTransportClient: setting up Daily client")
        daily_callbacks = DailyCallbacks(
            on_active_speaker_changed=partial(
                self._on_handle_callback, "on_active_speaker_changed"
            ),
            on_joined=self._on_joined,
            on_left=self._on_left,
            on_before_leave=partial(self._on_handle_callback, "on_before_leave"),
            on_error=self._on_error,
            on_app_message=partial(self._on_handle_callback, "on_app_message"),
            on_call_state_updated=partial(self._on_handle_callback, "on_call_state_updated"),
            on_client_connected=partial(self._on_handle_callback, "on_client_connected"),
            on_client_disconnected=partial(self._on_handle_callback, "on_client_disconnected"),
            on_dialin_connected=partial(self._on_handle_callback, "on_dialin_connected"),
            on_dialin_ready=partial(self._on_handle_callback, "on_dialin_ready"),
            on_dialin_stopped=partial(self._on_handle_callback, "on_dialin_stopped"),
            on_dialin_error=partial(self._on_handle_callback, "on_dialin_error"),
            on_dialin_warning=partial(self._on_handle_callback, "on_dialin_warning"),
            on_dialout_answered=partial(self._on_handle_callback, "on_dialout_answered"),
            on_dialout_connected=partial(self._on_handle_callback, "on_dialout_connected"),
            on_dialout_stopped=partial(self._on_handle_callback, "on_dialout_stopped"),
            on_dialout_error=partial(self._on_handle_callback, "on_dialout_error"),
            on_dialout_warning=partial(self._on_handle_callback, "on_dialout_warning"),
            on_dtmf_event=partial(self._on_handle_callback, "on_dtmf_event"),
            on_participant_joined=self._on_participant_connected,
            on_participant_left=self._on_participant_disconnected,
            on_participant_updated=partial(self._on_handle_callback, "on_participant_updated"),
            on_transcription_message=partial(self._on_handle_callback, "on_transcription_message"),
            on_recording_started=partial(self._on_handle_callback, "on_recording_started"),
            on_recording_stopped=partial(self._on_handle_callback, "on_recording_stopped"),
            on_recording_error=partial(self._on_handle_callback, "on_recording_error"),
            on_transcription_stopped=partial(self._on_handle_callback, "on_transcription_stopped"),
            on_transcription_error=partial(self._on_handle_callback, "on_transcription_error"),
        )
        self._daily_client = DailyTransportClient(
            self._daily_room_url,
            self._daily_bot_token,
            self._bot_name,
            self._params,
            daily_callbacks,
            "AnamPipecat",
        )
        await self._daily_client.setup(setup)

    async def cleanup(self) -> None:
        """Tear down the Anam session lifecycle and the Daily client. Idempotent."""
        await self.stop()
        if self._daily_client is not None:
            try:
                await self._daily_client.cleanup()
            except Exception as exc:
                logger.error(f"AnamTransportClient: error during Daily cleanup: {exc}")

    async def start(self, frame: StartFrame) -> None:
        """Connect to Anam, join Daily in parallel, then wait for the avatar."""
        if self._daily_client is None:
            raise RuntimeError("AnamTransportClient not initialized. Call setup() first.")
        logger.debug("AnamTransportClient: Connecting to Anam Avatar service")
        anam_task = asyncio.create_task(self._anam_connect())
        daily_task = asyncio.create_task(self._daily_start_join(frame))
        try:
            try:
                await asyncio.gather(anam_task, daily_task)
            except BaseException:
                anam_task.cancel()
                daily_task.cancel()
                # Drain so cancellation actually completes and any session that
                # was assigned to self._session is visible to stop().
                await asyncio.gather(anam_task, daily_task, return_exceptions=True)
                raise
            try:
                await asyncio.wait_for(
                    self._avatar_connected_event.wait(),
                    timeout=AVATAR_CONNECT_TIMEOUT,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"AnamTransport: avatar did not join Daily within {AVATAR_CONNECT_TIMEOUT:.0f}s"
                ) from exc
            if self._stop_called:
                return
            self._create_agent_audio_stream(frame)
        except Exception as exc:
            await self.stop()
            await self._on_error(f"AnamTransport failed to start: {exc}")
            return

    async def _anam_connect(self) -> None:
        """Open the Anam signalling session."""
        anam_client = AnamClient(
            api_key=self._api_key,
            persona_config=self._persona_config,
            options=ClientOptions(
                api_base_url=self._api_base_url,
                api_version=self._api_version,
                ice_servers=self._ice_servers,
                client_label="Pipecat:AnamTransport",
            ),
        )
        anam_client.add_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)
        # Session replay is not supported for AnamTransport avatars.
        session_options = SessionOptions(
            enable_session_replay=False,
            egress=EgressOptions(
                mode="daily",
                daily=EgressDailyOptions(
                    room_url=self._daily_room_url,
                    token=self._daily_avatar_token,
                    user_name=self._daily_avatar_user_name,
                ),
            ),
        )
        self._session = await anam_client.connect_async(session_options=session_options)

    def _create_agent_audio_stream(self, frame: StartFrame) -> None:
        if self._session is None:
            raise RuntimeError(
                "Anam session was not established before creating agent audio stream"
            )
        audio_config = AgentAudioInputConfig(
            encoding="pcm_s16le",
            sample_rate=frame.audio_out_sample_rate,
            channels=1,
        )
        self._agent_audio_stream = self._session.create_agent_audio_input_stream(audio_config)

    async def _daily_start_join(self, frame: StartFrame) -> None:
        """Start the Daily client and join the room."""
        if self._daily_client is None:
            raise RuntimeError("AnamTransportClient not initialized. Call setup() first.")
        await self._daily_client.start(frame)
        await self._daily_client.join()
        # DailyTransportClient.join() reports failures via on_error and returns silently.
        # We need a hard raise to prevent returning "successful" while the bot is not in the room.
        if not getattr(self._daily_client, "_joined", False):
            raise RuntimeError(f"Failed to join Daily room {self._daily_room_url}")
        await self.update_subscriptions(profile_settings=_OPT_IN_SUBSCRIPTION_PROFILE)

    @property
    def stop_called(self) -> bool:
        """Whether :meth:`stop` has been entered."""
        return self._stop_called

    async def stop(self) -> None:
        """Leave Daily and close the Anam session. Idempotent."""
        if self._stop_called:
            return
        self._stop_called = True
        self._avatar_connected_event.clear()

        if self._daily_client is not None:
            try:
                await self._daily_client.leave()
            except Exception as exc:
                logger.error(f"AnamTransportClient: error leaving Daily: {exc}")
        if self._session is not None and self._session.is_active:
            try:
                await self._session.close()
            except Exception as exc:
                logger.error(f"AnamTransportClient: error closing Anam session: {exc}")
        self._session = None
        self._agent_audio_stream = None

    async def _on_joined(self, data: Mapping[str, Any]) -> None:
        await self._on_connected(data)

    async def _on_left(self) -> None:
        logger.debug("AnamTransportClient: Daily client left the room")
        # Surface unexpected leaves (room ended, kicked, token revoked, ...)
        # as fatal so the pipeline runner unwinds.
        if not self._stop_called:
            await self._on_error(f"Daily room disconnected unexpectedly: {self._daily_room_url}")

    async def _on_handle_callback(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        logger.trace(
            f"AnamTransportClient[Daily callback] {event_name} args={args} kwargs={kwargs}"
        )

    async def _on_connection_closed(self, code: str, reason: Optional[str]) -> None:
        self._session = None

        # NORMAL is Anam's WS-protocol "graceful close" code.
        if code == ConnectionClosedCode.NORMAL.value:
            return

        await self._on_error(f"Anam session closed unexpectedly: {code} ({reason})")

    @property
    def agent_audio_stream(self) -> Optional[AgentAudioInputStream]:
        """The Anam backend's TTS PCM input stream, or None until the avatar joins the Daily room."""
        return self._agent_audio_stream

    @property
    def session(self) -> Optional[Session]:
        """The active Anam SDK session, or None until start() completes / after stop()."""
        return self._session

    @property
    def avatar_connected_event(self) -> asyncio.Event:
        """Set when the avatar participant joins Daily (``on_avatar_connected``)."""
        return self._avatar_connected_event

    def signal_avatar_connected(self) -> None:
        """Unblock :meth:`start` once the avatar participant is present. Idempotent."""
        self._avatar_connected_event.set()

    # --- Daily-side helpers exposed to input/output transports ----------------------

    async def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable[[str, AudioData, str], Awaitable[None]],
        audio_source: str = "microphone",
        sample_rate: Optional[int] = None,
        callback_interval_ms: int = 20,
    ) -> None:
        if self._daily_client is None:
            raise RuntimeError("capture_participant_audio called before setup() completed.")
        await self._daily_client.capture_participant_audio(
            participant_id,
            callback,
            audio_source,
            sample_rate or self._daily_client.in_sample_rate,
            callback_interval_ms,
        )

    async def update_subscriptions(
        self,
        participant_settings: Optional[Mapping[str, Any]] = None,
        profile_settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._daily_client is None:
            raise RuntimeError("update_subscriptions called before setup() completed.")
        await self._daily_client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    @property
    def in_sample_rate(self) -> int:
        if self._daily_client is None:
            raise RuntimeError("AnamTransportClient not initialized. Call setup() first.")
        return self._daily_client.in_sample_rate

    @property
    def out_sample_rate(self) -> int:
        if self._daily_client is None:
            raise RuntimeError("AnamTransportClient not initialized. Call setup() first.")
        return self._daily_client.out_sample_rate


class AnamInputTransport(BaseInputTransport):
    """Receives end-user microphone from the Daily room and pushes ``InputAudioRawFrame``."""

    def __init__(
        self,
        client: AnamTransportClient,
        params: TransportParams,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self._client = client
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True
        # setup BaseInputTransport._audio_in_queue before audio callbacks can fire.
        await self.set_transport_ready(frame)
        await self._client.start(frame)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._client.stop()

    async def start_capturing_audio(self, participant: Mapping[str, Any]) -> None:
        if not self._params.audio_in_enabled:
            return
        participant_id = participant["id"]
        logger.debug(f"AnamInputTransport: capturing audio for participant {participant_id}")
        await self._client.capture_participant_audio(
            participant_id=participant_id,
            callback=self._on_participant_audio_data,
            sample_rate=self._client.in_sample_rate,
        )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ) -> None:
        frame = InputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


class AnamOutputTransport(BaseOutputTransport):
    """Routes outbound TTS audio + interrupts to the Anam backend via WebSocket.

    We bypass ``BaseOutputTransport`` and ingest TTS directly into the Anam backend to reduce latency
    and avoid HoL blocking on late chunks.

    Per-context filtering: every utterance carries a ``context_id`` on its lifecycle
    frames. We gate TTS ingestion by tracking the active context.

    End-of-utterance grace: ``TTSStoppedFrame`` can arrive *before* the final audio
    chunks, so ``end_sequence`` runs on a short delay (:data:`END_OF_UTTERANCE_TIMEOUT`)
    to let stragglers through.

    Interrupts: Resets the active context and drops any late chunks.
    """

    def __init__(
        self,
        client: AnamTransportClient,
        params: TransportParams,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, **kwargs)
        self._client = client
        self._initialized = False
        # TTS-context state machine. See class docstring for rationale.
        self._active_tts_context_id: Optional[str] = None
        self._end_sequence_task: Optional[asyncio.Task] = None
        # We emit Bot{Started,Stopped}SpeakingFrame manually because we
        # bypassed BaseOutputTransport's audio path (audio_out_enabled=False).
        self._bot_speaking: bool = False

    async def setup(self, setup: FrameProcessorSetup) -> None:
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        if self._initialized:
            return
        self._initialized = True
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._client.stop()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # TTSAudioRawFrame is swallowed to prevent BaseOutputTransport from sending it outbound.
        # Lifecycle frames still flow through super() so downstream aggregators / metrics see them.
        if isinstance(frame, TTSStartedFrame):
            await self._on_tts_started(frame)
            await super().process_frame(frame, direction)
            return
        if isinstance(frame, TTSAudioRawFrame):
            await self._on_tts_audio(frame)
            return
        if isinstance(frame, TTSStoppedFrame):
            await self._on_tts_stopped(frame)
            await super().process_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            await self._on_interruption()
            await super().process_frame(frame, direction)
            return
        await super().process_frame(frame, direction)

    # --- TTS state machine ------------------------------------------------------

    async def _on_tts_started(self, frame: TTSStartedFrame) -> None:
        await self._cancel_end_sequence_task()
        ctx = self._normalize_context_id(getattr(frame, "context_id", None))
        if ctx != self._active_tts_context_id:
            logger.debug(f"AnamOutputTransport: TTS context started: {ctx}")
        self._active_tts_context_id = ctx

    async def _on_tts_audio(self, frame: TTSAudioRawFrame) -> None:
        ctx = self._normalize_context_id(getattr(frame, "context_id", None))
        if ctx != self._active_tts_context_id:
            logger.warning(
                f"AnamOutputTransport: dropping stale TTS chunk "
                f"(context={ctx}, active={self._active_tts_context_id})"
            )
            return
        if not frame.audio:
            return
        stream = self._client.agent_audio_stream
        if stream is None:
            return
        # Don't re-open speaking state for late chunks arriving inside the grace period.
        if not self._bot_speaking and (
            self._end_sequence_task is None or self._end_sequence_task.done()
        ):
            self._bot_speaking = True
            await self.broadcast_frame(BotStartedSpeakingFrame)
        await stream.send_audio_chunk(bytes(frame.audio))

    async def _on_tts_stopped(self, frame: TTSStoppedFrame) -> None:
        ctx = self._normalize_context_id(getattr(frame, "context_id", None))
        if ctx != self._active_tts_context_id:
            return
        # Broadcast BotStoppedSpeaking to unblock upstream plugins (e.g. function calling).
        if self._bot_speaking:
            self._bot_speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)
        self._end_sequence_task = asyncio.create_task(
            self._send_end_sequence_after_grace(self._active_tts_context_id)
        )

    async def _on_interruption(self) -> None:
        await self._cancel_end_sequence_task()
        session = self._client.session
        if session is not None:
            try:
                await session.interrupt()
            except Exception as exc:
                logger.error(f"AnamOutputTransport: interrupt failed: {exc}")

        stream = self._client.agent_audio_stream
        if stream is not None:
            try:
                await stream.end_sequence()
            except Exception as exc:
                logger.error(f"AnamOutputTransport: end_sequence on interrupt failed: {exc}")
        self._active_tts_context_id = None
        if self._bot_speaking:
            self._bot_speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)

    async def _cancel_end_sequence_task(self) -> None:
        task = self._end_sequence_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _send_end_sequence_after_grace(self, context_id: Optional[str]) -> None:
        await asyncio.sleep(END_OF_UTTERANCE_TIMEOUT)
        if self._active_tts_context_id != context_id:
            return
        stream = self._client.agent_audio_stream
        if stream is None:
            return
        try:
            await stream.end_sequence()
        except Exception as exc:
            logger.error(f"AnamOutputTransport: end_sequence failed: {exc}")
            return
        self._active_tts_context_id = None

    @staticmethod
    def _normalize_context_id(context_id: Optional[str]) -> str:
        """Map ``None`` context_ids (TTS services that don't emit one) to a sentinel
        so the state machine's equality checks still work."""
        return context_id if context_id is not None else "_no_context"


class AnamTransport(BaseTransport):
    """Pipecat transport that owns one Daily room and one Anam avatar session.

    Drop-in replacement for ``DailyTransport`` that has Anam publish the avatar
    directly into the Daily room.

    Event handlers:

    - ``on_connected(transport, data)`` — Pipecat bot joined the Daily room.
    - ``on_client_connected(transport, participant)`` / ``on_client_disconnected(...)`` —
      a non-Anam participant joined / left.
    - ``on_avatar_connected(transport, participant)`` / ``on_avatar_disconnected(transport, participant, reason)`` —
      the Anam egress participant joined / left. Disconnect forces a fatal pipeline
      shutdown since the avatar does not auto-rejoin.
    - ``on_error(transport, error)`` — unrecoverable transport error; a fatal
      ``ErrorFrame`` is also pushed into the pipeline.
    """

    def __init__(
        self,
        *,
        api_key: str,
        persona_config: PersonaConfig,
        daily_room_url: str,
        daily_avatar_token: Optional[str] = None,
        daily_bot_token: Optional[str] = None,
        daily_avatar_user_name: Optional[str] = None,
        bot_name: str = PIPECAT_BOT_NAME,
        params: AnamParams = AnamParams(),
        api_base_url: str = "https://api.anam.ai",
        api_version: str = "v1",
        ice_servers: Optional[list[dict]] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> None:
        """Initialize the Anam transport.

        Args:
            api_key: Anam API key.
            persona_config: Avatar persona configuration. ``enable_audio_passthrough`` must be True.
            daily_room_url: Customer-supplied Daily room URL that both the Anam
                avatar and the Pipecat bot will join.
            daily_avatar_token: Daily meeting token for the Anam avatar. Each token
                is single-use; omit for public rooms.
            daily_bot_token: Daily meeting token for the Pipecat bot. Same constraints
                as ``daily_avatar_token``.
            daily_avatar_user_name: Display name the avatar joins Daily with;
                must match the ``user_name`` claim of ``daily_avatar_token`` (if any).
            bot_name: Display name for the Pipecat bot in the Daily room.
            params: Transport parameters. The default :class:`AnamParams` keeps the
                Pipecat bot publish-disabled so it doesn't compete with the avatar.
            api_base_url, api_version, ice_servers: Pass-through to the Anam SDK.
            input_name, output_name: Optional Pipecat transport names.

        Raises:
            ValueError: if ``persona_config.enable_audio_passthrough`` is not True.
        """
        # ``enable_audio_passthrough`` must be true for the avatar to be driven by your TTS.
        if not persona_config.enable_audio_passthrough:
            raise ValueError("AnamTransport requires PersonaConfig(enable_audio_passthrough=True).")
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params
        self._daily_avatar_user_name = daily_avatar_user_name or ANAM_AVATAR_USER_NAME

        self._client = AnamTransportClient(
            bot_name=bot_name,
            daily_room_url=daily_room_url,
            daily_bot_token=daily_bot_token,
            daily_avatar_token=daily_avatar_token,
            daily_avatar_user_name=self._daily_avatar_user_name,
            api_key=api_key,
            persona_config=replace(persona_config, enable_audio_passthrough=True),
            api_base_url=api_base_url,
            api_version=api_version,
            ice_servers=ice_servers,
            params=params,
            on_connected=self._on_connected,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_error=self._on_fatal_error,
        )
        self._input: Optional[AnamInputTransport] = None
        self._output: Optional[AnamOutputTransport] = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_avatar_connected")
        self._register_event_handler("on_avatar_disconnected")
        self._register_event_handler("on_error")

    def input(self) -> FrameProcessor:
        if self._input is None:
            self._input = AnamInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        if self._output is None:
            self._output = AnamOutputTransport(client=self._client, params=self._params)
        return self._output

    async def update_subscriptions(
        self,
        participant_settings: Optional[Mapping[str, Any]] = None,
        profile_settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        await self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    async def _on_fatal_error(self, error: str) -> None:
        """Propagate an unrecoverable transport error into the pipeline."""
        await self._call_event_handler("on_error", error)
        if self._input is not None:
            await self._input.push_error(error_msg=error, fatal=True)
        elif self._output is not None:
            await self._output.push_error(error_msg=error, fatal=True)
        else:
            logger.error(f"AnamTransport - fatal error: {error}")

    async def _on_connected(self, data: Mapping[str, Any]) -> None:
        await self._call_event_handler("on_connected", data)
        if self._input is not None:
            await self._input.push_frame(BotConnectedFrame())

    def _is_avatar_participant(self, participant: Mapping[str, Any]) -> bool:
        """Return True if remote participant is the Anam avatar."""
        info = participant.get("info") or {}
        user_name = info.get("userName") or ""
        return user_name == self._daily_avatar_user_name

    async def _on_participant_connected(self, participant: Mapping[str, Any]) -> None:
        if self._is_avatar_participant(participant):
            self._client.signal_avatar_connected()
            await self._call_event_handler("on_avatar_connected", participant)
            return

        await self._call_event_handler("on_client_connected", participant)
        if self._input is not None:
            await self._input.push_frame(ClientConnectedFrame())
            await self._input.start_capturing_audio(participant)

    async def _on_participant_disconnected(
        self, participant: Mapping[str, Any], reason: str
    ) -> None:
        if self._is_avatar_participant(participant):
            await self._call_event_handler("on_avatar_disconnected", participant, reason)
            # Fatal: the Anam service does not rejoin a Daily room, so any produced TTS goes nowhere.
            if not self._client.stop_called:
                await self._on_fatal_error(
                    f"Anam avatar participant left Daily room unexpectedly: {reason}"
                )
            return
        await self._call_event_handler("on_client_disconnected", participant)
