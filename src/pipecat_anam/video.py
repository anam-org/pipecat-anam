#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam video service implementation for Pipecat.

This module provides integration with Anam.ai for creating interactive avatars
through Anam's Python SDK. It uses audio input and provides realistic avatars
as synchronized raw audio/video frames.
"""

import asyncio
from typing import Optional

from anam import (
    AgentAudioInputConfig,
    AgentAudioInputStream,
    AnamClient,
    AnamEvent,
    ClientOptions,
    ConnectionClosedCode,
    PersonaConfig,
    Session,
    SessionOptions,
)
from av.audio.resampler import AudioResampler
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    OutputTransportReadyFrame,
    SpeechOutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService

# Wait briefly for late TTSAudioRawFrames after TTSStoppedFrame before closing the sequence.
TTS_TIMEOUT = 0.35  # seconds


class AnamVideoService(AIService):
    """Anam.ai's Video service that generates real-time interactive avatars from audio.

    This service uses Anam's Python SDK to manage sessions and communication with Anam's backend.
    It consumes audio and user interactions and receives synchronized audio/video frames. The SDK
    provides decoded WebRTC audio and video frames as PyAV objects. Ingested audio is passed through
    without resampling, but has been resampled to 48kHz stereo for webRTC delivery to the SDK.


    The service supports:

    - Real-time avatar animation based on audio input
    - Voice activity detection for natural interactions
    - Interrupt handling for more natural conversations
    - Automatic session management
    """

    def __init__(
        self,
        *,
        api_key: str,
        persona_config: PersonaConfig,
        ice_servers: Optional[list[dict]] = None,
        api_base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        enable_session_replay: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the Anam video service.

        Args:
            api_key: Anam API key for authentication.
            persona_config: Full persona configuration.
            ice_servers: Custom ICE servers for WebRTC (optional).
            api_base_url: Base URL for the Anam API.
            api_version: API version to use.
            enable_session_replay: Whether to enable session recording on Anam's backend.
            **kwargs: Additional arguments passed to parent AIService.
        """
        super().__init__(**kwargs)
        self._api_key = api_key
        self._persona_config = persona_config
        self._ice_servers = ice_servers
        self._api_base_url = api_base_url
        self._api_version = api_version
        self._enable_session_replay = enable_session_replay

        self._client: Optional[AnamClient] = None
        self._anam_session: Optional[Session] = None
        self._agent_audio_stream: Optional[AgentAudioInputStream] = None
        self._send_task: Optional[asyncio.Task] = None
        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue[TTSStartedFrame | TTSAudioRawFrame | TTSStoppedFrame] = (
            asyncio.Queue()
        )
        self._anam_resampler = AudioResampler("s16", "mono", 48000)
        self._transport_ready = False
        self._session_ready_event = asyncio.Event()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the Anam video service with necessary configuration.

        Initializes the Anam client and prepares the service for audio/video
        processing. Sets up audio/video streams and registers event handlers.

        Args:
            setup: Configuration parameters for the frame processor.
        """
        await super().setup(setup)

        # Initialize Anam client
        self._client = AnamClient(
            api_key=self._api_key,
            persona_config=self._persona_config,
            options=ClientOptions(
                api_base_url=self._api_base_url or "https://api.anam.ai",
                ice_servers=self._ice_servers,
                api_version=self._api_version,
            ),
        )

        # Register event handlers
        self._client.add_listener(AnamEvent.SESSION_READY, self._on_session_ready)
        self._client.add_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)

    async def cleanup(self):
        """Clean up the service and release resources."""
        await super().cleanup()
        await self._shutdown_client()

    def _detach_client_listeners(self, client: AnamClient | None = None) -> None:
        """Remove registered client listeners if the client is still available."""
        client = client or self._client
        if client is None:
            return

        client.remove_listener(AnamEvent.SESSION_READY, self._on_session_ready)
        client.remove_listener(AnamEvent.CONNECTION_CLOSED, self._on_connection_closed)

    async def _shutdown_client(self) -> None:
        """Detach listeners, close the session, and clean up local resources."""
        client = self._client
        self._detach_client_listeners(client)
        await self._close_session()
        await self._cleanup()

    async def start(self, frame: StartFrame):
        """Start the Anam video service and initialize the avatar session.

        Creates an Anam session and creates tasks to forward audio/video. Blocks until
        session_ready is received so audio is not dropped before backend is ready to receive audio.

        Args:
            frame: The start frame containing initialization parameters.
        """
        if not self._client:
            raise RuntimeError("Anam client not initialized. Call setup() first.")

        self._session_ready_event.clear()

        try:
            # Block until session_ready so the backend can receive TTS
            logger.debug("Connecting to Anam Avatar service")
            self._anam_session = await self._client.connect_async(
                session_options=SessionOptions(enable_session_replay=self._enable_session_replay)
            )
            await asyncio.wait_for(self._session_ready_event.wait(), timeout=30)
        except Exception as e:
            error_msg = (
                "Anam session connection timed out."
                if isinstance(e, asyncio.TimeoutError)
                else f"Error connecting to Anam: {e}"
            )
            logger.error(error_msg)
            await self._close_session()
            await self.push_error_frame(ErrorFrame(error=error_msg, fatal=True))
            raise

        # Allow the pipeline to continue start up
        await super().start(frame)

        # Create agent audio input stream for sending TTS audio
        audio_config = AgentAudioInputConfig(
            encoding="pcm_s16le",
            sample_rate=frame.audio_out_sample_rate,
            channels=1,
        )
        try:
            self._agent_audio_stream = self._anam_session.create_agent_audio_input_stream(
                audio_config
            )
        except Exception as e:
            error_msg = f"Anam agent audio stream error: {e}"
            logger.error(error_msg)
            await self._close_session()
            await self.push_error_frame(ErrorFrame(error=error_msg, fatal=True))
            raise

        # Set sample rate from StartFrame (set via PipelineParams).
        self._anam_resampler = AudioResampler("s16", "mono", frame.audio_out_sample_rate)

        # Create tasks for consuming video and audio frames
        self._video_task = self.create_task(self._consume_video_frames())
        self._audio_task = self.create_task(self._consume_audio_frames())
        await self._create_send_task()

    async def stop(self, frame: EndFrame):
        """Stop the Anam video service gracefully.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._shutdown_client()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Anam video service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._shutdown_client()

    async def _cleanup(self):
        """Clean up resources: end conversation and cancel all tasks."""
        await self._cancel_video_task()
        await self._cancel_audio_task()
        await self._cancel_send_task()
        self._agent_audio_stream = None
        self._transport_ready = False
        self._client = None
        self._anam_session = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and coordinate avatar behavior.

        Handles different types of frames to manage avatar interactions:

        - TTSStartedFrame: Opens or refreshes the active TTS sequence
        - TTSAudioRawFrame: Queues audio for ordered delivery to Anam (not pushed downstream)
        - TTSStoppedFrame: Starts the late-audio grace period before closing the sequence
        - InterruptionFrame: Handles interruptions
        - OutputTransportReadyFrame: Sets transport ready flag
        - BotStartedSpeakingFrame: Stops TTFB metrics
        - Other frames: Forwards them through the pipeline

        Args:
            frame: The frame to be processed.
            direction: The direction of frame processing (input/output).
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)):
            if self._send_task:
                await self._queue.put(frame)
            if isinstance(frame, TTSAudioRawFrame):
                return  # Do not forward TTS audio downstream; Anam syncs TTS with video

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruption()
        if isinstance(frame, OutputTransportReadyFrame):
            self._transport_ready = True
        if isinstance(frame, BotStartedSpeakingFrame):
            await self.stop_ttfb_metrics()

        await self.push_frame(frame, direction)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def _normalize_tts_context_id(self, context_id: Optional[str]) -> str:
        """Normalize optional Pipecat TTS context IDs for local tracking."""
        return context_id if context_id is not None else "__legacy__"

    async def _consume_video_frames(self) -> None:
        """Consume video frames from Anam iterator and push them downstream."""
        if not self._anam_session:
            return

        try:
            async for video_frame in self._anam_session.video_frames():
                if not self._transport_ready:
                    continue

                frame = OutputImageRawFrame(
                    image=video_frame.to_ndarray(format="rgb24").tobytes(),
                    size=(video_frame.width, video_frame.height),
                    format="RGB",
                )

                await self.push_frame(frame)
        except Exception as e:
            error_msg = f"Anam error consuming video frames: {e}"
            logger.error(error_msg)
            await self.push_error_frame(ErrorFrame(error=error_msg))

    async def _consume_audio_frames(self) -> None:
        """Consume audio frames from Anam iterator and push them downstream.

        Audio frames are decoded WebRTC OPUS: 16 bit 48kHz stereo PCM samples.
        """
        if not self._anam_session:
            return

        try:
            async for audio_frame in self._anam_session.audio_frames():
                if not self._transport_ready:
                    continue

                # Resample to mono as some downstream transports cannot handle stereo audio.
                resampled_audio = self._anam_resampler.resample(audio_frame)
                for resampled_frame in resampled_audio:
                    frame = SpeechOutputAudioRawFrame(
                        audio=resampled_frame.to_ndarray().tobytes(),
                        sample_rate=self._anam_resampler.rate,
                        num_channels=self._anam_resampler.layout.nb_channels,
                    )
                    await self.push_frame(frame)

        except Exception as e:
            error_msg = f"Anam error consuming audio frames: {e}"
            logger.error(error_msg)
            await self.push_error_frame(ErrorFrame(error=error_msg))

    async def _cancel_video_task(self):
        """Cancel the video frame consumption task if it exists."""
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None

    async def _cancel_audio_task(self):
        """Cancel the audio frame consumption task if it exists."""
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

    async def _on_session_ready(self) -> None:
        """Handle session ready event (backend service is ready to receive audio).

        Unblocks the pipeline to propagate StartFrame and allow audio to be ingested.
        """
        logger.debug("Anam connection established")
        self._session_ready_event.set()

    async def _on_connection_closed(self, code: str, reason: Optional[str]) -> None:
        """Handle connection closed event.

        Client and session are closed by the SDK prior to emitting this event.

        Args:
            code: Connection close code (from ConnectionClosedCode enum).
            reason: Optional reason for closure.
        """
        if code != ConnectionClosedCode.NORMAL.value:
            error_message = f"Anam connection closed: {code}"
            if reason:
                error_message += f" - {reason}"
            logger.error(f"{error_message}")
            await self._cleanup()
            await self.push_error_frame(ErrorFrame(error=error_message))

    async def _handle_interruption(self) -> None:
        """Handle interruption events by signaling the session to interrupt."""
        if self._anam_session:
            await self._anam_session.interrupt()

        await self._cancel_send_task()
        if self._agent_audio_stream:
            # End sequence resets the audio chunk sequence number in the SDK.
            await self._agent_audio_stream.end_sequence()
        await self._create_send_task()

    async def _close_session(self):
        """Close the Anam client."""
        await self._cancel_send_task()
        if self._client and self._anam_session and self._anam_session.is_active:
            try:
                logger.debug("Disconnecting from Anam")
                await self._anam_session.close()
            except Exception as e:
                logger.warning(f"Error closing Anam session: {e}")
            finally:
                self._anam_session = None
                self._client = None

    async def _create_send_task(self):
        """Create the audio sending task if it doesn't exist."""
        if not self._send_task:
            self._queue = asyncio.Queue()
            self._send_task = self.create_task(self._send_task_handler())

    async def _cancel_send_task(self):
        """Cancel the audio sending task if it exists."""
        if self._send_task:
            await self.cancel_task(self._send_task)
            self._send_task = None

    async def _send_task_handler(self):
        """Serialize TTS chunks and end_sequence notifications sent to Anam.

        TTSStoppedFrame can arrive before the final TTSAudioRawFrame chunks, so once we see a
        stop frame we wait briefly for late audio before ending the current sequence.
        """
        if not self._agent_audio_stream:
            logger.error("Agent audio stream not initialized")
            return

        should_measure_ttfb = False
        waiting_for_end_sequence = False
        active_tts_context_id: str | None = None

        while True:
            if not self._agent_audio_stream:
                break
            try:
                if waiting_for_end_sequence:
                    frame = await asyncio.wait_for(self._queue.get(), timeout=TTS_TIMEOUT)
                else:
                    frame = await self._queue.get()

                try:
                    ctx = self._normalize_tts_context_id(frame.context_id)
                    if isinstance(frame, TTSStartedFrame):
                        if ctx != active_tts_context_id or waiting_for_end_sequence:
                            should_measure_ttfb = True
                        active_tts_context_id = ctx
                        waiting_for_end_sequence = False
                    elif isinstance(frame, TTSAudioRawFrame):
                        if ctx != active_tts_context_id:
                            # includes late TTSAudioRawFrames after TTSStoppedFrame.
                            continue
                        if frame.audio:
                            await self._agent_audio_stream.send_audio_chunk(frame.audio)
                            if should_measure_ttfb:
                                await self.start_ttfb_metrics()
                                should_measure_ttfb = False
                    elif isinstance(frame, TTSStoppedFrame):
                        if ctx == active_tts_context_id:
                            waiting_for_end_sequence = True
                finally:
                    self._queue.task_done()

            except asyncio.TimeoutError:
                if self._agent_audio_stream and waiting_for_end_sequence:
                    await self._agent_audio_stream.end_sequence()
                    active_tts_context_id = None
                should_measure_ttfb = False
                waiting_for_end_sequence = False

            except asyncio.CancelledError:
                raise

            except Exception as e:
                error_msg = f"Anam audio send error: {e}"
                logger.error(error_msg)
                await self.push_error_frame(ErrorFrame(error=error_msg))
                break
