#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anam avatar with a center-aspect video post-filter.

The output transport scales the avatar resolution to the configured output
size. When the source and target aspect ratios differ this stretches the
image. The filter below center-crops packed RGB24 ``OutputImageRawFrame``
bytes to match the target aspect ratio without rescaling. Pipecat's output
transport still scales to the final dimensions afterwards.

Run with the Daily transport (``-t daily``) or the built-in WebRTC
transport (``-t webrtc``).
"""

from __future__ import annotations

import os
from typing import Final

from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMRunFrame, OutputImageRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_anam import AnamVideoService

load_dotenv(override=True)

VIDEO_OUT_SIZE: Final[tuple[int, int]] = (720, 720)
RGB24_BYTES_PER_PIXEL: Final[int] = 3
REQUIRED_ENV_VARS: Final[list[str]] = [
    "ANAM_API_KEY",
    "DEEPGRAM_API_KEY",
    "CARTESIA_API_KEY",
    "GOOGLE_API_KEY",
]


def center_crop_rgb24_bytes_to_aspect(
    image: bytes,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[bytes, tuple[int, int]]:
    """Center-crop packed RGB24 bytes to match ``target_size`` aspect ratio.

    Crops only; never up- or downscales. Returns the cropped bytes and
    cropped ``(width, height)``. If the source already matches the target
    aspect ratio, returns the input unchanged.
    """
    source_width, source_height = source_size
    target_width, target_height = target_size

    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"source_size must be positive, got {source_size!r}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target_size must be positive, got {target_size!r}")

    expected_len = source_width * source_height * RGB24_BYTES_PER_PIXEL
    if len(image) != expected_len:
        raise ValueError(
            f"image length does not match source_size for RGB24: got {len(image)}, "
            f"expected {expected_len}"
        )

    # Compare aspect ratios via cross-multiplication to avoid float drift.
    lhs = source_width * target_height
    rhs = source_height * target_width
    if lhs == rhs:
        return image, source_size

    stride_bytes = source_width * RGB24_BYTES_PER_PIXEL
    if lhs > rhs:
        # Source wider than target -> crop left/right.
        cropped_width = int(round((source_height * target_width) / target_height))
        cropped_width = max(1, min(cropped_width, source_width))
        if cropped_width == source_width:
            return image, source_size

        x = (source_width - cropped_width) // 2
        cropped_row_bytes = cropped_width * RGB24_BYTES_PER_PIXEL
        cropped = bytearray(cropped_row_bytes * source_height)
        dst = 0
        for row in range(source_height):
            src_start = row * stride_bytes + x * RGB24_BYTES_PER_PIXEL
            src_end = src_start + cropped_row_bytes
            cropped[dst : dst + cropped_row_bytes] = image[src_start:src_end]
            dst += cropped_row_bytes
        return bytes(cropped), (cropped_width, source_height)

    # Source taller (or narrower) than target -> crop top/bottom.
    cropped_height = int(round((source_width * target_height) / target_width))
    cropped_height = max(1, min(cropped_height, source_height))
    if cropped_height == source_height:
        return image, source_size

    y = (source_height - cropped_height) // 2
    src_start = y * stride_bytes
    src_end = src_start + cropped_height * stride_bytes
    return image[src_start:src_end], (source_width, cropped_height)


def center_crop_output_image_frame(
    frame: OutputImageRawFrame, target_size: tuple[int, int]
) -> OutputImageRawFrame:
    """Center-crop an ``OutputImageRawFrame`` (RGB24) to the target aspect ratio."""
    if frame.format != "RGB":
        raise ValueError(
            f"CenterAspectCropFilter expects RGB24 OutputImageRawFrame, got {frame.format!r}"
        )

    cropped_image, cropped_size = center_crop_rgb24_bytes_to_aspect(
        frame.image,
        source_size=frame.size,
        target_size=target_size,
    )
    if cropped_size == frame.size:
        return frame

    cropped_frame = OutputImageRawFrame(
        image=cropped_image,
        size=cropped_size,
        format=frame.format,
    )
    cropped_frame.pts = frame.pts
    cropped_frame.metadata = dict(frame.metadata)
    cropped_frame.transport_source = frame.transport_source
    cropped_frame.transport_destination = frame.transport_destination
    # ``sync_with_audio`` is present on newer Pipecat (>=0.0.104) but not on 0.0.103.
    if hasattr(frame, "sync_with_audio"):
        setattr(cropped_frame, "sync_with_audio", getattr(frame, "sync_with_audio"))
    return cropped_frame


class CenterAspectCropFilter(FrameProcessor):
    """Output-image filter that center-crops to a target aspect ratio."""

    def __init__(self, target_size: tuple[int, int]):
        super().__init__()
        self._target_size = target_size

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            frame = center_crop_output_image_frame(frame, self._target_size)
        await self.push_frame(frame, direction)


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=VIDEO_OUT_SIZE[0],
        video_out_height=VIDEO_OUT_SIZE[1],
        video_out_bitrate=5_000_000,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=VIDEO_OUT_SIZE[0],
        video_out_height=VIDEO_OUT_SIZE[1],
        video_out_bitrate=5_000_000,
    ),
}


def ensure_required_env_vars() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    ensure_required_env_vars()
    logger.info("Starting bot with center-aspect video post-filter")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="e8e5fffb-252c-436d-b842-8879b84445b6",
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    persona_config = PersonaConfig(
        avatar_id=os.getenv("ANAM_AVATAR_ID", "071b0286-4cce-4808-bee2-e642f1062de3").strip('"'),
        enable_audio_passthrough=True,
    )
    anam = AnamVideoService(
        api_key=os.getenv("ANAM_API_KEY"),
        persona_config=persona_config,
        api_base_url="https://api.anam.ai",
        api_version="v1",
        enable_session_replay=False,
    )

    video_post_filter = CenterAspectCropFilter(target_size=VIDEO_OUT_SIZE)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Your output will be spoken aloud, so avoid "
                "special characters that cannot easily be spoken, such as emojis."
            ),
        },
    ]
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            anam,
            video_post_filter,  # Aspect-only crop; no scaling.
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if isinstance(transport, DailyTransport):
            await transport.update_publishing(
                publishing_settings={"camera": {"sendSettings": {"allowAdaptiveLayers": True}}}
            )
        messages.append(
            {
                "role": "system",
                "content": "Start by saying hello and then give a short greeting.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
