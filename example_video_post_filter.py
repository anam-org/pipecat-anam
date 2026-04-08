#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: Anam avatar with an anam-agnostic center-aspect video post-filter.

The post-filter only crops RGB24 bytes to match the configured output aspect ratio.
It never rescales. Pipecat's output transport can still scale to final dimensions.
"""

import os
from typing import Final

from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from examples.video_post_filter import CenterAspectCropFilter
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_anam import AnamVideoService

load_dotenv(override=True)

VIDEO_OUT_SIZE: Final[tuple[int, int]] = (1200, 800)
REQUIRED_ENV_VARS: Final[list[str]] = [
    "ANAM_API_KEY",
    "DEEPGRAM_API_KEY",
    "CARTESIA_API_KEY",
    "GOOGLE_API_KEY",
]

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=VIDEO_OUT_SIZE[0],
        video_out_height=VIDEO_OUT_SIZE[1],
        video_out_bitrate=1_000_000,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=VIDEO_OUT_SIZE[0],
        video_out_height=VIDEO_OUT_SIZE[1],
    ),
}


def ensure_required_env_vars() -> None:
    """Fail fast when a required key is missing."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Build and run a single bot task."""
    ensure_required_env_vars()

    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.google.llm import GoogleLLMService

    logger.info("Starting bot with center-aspect video post-filter")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="e8e5fffb-252c-436d-b842-8879b84445b6",
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    persona_config = PersonaConfig(
        avatar_id=os.getenv("ANAM_AVATAR_ID", "071b0286-4cce-4808-bee2-e642f1062de3").strip(
            '"'
        ),
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
            video_post_filter,  # Crops by aspect only; no scaling.
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
    """Entry point used by `pipecat.runner.run`."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

