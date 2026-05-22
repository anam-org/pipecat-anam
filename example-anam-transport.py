#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice assistant with an Anam avatar using direct Daily egress.

The customer brings their own Daily room URL and two meeting tokens (minted
before running this example). The Anam engine joins as the avatar publisher;
the Pipecat bot joins separately to capture the user's microphone for STT.

Required env vars:

    ANAM_API_KEY           Anam API key.
    ANAM_AVATAR_ID         Avatar id (optional; defaults to a public sample).
    DAILY_ROOM_URL         https://your-domain.daily.co/<room>
    DAILY_AVATAR_TOKEN     Meeting token for the Anam engine egress sidecar.
    DAILY_BOT_TOKEN        Meeting token for the Pipecat bot (separate JWT).
    DEEPGRAM_API_KEY       Deepgram STT.
    CARTESIA_API_KEY       Cartesia TTS.
    GOOGLE_API_KEY         Google Gemini LLM.

Optional env vars:

    DAILY_USER_NAME        Display name for the Anam egress sidecar (default:
                           anam-avatar). The avatar meeting token's user_name
                           claim must match this value so the transport can
                           tell the avatar apart from end users.
"""

import asyncio
import os
import sys

from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService

from pipecat_anam import AnamTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = AnamTransport(
        api_key=os.environ["ANAM_API_KEY"],
        api_base_url=os.getenv("ANAM_API_BASE_URL", "https://api.anam.ai"),
        api_version="v1",
        persona_config=PersonaConfig(
            avatar_id=os.getenv("ANAM_AVATAR_ID", "071b0286-4cce-4808-bee2-e642f1062de3"),
        ),
        daily_room_url=os.environ["DAILY_ROOM_URL"],
        daily_avatar_token=os.getenv("DAILY_AVATAR_TOKEN"),
        daily_bot_token=os.getenv("DAILY_BOT_TOKEN"),
        daily_user_name=os.getenv("DAILY_USER_NAME"),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="e8e5fffb-252c-436d-b842-8879b84445b6",
        ),
    )

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_connected")
    async def on_connected(transport, data):
        logger.info(f"Pipecat bot joined Daily room: {os.environ['DAILY_ROOM_URL']}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Start by greeting the user and ask how you can help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, participant):
        logger.info("Client disconnected")
        await task.cancel()

    @transport.event_handler("on_avatar_connected")
    async def on_avatar_connected(transport, participant):
        logger.info("Anam avatar connected (joined Daily room)")

    @transport.event_handler("on_avatar_disconnected")
    async def on_avatar_disconnected(transport, participant, reason):
        logger.info(f"Anam avatar disconnected. Reason: {reason}")

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
