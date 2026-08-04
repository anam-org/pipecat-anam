#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal Pipecat voice assistant with an Anam avatar over Daily egress.

Demonstrates :class:`AnamTransport`: a customer-supplied Daily room, a
Deepgram STT / Google LLM / Cartesia TTS pipeline, and the avatar published
directly into the room by the Anam Backend.

Required env vars:

    ANAM_API_KEY           Anam API key.
    DAILY_ROOM_URL         https://your-domain.daily.co/<room>
    DAILY_AVATAR_TOKEN     Daily meeting token for the Anam Avatar.
    DAILY_BOT_TOKEN        Daily meeting token for the Pipecat bot.
    DEEPGRAM_API_KEY       Deepgram STT.
    CARTESIA_API_KEY       Cartesia TTS.
    GOOGLE_API_KEY         Google Gemini LLM.

Optional env vars:

    ANAM_AVATAR_ID         Avatar id (defaults to a public sample).
    ANAM_AVATAR_MODEL      Avatar model.
    ANAM_VIDEO_WIDTH       Avatar output width. Defaults to ``1152`` (Cara-4 landscape).
    ANAM_VIDEO_HEIGHT      Avatar output height. Defaults to ``768`` (Cara-4 landscape).
    DAILY_AVATAR_USER_NAME Avatar display name. Must match the ``user_name``
                           claim in ``DAILY_AVATAR_TOKEN``. Defaults to
                           ``"anam-avatar"``.
"""

import asyncio
import os
import sys

from anam import DirectorNotes, PersonaConfig
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
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
    video_width = int(os.getenv("ANAM_VIDEO_WIDTH", "1152"))
    video_height = int(os.getenv("ANAM_VIDEO_HEIGHT", "768"))

    transport = AnamTransport(
        api_key=os.environ["ANAM_API_KEY"],
        persona_config=PersonaConfig(
            avatar_id=os.getenv("ANAM_AVATAR_ID", "071b0286-4cce-4808-bee2-e642f1062de3"),
            # Direct Daily egress requires a Cara-4 avatar; stock avatars default to cara-3.
            avatar_model=os.getenv("ANAM_AVATAR_MODEL", "cara-4") or None,
            director_notes=DirectorNotes(
                preset_style="warm",
                expressivity=0.7,
            ),
            enable_audio_passthrough=True,
        ),
        daily_room_url=os.environ["DAILY_ROOM_URL"],
        daily_avatar_token=os.getenv("DAILY_AVATAR_TOKEN"),
        daily_bot_token=os.getenv("DAILY_BOT_TOKEN"),
        daily_avatar_user_name=os.getenv("DAILY_AVATAR_USER_NAME"),
        video_width=video_width,
        video_height=video_height,
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
            system_instruction="You are a helpful assistant. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Be succinct and respond to what the user said in a creative and helpful way.",
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
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Start by saying 'Hello' and then a short greeting.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

        # Demonstrate broadcasting a Daily app-message from the bot. This reaches every
        # other participant in the room over the data channel, independent of TTS/audio.
        await transport.output().send_message(
            OutputTransportMessageUrgentFrame(
                message={"kind": "chat", "body": "hello from the bot"}
            )
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, participant):
        logger.info("Client disconnected")
        await task.cancel()

    @transport.event_handler("on_avatar_connected")
    async def on_avatar_connected(transport, participant):
        logger.info("Avatar connected")
        # Demonstrate a runtime director-note cue applied to the current turn.
        await transport.send_director_note_cue(tag="happy", at_seconds=2.0)

    @transport.event_handler("on_avatar_disconnected")
    async def on_avatar_disconnected(transport, participant, reason):
        logger.info(f"Avatar disconnected. Reason: {reason}")

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
