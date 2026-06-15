#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal Pipecat voice assistant with an Anam avatar over Daily egress for PCC.

This example illustrates the pattern for deploying ``AnamTransport``
to `Pipecat Cloud <https://docs.pipecat.ai/pipecat-cloud/overview>`_:

- The Pipecat runner hands the bot a ``DailyRunnerArguments`` containing 
  the Daily room URL and a *single* meeting token.
- Using ``DailyRESTHelper`` with ``DAILY_API_KEY`` to mint a second token for the avatar.

The file works in Pipecat Cloud and Locally:

- **Pipecat Cloud**: drop this file into your Pipecat Cloud project. 
  Then follow your regular deploy routine. The runner-args come from PCC's Start API.

- **Locally**: run with ``uv run python examples/video-avatar-anam-transport-pcc.py -d``,
  then join the created Daily room.

Required env vars:

    ANAM_API_KEY           Anam API key.
    DAILY_API_KEY          Daily REST API key able to mint tokens for the
                           room that the runner provides.
    DEEPGRAM_API_KEY       Deepgram STT.
    CARTESIA_API_KEY       Cartesia TTS.
    GOOGLE_API_KEY         Google Gemini LLM.

Optional env vars:

    ANAM_AVATAR_ID         Avatar id (defaults to a public sample).
    DAILY_ROOM_URL         Pin local runs to a specific Daily room instead
                           of letting the runner create a fresh one.
    DAILY_AVATAR_USER_NAME Avatar display name. Must match the ``user_name``
                           claim in ``DAILY_AVATAR_TOKEN``. Defaults to
                           ``"anam-avatar"``.
"""

import os

import aiohttp
from anam import PersonaConfig
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import DailyRunnerArguments, RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.daily.utils import (
    DailyMeetingTokenParams,
    DailyMeetingTokenProperties,
    DailyRESTHelper,
)

from pipecat_anam import ANAM_AVATAR_USER_NAME, AnamTransport

load_dotenv(override=True)


async def run_bot(transport: AnamTransport) -> None:
    """Wire the cascade pipeline around an already-constructed AnamTransport."""

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
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Your output will be spoken aloud, so avoid emojis or other "
                "characters that can't easily be spoken. Be succinct and "
                "respond to what the user said in a creative and helpful way."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, participant):
        logger.info("Client disconnected")
        await task.cancel()

    @transport.event_handler("on_avatar_connected")
    async def on_avatar_connected(transport, participant):
        logger.info("Avatar connected")

    @transport.event_handler("on_avatar_disconnected")
    async def on_avatar_disconnected(transport, participant, reason):
        logger.info(f"Avatar disconnected. Reason: {reason}")

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Pipecat runner entrypoint. Called by both Pipecat Cloud and the local runner."""

    if not isinstance(runner_args, DailyRunnerArguments):
        logger.error(
            f"AnamTransport requires the Daily runner (-t daily). Got: {type(runner_args)}"
        )
        return

    async with aiohttp.ClientSession() as http:
        helper = DailyRESTHelper(
            daily_api_key=os.environ["DAILY_API_KEY"],
            aiohttp_session=http,
        )
        avatar_user_name = os.getenv("DAILY_AVATAR_USER_NAME", ANAM_AVATAR_USER_NAME)
        # Mint a second token for the avatar.
        avatar_token = await helper.get_token(
            runner_args.room_url,
            params=DailyMeetingTokenParams(
                properties=DailyMeetingTokenProperties(user_name=avatar_user_name),
            ),
        )

        transport = AnamTransport(
            api_key=os.environ["ANAM_API_KEY"],
            persona_config=PersonaConfig(
                avatar_id=os.getenv("ANAM_AVATAR_ID", "071b0286-4cce-4808-bee2-e642f1062de3"),
                avatar_model="cara-4-latest",
                enable_audio_passthrough=True,
            ),
            daily_room_url=runner_args.room_url,
            daily_bot_token=runner_args.token,
            daily_avatar_token=avatar_token,
            daily_avatar_user_name=avatar_user_name,
        )

        await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
