# Pipecat Anam Integration

[![PyPI - Version](https://img.shields.io/pypi/v/pipecat-anam)](https://pypi.python.org/pypi/pipecat-anam)

Generate real-time video avatars for your Pipecat AI agents with [Anam](https://anam.ai).

**Maintainer:** Anam ([@anam-org](https://github.com/anam-org))

## Installation

```bash
pip install pipecat-anam
```

Or with uv:

```bash
uv add pipecat-anam
```

You'll also need Pipecat with the services you use (STT, TTS, LLM, transport). For this repo's examples:

```bash
uv sync --extra dev --extra example
```

That installs all required Pipecat extras (`deepgram`, `cartesia`, `google`, `daily`, `runner`, `webrtc`) plus local tooling.

If you prefer pip:

```bash
pip install -e ".[dev,example]"
```

If you are building your own pipeline, install only the Pipecat extras you need.

## Prerequisites

- [Anam API key](https://lab.anam.ai)
- API keys for STT, TTS, and LLM (e.g., Deepgram, Cartesia, Google)
- A [Daily.co](https://www.daily.co/) room and (optional) meeting tokens for Daily WebRTC transport — see [Auto-provisioning the Daily room](#auto-provisioning-the-daily-room).

## Usage with Pipecat Pipeline

The `AnamVideoService` wraps around Anam's Python SDK for a seamless integration with Pipecat to create conversational AI applications where an Anam avatar provides synchronized video and audio output while your application handles the conversation logic. The AnamVideoService iterates over the (decoded) audio and video frames from Anam and passes them to the next service in the pipeline.

`enable_audio_passthrough=True` renders the avatar directly from your TTS audio (no separate Anam-side LLM or voice generation).

`enable_session_replay=False` disables Anam-side session recording.

```python
from anam import PersonaConfig
from pipecat_anam import AnamVideoService

persona_config = PersonaConfig(
    avatar_id="your-avatar-id",
    enable_audio_passthrough=True,
)

anam = AnamVideoService(
    api_key=os.environ["ANAM_API_KEY"],
    persona_config=persona_config,
    api_base_url="https://api.anam.ai",
    api_version="v1",
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    anam,  # Video avatar (returns synchronized audio/video)
    transport.output(),
    context_aggregator.assistant(),
])
```

See [examples/video-avatar-anam-video-service.py](examples/video-avatar-anam-video-service.py) for a complete working example.

## Initializing the Anam avatar session

`AnamVideoService` opens its connection to the Anam Backend asynchronously. The `StartFrame` is propagated downstream immediately so the rest of the pipeline (LLM/TTS/...) can warm up in parallel. TTS audio starts forwarding once the avatar is ready; any TTS produced before then is held back so it doesn't get dropped on the way in or accumulates latency.

Prior to v0.0.4, `AnamVideoService` blocked on `StartFrame` until the avatar was ready, which serialised pipeline startup. The async path keeps initial response latency low.

## Publishing directly to Daily

> [!WARNING]
> Direct Daily egress is experimental and only supported for Cara-4 avatars.
> The transport and signalling path will change in upcoming `anam` alpha
> releases. Pin to an exact alpha if you build on this; expect breaking
> changes between alphas.

`AnamTransport` is a drop-in replacement for Pipecat's `DailyTransport`. The Anam Backend publishes the avatar's synchronised audio and video **directly** into your Daily room, so the Pipecat bot doesn't have to receive and re-publish the avatar's A/V tracks.

The Daily room is bring-your-own: provision the room and mint two separate meeting tokens before starting the pipeline.
See the [Daily REST API docs](https://docs.daily.co/reference/rest-api) for `rooms` and `meeting-tokens` (or use [pipecat's Daily helpers](https://docs.pipecat.ai/server/services/transport/daily)).

- `daily_avatar_token` — for the Anam Backend (optional, but required for private rooms). If a `user_name` claim is set, it **must match** `daily_avatar_user_name` (or leave the claim empty). This lets the transport tell the avatar apart from end users. The transport will not forward TTS until the avatar has joined.
- `daily_bot_token` — for the Pipecat bot itself, used to capture the user's microphone for STT.

Requires `anam==0.5.0a1` (pinned exactly — see the SDK's experimental-alpha warning).

```python
from anam import PersonaConfig
from pipecat_anam import AnamTransport

transport = AnamTransport(
    api_key=os.environ["ANAM_API_KEY"],
    persona_config=PersonaConfig(
        avatar_id=os.environ["ANAM_AVATAR_ID"],
        # Direct Daily egress requires a Cara-4 avatar; stock avatars default to cara-3.
        avatar_model="cara-4-latest",
        enable_audio_passthrough=True,
    ),
    daily_room_url=os.environ["DAILY_ROOM_URL"],
    daily_bot_token=os.environ["DAILY_BOT_TOKEN"],
    daily_avatar_token=os.environ["DAILY_AVATAR_TOKEN"],
    daily_avatar_user_name=os.environ["DAILY_AVATAR_USER_NAME"],
)
```

### Auto-provisioning the Daily room

`AnamTransport` does not mint Daily rooms or tokens itself. If you'd rather provision a room programmatically than pre-create one, use Pipecat's [`DailyRESTHelper`](https://docs.pipecat.ai/server/services/transport/daily) with your `DAILY_API_KEY` to create the room and the two meeting tokens before constructing the transport:

```python
import aiohttp
from pipecat.transports.daily.utils import (
    DailyMeetingTokenParams,
    DailyMeetingTokenProperties,
    DailyRESTHelper,
    DailyRoomParams,
)

async with aiohttp.ClientSession() as session:
    helper = DailyRESTHelper(
        daily_api_key=os.environ["DAILY_API_KEY"],
        aiohttp_session=session,
    )
    room = await helper.create_room(DailyRoomParams())
    avatar_token = await helper.get_token(
        room.url,
        params=DailyMeetingTokenParams(
            properties=DailyMeetingTokenProperties(user_name="anam-avatar"),
        ),
    )
    bot_token = await helper.get_token(room.url)

    transport = AnamTransport(
        api_key=os.environ["ANAM_API_KEY"],
        persona_config=PersonaConfig(
            avatar_id=os.environ["ANAM_AVATAR_ID"],
            # Direct Daily egress requires a Cara-4 avatar; stock avatars default to cara-3.
            avatar_model="cara-4-latest",
            enable_audio_passthrough=True,
        ),
        daily_room_url=room.url,
        daily_avatar_token=avatar_token,
        daily_bot_token=bot_token,
    )
```

### Using the Pipecat runner (recommended for Pipecat Cloud)

When the Pipecat runner dispatches the session (locally via
`pipecat.runner.run.main`, or in [Pipecat Cloud](https://docs.pipecat.ai/pipecat-cloud/overview)), it hands the bot a `DailyRunnerArguments` with the Daily room URL and a *single* meeting token for the bot itself. 
`AnamTransport` needs *two* tokens (one for the bot, one for the avatar), so the bot mints the second one in-process via
`DailyRESTHelper` keyed on `DAILY_API_KEY`. The same file runs unchanged both locally and on Pipecat Cloud.

See [`examples/video-avatar-anam-transport-pcc.py`](examples/video-avatar-anam-transport-pcc.py)
for the full example.

## Video Post-Filter Example

The output transport scales the avatar resolution to the configured output resolution. When the aspect ratios mismatch the video is stretched or squeezed. To avoid this, apply a video post-processing filter that crops the avatar to the output aspect ratio.

[`examples/video-avatar-anam-postfilter.py`](examples/video-avatar-anam-postfilter.py) adds a `CenterAspectCropFilter` after `AnamVideoService`:

- Works on `OutputImageRawFrame`; does not depend on Anam internals.
- Assumes packed RGB24 bytes (`format="RGB"`).
- Performs a centered crop to match the configured output aspect ratio.
- Does not scale. Pipecat's output transport can still scale as needed.
- No-op when source and target aspect ratios already match.

The filter is self-contained in that file and can be lifted into any Pipecat pipeline that produces `OutputImageRawFrame`.

## Running the Example

1. Install dependencies:

```bash
uv sync --extra dev --extra example
```

2. Set up your environment:

```bash
cp env.example .env
# Edit .env with your API keys
```

3. Run the `AnamVideoService` example (Pipecat's built-in transports):

```bash
uv run python examples/video-avatar-anam-video-service.py -t daily
```

Or with the built-in WebRTC transport:

```bash
uv run python examples/video-avatar-anam-video-service.py -t webrtc
```

To run the `AnamTransport` example (direct Daily egress, BYO room and tokens, Deepgram + Google + Cartesia):

```bash
uv run python examples/video-avatar-anam-transport.py
```

To run the Pipecat-Cloud-shaped `AnamTransport` example (same pipeline, but the room is minted by the runner and the avatar token is minted in-process). Use `-d` so the runner prints a ready-to-click Daily URL:

```bash
uv run python examples/video-avatar-anam-transport-pcc.py -d
```

To run the center-aspect post-filter example with the WebRTC transport:

```bash
uv run python examples/video-avatar-anam-postfilter.py -t webrtc
```

Or with the Daily transport:

```bash
uv run python examples/video-avatar-anam-postfilter.py -t daily
```

## Compatibility

- **Tested with Pipecat v0.0.100+**
- Python 3.10+
- Daily transport or built-in WebRTC transport

## License

BSD-2-Clause - see [LICENSE](LICENSE)

## Support

- [Anam Lab](https://lab.anam.ai) (Build and test your persona and get your avatar_id.)
- [Anam Documentation](https://docs.anam.ai) (API reference and SDK documentation)
- [Anam Community Slack](https://join.slack.com/t/anamcommunity/shared_invite/zt-3qwaauo52-ZPqdt8HgW9u6T9iOshc_6Q)
- [Pipecat Discord](https://discord.gg/pipecat) (`#community-integrations`)
