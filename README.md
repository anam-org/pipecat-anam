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
- [Daily.co](https://www.daily.co/) API key for WebRTC transport (optional)

## Usage with Pipecat Pipeline

The `AnamVideoService` wraps around Anam's Python SDK for a seamless integration with Pipecat to create conversational AI applications where an Anam avatar provides synchronized video and audio output while your application handles the conversation logic. The AnamVideoService iterates over the (decoded) audio and video frames from Anam and passes them to the next service in the pipeline.

`enable_audio_passthrough=True` bypasses Anam's orchestration layer and renders the avatar directly from TTS audio.

`enable_session_replay=False` disables session recording on Anam's backend.

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

See [example.py](example.py) for a complete working example.

## Initializing the Anam avatar session

On initialization, the `AnamVideoService` starts a non-blocking connection to the Anam backend. The `StartFrame` is propagated downstream immediately, and the `AnamVideoService` buffers TTS frames while the avatar backend is warming up. Only when `SESSION_READY` is received, the `AnamVideoService` will start forwarding TTS audio. If we don't wait for `SESSION_READY`, the audio will be dropped at the backend, as the engine conservatively drops incoming TTS to avoid accumulating audio in the buffer that can cause a latency buildup.

Up to and including v.0.0.3, the `AnamVideoService` blocked on `StartFrame` until the avatar backend was ready to receive audio. This results in higher pipeline startup latency as the other pipeline components (LLM/TTS/...) can only start and generate output after the avatar backend is available.

## Publishing directly to Daily

> [!WARNING]
> Direct Daily egress is experimental and only supported for Cara-4 avatars.
> The transport and signalling path will change in upcoming `anam` alpha
> releases. Pin to an exact alpha if you build on this; expect breaking
> changes between alphas.

`AnamTransport` is a drop-in replacement for Pipecat's `DailyTransport` that has the Anam backend publish the avatar's synchronised audio + video **directly**
into your Daily room. This avoids routing the avatar through the Pipecat bot and removes the bot's A/V receive-and-republish overhead.

The Daily room is bring-your-own: provision the room and mint two separate meeting tokens before starting the pipeline.
See the [Daily REST API docs](https://docs.daily.co/reference/rest-api) for `rooms` and `meeting-tokens` (or use [pipecat's Daily helpers](https://docs.pipecat.ai/server/services/transport/daily)).

- `daily_avatar_token` — for the Anam backend. Its `user_name` claim **must match** `daily_avatar_user_name` (or leave claim empty). This is required for the transport to tell the avatar apart from end users.
- `daily_bot_token` — for the Pipecat bot itself, used to capture the user's microphone for STT.

Requires `anam==0.5.0a1` (pinned exactly — see the SDK's experimental-alpha warning).

```python
from anam import PersonaConfig
from pipecat_anam import AnamTransport

transport = AnamTransport(
    api_key=os.environ["ANAM_API_KEY"],
    persona_config=PersonaConfig(avatar_id=os.environ["ANAM_AVATAR_ID"]),
    daily_room_url=os.environ["DAILY_ROOM_URL"],
    daily_bot_token=os.environ["DAILY_BOT_TOKEN"],
    daily_avatar_token=os.environ["DAILY_AVATAR_TOKEN"],
    daily_avatar_user_name=os.environ["DAILY_AVATAR_USER_NAME"],
)

```
## Video Post-Filter Example

The output transport scales the avatar resolution to the specified output resolution. This result in an amorphous scaling when the aspect ratios between output and avatar mismatch, i.e., the video is stretched or squeezed in on or both dimensions. To avoid this, you can apply a video post-processing filter to crop the avatar to the output aspect ratio.

[`example_video_post_filter.py`](example_video_post_filter.py) adds a video
post processing filter after `AnamVideoService`:

- It works on `OutputImageRawFrame` and does not depend on Anam internals.
- It assumes packed RGB24 bytes (`format="RGB"`).
- It performs a centered crop to match the configured output aspect ratio.
- It does not scale. Pipecat output transport can still scale as needed.
- It is a no-op when source and target aspect ratios already match.

The reusable helper lives in [`examples/video_post_filter.py`](examples/video_post_filter.py).
The same helper can be used with any Pipecat service producing `OutputImageRawFrame`.

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

3. Run:

```bash
uv run python example.py -t daily
```

Or with the built-in WebRTC transport:

```bash
uv run python example.py -t webrtc
```

The bot will create a room (or use the built-in client) with a video avatar that responds to your voice.

To run the Anam transport example:

```bash
uv run python example-anam-transport.py
```

To run the center-aspect post-filter example:

```bash
uv run python example_video_post_filter.py
```
or with the Daily transport:
```bash
uv run python example_video_post_filter.py -t daily
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
