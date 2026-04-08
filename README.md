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

## Video Post-Filter Example (Anam-Agnostic)

[`example_video_post_filter.py`](example_video_post_filter.py) adds an anam-agnostic
post-filter after `AnamVideoService`:

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

To run the center-aspect post-filter example:

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
