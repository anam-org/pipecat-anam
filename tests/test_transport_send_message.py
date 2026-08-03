#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for :meth:`AnamOutputTransport.send_message` and
:meth:`AnamTransportClient.send_message`.

These exercise the delegation added so that ``transport.output().send_message(frame)``
actually reaches the Daily app-message channel on :class:`AnamTransport` sessions,
mirroring ``DailyOutputTransport.send_message``. No live Daily/Anam connection is used;
constructor arguments that would normally reach Daily/Anam (tokens, callbacks) are
dummy values or mocks.
"""

from unittest.mock import AsyncMock

import pytest
from anam import PersonaConfig
from pipecat.frames.frames import OutputTransportMessageUrgentFrame
from pipecat.transports.base_transport import TransportParams

from pipecat_anam.transport import AnamOutputTransport, AnamParams, AnamTransport, AnamTransportClient


def make_client() -> AnamTransportClient:
    return AnamTransportClient(
        bot_name="test-bot",
        daily_room_url="https://example.daily.co/test",
        daily_bot_token=None,
        daily_avatar_token=None,
        daily_avatar_user_name=None,
        api_key="test-api-key",
        persona_config=PersonaConfig(),
        api_base_url="https://api.anam.ai",
        api_version="v1",
        ice_servers=None,
        video_width=None,
        video_height=None,
        params=AnamParams(),
        on_connected=AsyncMock(),
        on_participant_connected=AsyncMock(),
        on_participant_disconnected=AsyncMock(),
        on_error=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_output_transport_send_message_delegates_to_client() -> None:
    """AnamOutputTransport.send_message awaits self._client.send_message with the frame."""
    frame = OutputTransportMessageUrgentFrame(message={"kind": "chat", "body": "hi"})
    transport = AnamOutputTransport(client=AsyncMock(), params=TransportParams())
    transport._client.send_message.return_value = None
    transport.push_error = AsyncMock()

    await transport.send_message(frame)

    transport._client.send_message.assert_awaited_once_with(frame)
    transport.push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_output_transport_send_message_pushes_error_on_failure() -> None:
    """A truthy error from the client triggers push_error; no exception is raised."""
    frame = OutputTransportMessageUrgentFrame(message={"kind": "chat", "body": "hi"})
    transport = AnamOutputTransport(client=AsyncMock(), params=TransportParams())
    transport._client.send_message.return_value = "boom"
    transport.push_error = AsyncMock()

    await transport.send_message(frame)

    transport.push_error.assert_awaited_once()
    (error_message,), _ = transport.push_error.call_args
    assert "boom" in error_message


@pytest.mark.asyncio
async def test_client_send_message_raises_before_setup() -> None:
    """AnamTransportClient.send_message raises RuntimeError when setup() hasn't run yet."""
    client = make_client()

    with pytest.raises(RuntimeError, match="send_message called before setup"):
        await client.send_message(OutputTransportMessageUrgentFrame(message="hi"))


@pytest.mark.asyncio
async def test_client_send_message_delegates_to_daily_client() -> None:
    """AnamTransportClient.send_message delegates to the underlying DailyTransportClient."""
    frame = OutputTransportMessageUrgentFrame(message="hi")
    client = make_client()
    client._daily_client = AsyncMock()
    client._daily_client.send_message.return_value = None

    result = await client.send_message(frame)

    client._daily_client.send_message.assert_awaited_once_with(frame)
    assert result is None


@pytest.mark.asyncio
async def test_client_send_director_note_cue_raises_before_session_active() -> None:
    """AnamTransportClient.send_director_note_cue raises without an active session."""
    client = make_client()

    with pytest.raises(RuntimeError, match="before Anam session is active"):
        await client.send_director_note_cue("warm")


@pytest.mark.asyncio
async def test_client_send_director_note_cue_delegates_to_session() -> None:
    """AnamTransportClient.send_director_note_cue delegates to the Anam Session."""
    client = make_client()
    client._session = AsyncMock()
    client._session.is_active = True

    await client.send_director_note_cue("warm", at_seconds=0.0)

    client._session.send_director_note_cue.assert_awaited_once_with(
        tag="warm",
        at_seconds=0.0,
        in_seconds=None,
    )


@pytest.mark.asyncio
async def test_transport_send_director_note_cue_delegates_to_client() -> None:
    """AnamTransport.send_director_note_cue forwards cue messages to its client."""
    transport = AnamTransport(
        api_key="test-api-key",
        persona_config=PersonaConfig(enable_audio_passthrough=True),
        daily_room_url="https://example.daily.co/test",
        params=AnamParams(),
    )
    transport._client = AsyncMock()

    await transport.send_director_note_cue("warm", in_seconds=0.25)

    transport._client.send_director_note_cue.assert_awaited_once_with(
        tag="warm",
        at_seconds=None,
        in_seconds=0.25,
    )
