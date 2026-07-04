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
the transport objects are instantiated via ``__new__`` so only the attributes each method
touches need to be set up.
"""

from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import OutputTransportMessageUrgentFrame

from pipecat_anam.transport import AnamOutputTransport, AnamTransportClient


@pytest.mark.asyncio
async def test_output_transport_send_message_delegates_to_client() -> None:
    """AnamOutputTransport.send_message awaits self._client.send_message with the frame."""
    frame = OutputTransportMessageUrgentFrame(message={"kind": "chat", "body": "hi"})
    transport = AnamOutputTransport.__new__(AnamOutputTransport)
    transport._client = AsyncMock()
    transport._client.send_message.return_value = None
    transport.push_error = AsyncMock()

    await transport.send_message(frame)

    transport._client.send_message.assert_awaited_once_with(frame)
    transport.push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_output_transport_send_message_pushes_error_on_failure() -> None:
    """A truthy error from the client triggers push_error; no exception is raised."""
    frame = OutputTransportMessageUrgentFrame(message={"kind": "chat", "body": "hi"})
    transport = AnamOutputTransport.__new__(AnamOutputTransport)
    transport._client = AsyncMock()
    transport._client.send_message.return_value = "boom"
    transport.push_error = AsyncMock()

    await transport.send_message(frame)

    transport.push_error.assert_awaited_once()
    (error_message,), _ = transport.push_error.call_args
    assert "boom" in error_message


@pytest.mark.asyncio
async def test_client_send_message_raises_before_setup() -> None:
    """AnamTransportClient.send_message raises RuntimeError when setup() hasn't run yet."""
    client = AnamTransportClient.__new__(AnamTransportClient)
    client._daily_client = None

    with pytest.raises(RuntimeError, match="send_message called before setup"):
        await client.send_message(OutputTransportMessageUrgentFrame(message="hi"))


@pytest.mark.asyncio
async def test_client_send_message_delegates_to_daily_client() -> None:
    """AnamTransportClient.send_message delegates to the underlying DailyTransportClient."""
    frame = OutputTransportMessageUrgentFrame(message="hi")
    client = AnamTransportClient.__new__(AnamTransportClient)
    client._daily_client = AsyncMock()
    client._daily_client.send_message.return_value = None

    result = await client.send_message(frame)

    client._daily_client.send_message.assert_awaited_once_with(frame)
    assert result is None
