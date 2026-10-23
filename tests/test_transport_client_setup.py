#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for how :class:`AnamTransportClient` wires up and joins Daily.

``DailyTransportClient`` is mocked, so no live Daily/Anam connection is used.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import make_client
from pipecat.frames.frames import StartFrame
from pipecat.transports.daily.transport import DailyCallbacks

from pipecat_anam.transport import _OPT_IN_SUBSCRIPTION_PROFILE


def _mock_daily_client(client, *, joined: bool) -> MagicMock:
    daily = MagicMock(leave=AsyncMock())

    async def join() -> None:
        if joined:
            await client._on_joined({})

    daily.join = AsyncMock(side_effect=join)
    return daily


@pytest.mark.asyncio
async def test_setup_builds_daily_callbacks_with_all_fields() -> None:
    """setup() passes a complete DailyCallbacks and builds the Daily client only once."""
    client = make_client()
    setup = MagicMock(audio_out_sample_rate=24000)

    with patch("pipecat_anam.transport.DailyTransportClient") as daily_cls:
        daily_cls.return_value.setup = AsyncMock()
        await client.setup(setup)
        await client.setup(setup)

    daily_cls.assert_called_once()
    callbacks = daily_cls.call_args.args[4]
    assert isinstance(callbacks, DailyCallbacks)
    # Catches callbacks that a later Pipecat version makes required.
    for name in DailyCallbacks.model_fields:
        callback = getattr(callbacks, name)
        assert callback is not None, name
        assert callable(callback), name
    assert callbacks.on_left == client._on_left
    assert callbacks.on_error is client._on_error
    assert callbacks.on_participant_left is client._on_participant_disconnected
    status_callback = callbacks.on_network_connection_status_updated
    assert status_callback.args == ("on_network_connection_status_updated",)
    await status_callback({"status": "x"})
    daily_cls.return_value.setup.assert_awaited_once_with(setup)


@pytest.mark.asyncio
async def test_daily_join_without_on_joined_fails_start() -> None:
    """A Daily join that never fires on_joined fails start() and stops the client."""
    client = make_client()
    daily = _mock_daily_client(client, joined=False)
    client._daily_client = daily
    client._anam_connect = AsyncMock()
    client.update_subscriptions = AsyncMock()

    await client.start(StartFrame())

    client._on_error.assert_awaited_once()
    [message] = client._on_error.await_args.args
    assert message.startswith(
        "AnamTransport failed to start: Failed to join Daily room https://example.daily.co/test"
    )
    client.update_subscriptions.assert_not_awaited()
    daily.leave.assert_awaited_once()
    assert client.stop_called is True


@pytest.mark.asyncio
async def test_daily_join_on_joined_updates_subscriptions() -> None:
    """A successful Daily join applies the opt-in subscription profile."""
    client = make_client()
    daily = _mock_daily_client(client, joined=True)
    client._daily_client = daily
    client.update_subscriptions = AsyncMock()

    await client._daily_start_join()

    daily.join.assert_awaited_once()
    client.update_subscriptions.assert_awaited_once_with(
        profile_settings=_OPT_IN_SUBSCRIPTION_PROFILE
    )
