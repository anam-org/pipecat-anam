#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for the ``dev_settings`` option on :class:`AnamVideoService`.

``dev_settings`` carries a generic per-session ``devSettings`` object (e.g.
``{"low_latency_frame_output": True}``) that the engine resolves ahead of
PostHog feature flags. It must be stored and forwarded into the Anam SDK's
``ClientOptions.dev_settings`` so the session request carries it. Silently
dropping it would let the engine fall back to its default flag resolution. No
live connection is made.
"""

import inspect

import pytest
from anam import PersonaConfig
from anam.types import ClientOptions
from pipecat.services.ai_service import AIService

from pipecat_anam import AnamVideoService


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


async def _noop_setup(self, setup) -> None:
    """Stand-in for AIService.setup so tests exercise only AnamVideoService.setup."""
    return


def test_video_service_stores_dev_settings() -> None:
    dev_settings = {"low_latency_frame_output": True}
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        dev_settings=dev_settings,
    )
    assert service._dev_settings == dev_settings


def test_video_service_defaults_to_no_dev_settings() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())
    assert service._dev_settings is None


class _FakeClient:
    def __init__(self, *, api_key, persona_config, options):
        self.options = options

    def add_listener(self, *args, **kwargs) -> None:
        pass


@pytest.mark.asyncio
async def test_setup_forwards_dev_settings_to_client_options(monkeypatch) -> None:
    """setup() must thread `dev_settings` into ClientOptions so it reaches the
    session request (or fail loudly if the SDK cannot carry it)."""
    dev_settings = {"low_latency_frame_output": True}
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        dev_settings=dev_settings,
    )

    monkeypatch.setattr(AIService, "setup", _noop_setup, raising=True)
    monkeypatch.setattr("pipecat_anam.video.AnamClient", _FakeClient)

    supports_dev_settings = "dev_settings" in inspect.signature(ClientOptions).parameters
    try:
        await service.setup(None)
    except RuntimeError:
        # Expected only when the installed SDK's ClientOptions lacks `dev_settings`.
        assert not supports_dev_settings
        return

    assert supports_dev_settings
    assert service._client.options.dev_settings == dev_settings


@pytest.mark.asyncio
async def test_setup_omits_dev_settings_when_unset(monkeypatch) -> None:
    """Without an override, ClientOptions carries no dev_settings (backwards
    compatible with SDKs whose ClientOptions predates the field)."""
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())

    monkeypatch.setattr(AIService, "setup", _noop_setup, raising=True)
    monkeypatch.setattr("pipecat_anam.video.AnamClient", _FakeClient)

    await service.setup(None)

    assert getattr(service._client.options, "dev_settings", None) is None
