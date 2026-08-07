#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for the ``environment`` engine-routing option on
:class:`AnamVideoService`.

``environment`` pins a session to a specific engine (e.g. ``{"podName": ...}``);
it must be stored and forwarded into the Anam SDK's ``ClientOptions.environment``
so the session request carries the routing. Silently dropping it (the pre-fix
behaviour) routes the session to a default engine. No live connection is made.
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


def test_video_service_stores_environment() -> None:
    routing = {"podName": "anam-engine-abc-123"}
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        environment=routing,
    )
    assert service._environment == routing


def test_video_service_defaults_to_no_environment() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())
    assert service._environment is None


class _FakeClient:
    def __init__(self, *, api_key, persona_config, options):
        self.options = options

    def add_listener(self, *args, **kwargs) -> None:
        pass


@pytest.mark.asyncio
async def test_setup_forwards_environment_to_client_options(monkeypatch) -> None:
    """setup() must thread `environment` into ClientOptions so the pin reaches
    the session request (or fail loudly if the SDK cannot carry it)."""
    routing = {"podName": "anam-engine-abc-123"}
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        environment=routing,
    )

    # Isolate setup() from the pipecat base and the real client.
    monkeypatch.setattr(AIService, "setup", _noop_setup, raising=True)
    monkeypatch.setattr("pipecat_anam.video.AnamClient", _FakeClient)

    supports_environment = "environment" in inspect.signature(ClientOptions).parameters
    try:
        await service.setup(None)
    except RuntimeError:
        # Expected only when the installed SDK's ClientOptions lacks `environment`.
        assert not supports_environment
        return

    assert supports_environment
    assert service._client.options.environment == routing


@pytest.mark.asyncio
async def test_setup_omits_environment_when_unset(monkeypatch) -> None:
    """Without a pin, ClientOptions carries no environment (backwards compatible
    with SDKs whose ClientOptions predates the field)."""
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())

    monkeypatch.setattr(AIService, "setup", _noop_setup, raising=True)
    monkeypatch.setattr("pipecat_anam.video.AnamClient", _FakeClient)

    await service.setup(None)

    assert getattr(service._client.options, "environment", None) is None
