#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the ``video_width`` / ``video_height`` (portrait) options on both
:class:`AnamVideoService` and :class:`AnamTransport`.

These assert the paired-argument validation and that the dimensions are forwarded into
the Anam SDK's ``SessionOptions`` (which serializes them to ``videoWidth`` / ``videoHeight``).
No live Anam/Daily connection is made.
"""

import pytest
from anam import PersonaConfig, SessionOptions

from pipecat_anam import AnamTransport, AnamVideoService


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


# --- Paired-argument validation ------------------------------------------------------


@pytest.mark.parametrize("width,height", [(768, None), (None, 1152)])
def test_video_service_rejects_partial_dimensions(width, height) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        AnamVideoService(
            api_key="k",
            persona_config=_passthrough_persona(),
            video_width=width,
            video_height=height,
        )


@pytest.mark.parametrize("width,height", [(768, None), (None, 1152)])
def test_transport_rejects_partial_dimensions(width, height) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        AnamTransport(
            api_key="k",
            persona_config=_passthrough_persona(),
            daily_room_url="https://example.daily.co/test",
            video_width=width,
            video_height=height,
        )


# --- Dimensions stored on the service / transport client -----------------------------


def test_video_service_stores_dimensions() -> None:
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        video_width=768,
        video_height=1152,
    )
    assert service._video_width == 768
    assert service._video_height == 1152


def test_video_service_defaults_to_no_dimensions() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())
    assert service._video_width is None
    assert service._video_height is None


def test_transport_forwards_dimensions_to_client() -> None:
    transport = AnamTransport(
        api_key="k",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
        video_width=768,
        video_height=1152,
    )
    assert transport._client._video_width == 768
    assert transport._client._video_height == 1152


# --- SessionOptions serialization guard (mirrors what both services build) ------------


def test_session_options_serialize_portrait_dimensions() -> None:
    options = SessionOptions(enable_session_replay=False, video_width=768, video_height=1152)
    payload = options.to_dict()
    assert payload["videoWidth"] == 768
    assert payload["videoHeight"] == 1152


def test_session_options_omit_dimensions_when_unset() -> None:
    payload = SessionOptions(enable_session_replay=False).to_dict()
    assert "videoWidth" not in payload
    assert "videoHeight" not in payload
