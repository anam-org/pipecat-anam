#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for the ``show_ai_avatar_disclosure`` option on both
:class:`AnamVideoService` and :class:`AnamTransport`.

These assert the value is stored and forwarded into the Anam SDK's ``SessionOptions``
(which serializes it to ``showAIAvatarDisclosure``). No live Anam/Daily connection is made.
"""

from anam import PersonaConfig, SessionOptions

from pipecat_anam import AnamTransport, AnamVideoService


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


# --- Value stored on the service / transport client ----------------------------------


def test_video_service_stores_disclosure_flag() -> None:
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        show_ai_avatar_disclosure=False,
    )
    assert service._show_ai_avatar_disclosure is False


def test_video_service_defaults_to_no_disclosure_override() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())
    assert service._show_ai_avatar_disclosure is None


def test_transport_forwards_disclosure_flag_to_client() -> None:
    transport = AnamTransport(
        api_key="k",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
        show_ai_avatar_disclosure=False,
    )
    assert transport._client._show_ai_avatar_disclosure is False


# --- SessionOptions serialization guard (mirrors what both services build) ------------


def test_session_options_serialize_disclosure_flag() -> None:
    options = SessionOptions(enable_session_replay=False, show_ai_avatar_disclosure=False)
    payload = options.to_dict()
    assert payload["showAIAvatarDisclosure"] is False


def test_session_options_omit_disclosure_flag_when_unset() -> None:
    payload = SessionOptions(enable_session_replay=False).to_dict()
    assert "showAIAvatarDisclosure" not in payload
