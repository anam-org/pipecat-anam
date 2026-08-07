#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Unit tests for the ``region`` / ``region_policy`` options on both
:class:`AnamVideoService` and :class:`AnamTransport`.

These assert the strict-without-region validation and that the values are forwarded into
the Anam SDK's ``SessionOptions`` (which serializes them to ``region`` / ``regionPolicy``).
No live Anam/Daily connection is made.
"""

import pytest
from anam import PersonaConfig, SessionOptions
from loguru import logger

from pipecat_anam import AnamTransport, AnamVideoService


def _passthrough_persona() -> PersonaConfig:
    return PersonaConfig(avatar_id="avatar-1", enable_audio_passthrough=True)


# --- strict-without-region validation ------------------------------------------------


def test_video_service_rejects_strict_without_region() -> None:
    with pytest.raises(ValueError, match="requires region to be set"):
        AnamVideoService(
            api_key="k",
            persona_config=_passthrough_persona(),
            region_policy="strict",
        )


def test_transport_rejects_strict_without_region() -> None:
    with pytest.raises(ValueError, match="requires region to be set"):
        AnamTransport(
            api_key="k",
            persona_config=_passthrough_persona(),
            daily_room_url="https://example.daily.co/test",
            region_policy="strict",
        )


# --- unknown policy value ------------------------------------------------------------
# Both are typed ``Literal["preferred", "strict"]``, but env-var wiring (as in the
# examples) passes arbitrary strings through, so the runtime check has to hold.


def test_video_service_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match='must be either "preferred" or "strict"'):
        AnamVideoService(
            api_key="k",
            persona_config=_passthrough_persona(),
            region="eu",
            region_policy="Strict",  # type: ignore[arg-type]
        )


def test_transport_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match='must be either "preferred" or "strict"'):
        AnamTransport(
            api_key="k",
            persona_config=_passthrough_persona(),
            daily_room_url="https://example.daily.co/test",
            region="eu",
            region_policy="Strict",  # type: ignore[arg-type]
        )


# --- Values stored on the service / transport client ---------------------------------


def test_video_service_stores_region() -> None:
    service = AnamVideoService(
        api_key="k",
        persona_config=_passthrough_persona(),
        region="eu",
        region_policy="strict",
    )
    assert service._region == "eu"
    assert service._region_policy == "strict"


def test_video_service_defaults_to_no_region() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona())
    assert service._region is None
    assert service._region_policy is None


def test_transport_forwards_region_to_client() -> None:
    transport = AnamTransport(
        api_key="k",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
        region="eu",
        region_policy="preferred",
    )
    assert transport._client._region == "eu"
    assert transport._client._region_policy == "preferred"


def test_transport_allows_region_without_policy() -> None:
    transport = AnamTransport(
        api_key="k",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
        region="us",
    )
    assert transport._client._region == "us"
    assert transport._client._region_policy is None


# --- SessionOptions serialization guard (mirrors what both services build) ------------


def test_session_options_serialize_region() -> None:
    options = SessionOptions(enable_session_replay=False, region="eu", region_policy="strict")
    payload = options.to_dict()
    assert payload["region"] == "eu"
    assert payload["regionPolicy"] == "strict"


def test_session_options_omit_region_when_unset() -> None:
    payload = SessionOptions(enable_session_replay=False).to_dict()
    assert "region" not in payload
    assert "regionPolicy" not in payload


# --- cross-region fallback is surfaced -----------------------------------------------


def test_video_service_warns_when_served_by_another_region() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona(), region="eu")
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m), level="WARNING")
    try:
        service._log_served_region("us")
    finally:
        logger.remove(sink)
    assert any("requested region 'eu'" in r and "'us'" in r for r in records)


def test_video_service_does_not_warn_when_served_by_requested_region() -> None:
    service = AnamVideoService(api_key="k", persona_config=_passthrough_persona(), region="eu")
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m), level="WARNING")
    try:
        service._log_served_region("eu")
    finally:
        logger.remove(sink)
    assert records == []


def test_transport_warns_when_served_by_another_region() -> None:
    transport = AnamTransport(
        api_key="k",
        persona_config=_passthrough_persona(),
        daily_room_url="https://example.daily.co/test",
        region="eu",
    )
    records: list[str] = []
    sink = logger.add(lambda m: records.append(m), level="WARNING")
    try:
        transport._client._log_served_region("us")
    finally:
        logger.remove(sink)
    assert any("requested region 'eu'" in r and "'us'" in r for r in records)
