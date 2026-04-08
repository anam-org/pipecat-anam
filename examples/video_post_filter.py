#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agnostic video post-filter helpers for Pipecat examples.

This module center-crops packed RGB24 image bytes to a target aspect ratio.
It does not resize; output dimensions are derived from the input frame.
"""

from __future__ import annotations

from typing import Final

from pipecat.frames.frames import Frame, OutputImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

RGB24_BYTES_PER_PIXEL: Final[int] = 3


def center_crop_rgb24_bytes_to_aspect(
    image: bytes,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[bytes, tuple[int, int]]:
    """Center-crop packed RGB24 bytes to match ``target_size`` aspect ratio.

    This helper only crops. It never upscales or downscales.

    Args:
        image: Packed RGB24 image bytes in row-major order.
        source_size: Source frame size as ``(width, height)``.
        target_size: Target output size as ``(width, height)``.
            Only the aspect ratio is used.

    Returns:
        Cropped image bytes and cropped size as ``(width, height)``.
    """
    source_width, source_height = source_size
    target_width, target_height = target_size

    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"source_size must be positive, got {source_size!r}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target_size must be positive, got {target_size!r}")

    expected_len = source_width * source_height * RGB24_BYTES_PER_PIXEL
    if len(image) != expected_len:
        raise ValueError(
            f"image length does not match source_size for RGB24: got {len(image)}, "
            f"expected {expected_len}"
        )

    # Compare aspect ratios via cross-multiplication to avoid floating point drift:
    # source_width / source_height ? target_width / target_height
    lhs = source_width * target_height
    rhs = source_height * target_width
    if lhs == rhs:
        return image, source_size

    # For packed RGB24 without row padding, row stride equals width * bytes_per_pixel.
    stride_bytes = source_width * RGB24_BYTES_PER_PIXEL
    if lhs > rhs:
        # Source is wider than target -> crop left/right.
        cropped_width = int(round((source_height * target_width) / target_height))
        cropped_width = max(1, min(cropped_width, source_width))
        if cropped_width == source_width:
            return image, source_size

        x = (source_width - cropped_width) // 2
        cropped_row_bytes = cropped_width * RGB24_BYTES_PER_PIXEL
        cropped = bytearray(cropped_row_bytes * source_height)

        dst = 0
        for row in range(source_height):
            src_start = row * stride_bytes + x * RGB24_BYTES_PER_PIXEL
            src_end = src_start + cropped_row_bytes
            cropped[dst : dst + cropped_row_bytes] = image[src_start:src_end]
            dst += cropped_row_bytes

        return bytes(cropped), (cropped_width, source_height)

    # Source is taller (or narrower) than target -> crop top/bottom.
    cropped_height = int(round((source_width * target_height) / target_width))
    cropped_height = max(1, min(cropped_height, source_height))
    if cropped_height == source_height:
        return image, source_size

    y = (source_height - cropped_height) // 2
    src_start = y * stride_bytes
    src_end = src_start + cropped_height * stride_bytes
    return image[src_start:src_end], (source_width, cropped_height)


def center_crop_output_image_frame(
    frame: OutputImageRawFrame, target_size: tuple[int, int]
) -> OutputImageRawFrame:
    """Center-crop an output image frame to the target aspect ratio.

    The function assumes packed RGB24 frames. If the source aspect ratio already
    matches the target aspect ratio, this is a no-op and returns ``frame``.
    """
    if frame.format != "RGB":
        raise ValueError(
            f"CenterAspectCropFilter expects RGB24 OutputImageRawFrame, got {frame.format!r}"
        )

    cropped_image, cropped_size = center_crop_rgb24_bytes_to_aspect(
        frame.image,
        source_size=frame.size,
        target_size=target_size,
    )
    if cropped_size == frame.size:
        return frame

    cropped_frame = OutputImageRawFrame(
        image=cropped_image,
        size=cropped_size,
        format=frame.format,
    )
    cropped_frame.pts = frame.pts
    cropped_frame.metadata = dict(frame.metadata)
    cropped_frame.transport_source = frame.transport_source
    cropped_frame.transport_destination = frame.transport_destination
    # Newer Pipecat versions include sync_with_audio on OutputImageRawFrame.
    # Older versions (e.g. 0.0.103) do not. Keep this helper compatible with both.
    if hasattr(frame, "sync_with_audio"):
        setattr(cropped_frame, "sync_with_audio", getattr(frame, "sync_with_audio"))
    return cropped_frame


class CenterAspectCropFilter(FrameProcessor):
    """A simple output-image filter that center-crops to a target aspect ratio."""

    def __init__(self, target_size: tuple[int, int]):
        super().__init__()
        self._target_size = target_size

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            frame = center_crop_output_image_frame(frame, self._target_size)
        await self.push_frame(frame, direction)

