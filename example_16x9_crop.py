#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Anam
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: crop Anam's 3:2 output to 16:9 on your Pipecat server."""

from __future__ import annotations

from typing import Final

from pipecat.frames.frames import Frame, OutputImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments

from example import bot as base_bot
from example import build_transport_params

DEFAULT_VIDEO_OUT_SIZE: Final[tuple[int, int]] = (720, 405)
_PACKED_PIXEL_SIZES: Final[dict[str, int]] = {
    "RGB": 3,
    "BGR": 3,
    "RGBA": 4,
    "BGRA": 4,
}


def crop_top_bottom_to_aspect_ratio(
    frame: OutputImageRawFrame,
    target_size: tuple[int, int] = DEFAULT_VIDEO_OUT_SIZE,
) -> OutputImageRawFrame:
    """Crop the top and bottom off a packed image frame to match the target ratio.

    This helper is intentionally crop-only. It uses ``target_size`` to derive the
    desired aspect ratio, preserves the source width, and returns the vertically
    centered crop. For the usual Anam 720x480 output, this produces 720x405.
    """

    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target_size must be positive, got {target_size!r}")

    source_width, source_height = frame.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"frame size must be positive, got {frame.size!r}")

    if frame.format not in _PACKED_PIXEL_SIZES:
        raise ValueError(
            "TopBottom16x9CropProcessor supports packed RGB/BGR/RGBA/BGRA frames only, "
            f"got format {frame.format!r}"
        )

    target_ratio = target_width / target_height
    crop_height = int(round(source_width / target_ratio))
    if crop_height >= source_height:
        return frame

    bytes_per_pixel = _PACKED_PIXEL_SIZES[frame.format]
    row_bytes = source_width * bytes_per_pixel
    crop_y = max((source_height - crop_height) // 2, 0)
    source_start = crop_y * row_bytes
    source_end = source_start + crop_height * row_bytes

    cropped_frame = OutputImageRawFrame(
        image=frame.image[source_start:source_end],
        size=(source_width, crop_height),
        format=frame.format,
    )
    cropped_frame.pts = frame.pts
    cropped_frame.metadata = dict(frame.metadata)
    cropped_frame.transport_source = frame.transport_source
    cropped_frame.transport_destination = frame.transport_destination
    cropped_frame.sync_with_audio = frame.sync_with_audio
    return cropped_frame


class TopBottom16x9CropProcessor(FrameProcessor):
    """Crop Anam output to 16:9 before it reaches the Pipecat transport."""

    def __init__(self, target_size: tuple[int, int] = DEFAULT_VIDEO_OUT_SIZE):
        super().__init__()
        self._target_size = target_size

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, OutputImageRawFrame):
            await self.push_frame(frame, direction)
            return

        cropped_frame = crop_top_bottom_to_aspect_ratio(frame, self._target_size)
        await self.push_frame(cropped_frame, direction)


transport_params = build_transport_params(*DEFAULT_VIDEO_OUT_SIZE)


async def bot(runner_args: RunnerArguments):
    await base_bot(
        runner_args,
        custom_transport_params=transport_params,
        post_anam_processors=[TopBottom16x9CropProcessor()],
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
