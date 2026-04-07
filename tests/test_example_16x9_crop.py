from pathlib import Path
import sys

from pipecat.frames.frames import OutputImageRawFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example_16x9_crop import crop_top_bottom_to_aspect_ratio


def _build_row_marked_rgb_frame(width: int, height: int) -> OutputImageRawFrame:
    rows = []
    for row in range(height):
        rows.append(bytes([row % 256, 0, 0]) * width)
    return OutputImageRawFrame(image=b"".join(rows), size=(width, height), format="RGB")


def test_crop_top_bottom_to_aspect_ratio_crops_center_rows():
    frame = _build_row_marked_rgb_frame(720, 480)

    cropped = crop_top_bottom_to_aspect_ratio(frame, target_size=(720, 405))

    assert cropped.size == (720, 405)

    first_row_marker = cropped.image[0]
    last_row_marker = cropped.image[-3]

    assert first_row_marker == 37
    assert last_row_marker == 185


def test_crop_top_bottom_to_aspect_ratio_passes_through_matching_aspect_ratio():
    frame = _build_row_marked_rgb_frame(720, 405)

    cropped = crop_top_bottom_to_aspect_ratio(frame, target_size=(720, 405))

    assert cropped is frame
