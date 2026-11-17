from pathlib import Path

import pytest
import yaml


SCENE_FILE = (
    Path(__file__).parents[1]
    / "config"
    / "scenes"
    / "scene_003_large_shelf.yaml"
)


def test_scene_003_large_shelf_geometry():
    scene = yaml.safe_load(SCENE_FILE.read_text(encoding="utf-8"))
    shelf = scene["shelf"]

    assert scene["scene_id"] == 3
    assert shelf["type"] == "shelf"
    assert shelf["long_axis"] == "y"
    assert shelf["length"] == pytest.approx(1.0)
    assert shelf["width"] == pytest.approx(0.30)
    assert shelf["height"] == pytest.approx(0.80)

    assert shelf["board_length"] == pytest.approx(1.0)
    assert shelf["board_width"] == pytest.approx(0.30)
    assert shelf["board_thickness"] == pytest.approx(0.03)
    assert shelf["post_diameter"] == pytest.approx(0.04)
    assert shelf["board_center_heights"] == pytest.approx(
        [0.115, 0.445, 0.785]
    )
    board_top_heights = [
        center_height + shelf["board_thickness"] / 2.0
        for center_height in shelf["board_center_heights"]
    ]
    assert board_top_heights == pytest.approx([0.13, 0.46, 0.80])

    measured_front_right_x = 0.4052727780508177
    measured_front_right_y = -0.5844874407567195
    assert shelf["center_x"] - shelf["width"] / 2.0 == pytest.approx(
        measured_front_right_x
    )
    assert shelf["center_y"] - shelf["length"] / 2.0 == pytest.approx(
        measured_front_right_y
    )
    assert shelf["center_x"] + shelf["width"] / 2.0 == pytest.approx(
        0.7052727780508177
    )
    assert shelf["center_y"] + shelf["length"] / 2.0 == pytest.approx(
        0.41551255924328045
    )

    # Both Piper bases lie inside the shelf's Y span, so the nearest horizontal
    # distance from either base to the shelf is the front/min-X coordinate.
    assert measured_front_right_y <= -0.3 <= measured_front_right_y + 1.0
    assert measured_front_right_y <= 0.3 <= measured_front_right_y + 1.0
    assert measured_front_right_x == pytest.approx(0.4052727780508177)

    assert shelf["side_panels_enabled"] is True
    assert shelf["side_panel_height"] == pytest.approx(0.80)
