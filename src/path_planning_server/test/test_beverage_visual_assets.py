import re
from pathlib import Path

import pytest
import yaml


ASSET_ROOT = Path(__file__).resolve().parents[1] / "meshes" / "beverages"
TEXTURE_SUFFIXES = {".jpg", ".png", ".exr"}
INIT_FROM_PATTERN = re.compile(r"<init_from>([^<]+)</init_from>")


def texture_references(dae_file):
    references = set()
    with dae_file.open(encoding="utf-8") as stream:
        for line in stream:
            for value in INIT_FROM_PATTERN.findall(line):
                reference = Path(value.strip())
                if reference.suffix.lower() in TEXTURE_SUFFIXES:
                    references.add(reference)
    return references


def test_all_beverage_visual_assets_are_packaged_and_resolvable():
    if not ASSET_ROOT.exists():
        pytest.skip("Optional beverage mesh assets are not installed")
    target_directories = sorted(
        path for path in ASSET_ROOT.iterdir() if path.is_dir()
    )
    assert len(target_directories) == 16

    texture_files = []
    for target_directory in target_directories:
        metadata_file = target_directory / "metadata.yaml"
        metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8"))
        visual_file = target_directory / metadata["visual_mesh"]
        collision_file = target_directory / metadata["collision_mesh"]

        assert metadata["unit"] == "meter"
        assert metadata["coordinate_system"] == "Z-up"
        assert metadata["origin"] == "bottom_center"
        assert visual_file.is_file() and visual_file.stat().st_size > 0
        assert collision_file.is_file() and collision_file.stat().st_size > 0

        with visual_file.open(encoding="utf-8") as stream:
            header = "".join(stream.readline() for _ in range(8))
        assert '<unit name="meter" meter="1.0"' in header
        assert "<up_axis>Z_UP</up_axis>" in header

        references = texture_references(visual_file)
        assert references
        for reference in references:
            assert not reference.is_absolute()
            texture_file = target_directory / reference
            assert texture_file.is_file() and texture_file.stat().st_size > 0

        texture_files.extend(
            path
            for path in (target_directory / "textures").rglob("*")
            if path.is_file() and path.suffix.lower() in TEXTURE_SUFFIXES
        )

    assert len(texture_files) == 57
