"""Land / background loading for the Tier 2 pipeline.

Centralises every way a land mask or background can be loaded:
  - parametric coastline (LandGenerator)
  - annotation.json → polar land mask
  - PNG water mask → polar land mask
  - real radar CSV frames (land_frames/)
  - annotation.json → polar water mask (for object placement)
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .core import (
    RadarCSVHandler,
    RadarConverter,
    BinaryMask,
    PolarMask,
    AnnotationSet,
    create_water_mask,
)
from .models import LandGenerator, LandConfig
from .radar import RadarConfig


@dataclass
class LandLoadResult:
    """Everything run_scene() needs after land loading."""
    land_mask: Optional[np.ndarray] = None          # bool, True = land
    land_frames: Optional[List[np.ndarray]] = None  # real CSV backgrounds
    water_mask: Optional[PolarMask] = None           # bool, True = water
    land_generator: Optional[LandGenerator] = None
    full_depth: bool = False


class LandLoader:
    """Static helpers that turn scene-YAML land config into polar masks / frames."""

    @staticmethod
    def load(land_config, scene_path: Path,
             converter: RadarConverter, radar: RadarConfig) -> LandLoadResult:
        """Dispatch to the right loader based on land_config.type.

        Parameters
        ----------
        land_config : LandSceneConfig
            The ``environment.land`` block from the scene YAML.
        scene_path : Path
            Path to the scene YAML (used for resolving relative paths).
        converter : RadarConverter
            Polar ↔ Cartesian converter.
        radar : RadarConfig
            Radar geometry (num_range_bins, samples_per_revolution, …).

        Returns
        -------
        LandLoadResult
        """
        result = LandLoadResult()
        land_type = land_config.type

        if land_type == "parametric":
            # Parametric land is handled by Tier2Config.__init__ — nothing to do here.
            pass

        elif land_type == "annotation":
            land_mask = LandLoader.load_annotation_land_mask(
                land_config.annotation_path, scene_path, converter)
            land_cfg = LandConfig(intensity=land_config.intensity)
            gen = LandGenerator(land_cfg)
            gen.set_azimuth_bounds_from_mask(land_mask)
            result.land_mask = land_mask
            result.land_generator = gen
            result.full_depth = True

        elif land_type == "mask":
            land_mask = LandLoader.load_png_land_mask(
                land_config.mask_path, scene_path, converter)
            land_cfg = LandConfig(intensity=land_config.intensity)
            gen = LandGenerator(land_cfg)
            gen.set_azimuth_bounds_from_mask(land_mask)
            result.land_mask = land_mask
            result.land_generator = gen
            result.full_depth = True

        elif land_type == "frames":
            result.land_frames = LandLoader.load_land_frames(
                land_config.land_frames_dir, scene_path, radar)

        # Build the water mask used for object placement
        num_pulses = radar.samples_per_revolution
        num_bins = radar.num_range_bins

        if result.land_frames is not None and land_config.annotation_path:
            water_polar = LandLoader.load_annotation_water_mask(
                land_config.annotation_path, scene_path, converter)
            print(f"Water mask from annotation: "
                  f"{np.sum(water_polar.data)}/{water_polar.data.size} water pixels "
                  f"({100 * np.sum(water_polar.data) / water_polar.data.size:.1f}%)")
            result.water_mask = water_polar

        elif result.land_frames is not None:
            result.water_mask = LandLoader.build_water_mask_from_land_frames(
                result.land_frames, radar)

        elif result.land_mask is not None:
            water_polar = PolarMask(num_pulses, fill=True)
            water_polar.data = ~result.land_mask
            result.water_mask = water_polar

        else:
            water_polar = PolarMask(num_pulses, fill=True)
            water_polar.data = np.ones((num_pulses, num_bins), dtype=bool)
            result.water_mask = water_polar

        return result

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------

    @staticmethod
    def load_annotation_water_mask(annotation_path: str, scene_path: Path,
                                   converter: RadarConverter) -> PolarMask:
        """Load annotation.json and return a polar water mask (True = water)."""
        ann_path = Path(annotation_path)
        if not ann_path.is_absolute():
            ann_path = (scene_path.parent / ann_path).resolve()

        annotations = AnnotationSet.load(str(ann_path))
        img_size = converter.config.image_size
        water_mask_cart = create_water_mask(annotations, img_size, img_size)
        return PolarMask.from_cartesian(water_mask_cart, converter)

    @staticmethod
    def load_annotation_land_mask(annotation_path: str, scene_path: Path,
                                  converter: RadarConverter) -> np.ndarray:
        """Load annotation.json and convert to a polar land mask (True = land)."""
        ann_path = Path(annotation_path)
        if not ann_path.is_absolute():
            ann_path = (scene_path.parent / ann_path).resolve()

        annotations = AnnotationSet.load(str(ann_path))
        img_size = converter.config.image_size
        water_mask_cart = create_water_mask(annotations, img_size, img_size)
        polar_water = PolarMask.from_cartesian(water_mask_cart, converter)
        return ~polar_water.data

    @staticmethod
    def load_png_land_mask(mask_path: str, scene_path: Path,
                           converter: RadarConverter) -> np.ndarray:
        """Load a PNG water mask and convert to a polar land mask (True = land)."""
        from PIL import Image

        mp = Path(mask_path)
        if not mp.is_absolute():
            mp = (scene_path.parent / mp).resolve()

        img = np.array(Image.open(str(mp)).convert('L'))
        water_cart = img > 127

        img_size = converter.config.image_size
        if water_cart.shape[0] != img_size or water_cart.shape[1] != img_size:
            resized = Image.fromarray(water_cart.astype(np.uint8) * 255).resize(
                (img_size, img_size), Image.NEAREST
            )
            water_cart = np.array(resized) > 127

        cart_mask = BinaryMask(img_size, img_size)
        cart_mask.data = water_cart
        polar_water = PolarMask.from_cartesian(cart_mask, converter)
        return ~polar_water.data

    @staticmethod
    def load_land_frames(land_frames_dir: str, scene_path: Path,
                         radar: RadarConfig) -> List[np.ndarray]:
        """Load real radar CSV land frames as uint8 arrays."""
        frames_dir = Path(land_frames_dir)
        if not frames_dir.is_absolute():
            frames_dir = (scene_path.parent / frames_dir).resolve()

        csv_files = sorted(frames_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {frames_dir}")

        land_frames = []
        for csv_path in csv_files:
            frame = RadarCSVHandler.read_csv_uncached(str(csv_path))
            arr = frame.to_regularized_array(radar.samples_per_revolution)
            land_frames.append(arr)

        print(f"Loaded {len(land_frames)} land frames from {frames_dir}")
        return land_frames

    @staticmethod
    def build_water_mask_from_land_frames(land_frames: List[np.ndarray],
                                          radar: RadarConfig) -> PolarMask:
        """Build a water mask by finding pixels that are zero across all frames."""
        stacked = np.stack(land_frames, axis=0)
        max_vals = stacked.max(axis=0)
        water_data = max_vals < 5
        mask = PolarMask(radar.samples_per_revolution, fill=False)
        mask.data = water_data
        return mask
