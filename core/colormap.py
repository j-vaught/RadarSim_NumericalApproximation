"""Colormap Application

Provides scientific colormaps for radar intensity visualization.
"""

from enum import Enum
from typing import Tuple

import numpy as np


class Colormap(Enum):
    VIRIDIS = "viridis"
    TURBO = "turbo"
    MAGMA = "magma"
    GRAYSCALE = "grayscale"
    HOT = "hot"


# Build 256-entry LUTs for matplotlib-based colormaps at import time
_LUT = {}


def _build_luts():
    try:
        import matplotlib.cm as cm
        for name in ("viridis", "turbo", "magma"):
            cmap = cm.get_cmap(name)
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                r, g, b, _ = cmap(i / 255.0)
                lut[i] = (int(r * 255), int(g * 255), int(b * 255))
            _LUT[name] = lut
    except ImportError:
        pass

    # Hot colormap: black -> red -> yellow -> white (piecewise)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.33:
            s = t / 0.33
            lut[i] = (int(s * 255), 0, 0)
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            lut[i] = (255, int(s * 255), 0)
        else:
            s = (t - 0.66) / 0.34
            lut[i] = (255, 255, int(s * 255))
    _LUT["hot"] = lut

    # Grayscale
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = (i, i, i)
    _LUT["grayscale"] = lut


_build_luts()


def apply_colormap(intensity: int, colormap: Colormap = Colormap.VIRIDIS) -> Tuple[int, int, int]:
    lut = _LUT.get(colormap.value)
    if lut is not None:
        r, g, b = lut[intensity]
        return (int(r), int(g), int(b))
    return (intensity, intensity, intensity)


def apply_colormap_to_image(grayscale: np.ndarray, colormap: Colormap = Colormap.VIRIDIS) -> np.ndarray:
    flat = grayscale.ravel()
    lut = _LUT.get(colormap.value)
    if lut is None:
        lut = _LUT["grayscale"]

    rgba = np.zeros((flat.size, 4), dtype=np.uint8)
    colors = lut[flat]
    rgba[:, 0] = colors[:, 0]
    rgba[:, 1] = colors[:, 1]
    rgba[:, 2] = colors[:, 2]
    rgba[:, 3] = np.where(flat > 0, 255, 0).astype(np.uint8)

    return rgba.reshape(grayscale.shape + (4,))
