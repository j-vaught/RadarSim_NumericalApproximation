# RadarSim_NumericalApproximation

Analytical/statistical synthetic X-band marine radar data generation using numerical approximation methods.

## Overview

Fast radar data generation using physics-based statistical models rather than computationally expensive ray tracing. Designed for generating large-scale training datasets for ML-based target detection.

## Features

- **Sea Clutter**: K-distribution and Weibull statistical models (sea states 0-7)
- **Target Models**: Swerling cases 0-4 with RCS fluctuation
- **Propagation**: Two-ray multipath, analytical radar equation
- **Land Generation**: Synthetic coastlines with radar shadows and occlusion
- **Output Formats**: Polar CSV, Cartesian PNG, YOLO labels
- **Quantization**: 4-bit (16 levels) matching real radar ADC

## Installation

```bash
pip install numpy scipy pillow pyyaml
```

## Quick Start

```python
from pathlib import Path
from pipeline import Tier2Pipeline, Tier2Config
from models import LandConfig, create_target_ensemble

config = Tier2Config(
    num_frames=100,
    seed=42,
    sea_state=3,
    output_dir=Path('./output'),
    land_enabled=True,
    land_config=LandConfig(
        coastline_range=0.5,
        land_start_az=20.0,
        land_end_az=140.0,
    ),
)

pipeline = Tier2Pipeline(config)

targets = create_target_ensemble(
    num_targets=25,
    range_limits=(100, 800),
    azimuth_limits=(150, 10),  # Water sector
    rcs_range=(5, 200),
)

pipeline.run(targets=targets)
```

## Configuration

### Radar Parameters
| Parameter | Value |
|-----------|-------|
| Frequency | 9.41 GHz (X-band) |
| Peak Power | 25 W |
| Range Resolution | 1.07 m |
| Max Range | ~0.5 NM |
| Azimuth Samples | 720/revolution |
| Range Bins | 868 |

### Environment Parameters
| Parameter | Range |
|-----------|-------|
| Sea State | 0 (glassy) to 7 (high) |
| Rain Rate | mm/hr |
| Land | Configurable coastline with bay features |

## Output Formats

| Format | Description |
|--------|-------------|
| CSV | Polar radar data (720 x 868) |
| PNG | Cartesian grayscale images |
| TXT | YOLO format bounding boxes |

## Statistical Models

### Sea Clutter (K-Distribution)
Compound Gaussian model with gamma-distributed texture and exponential speckle. Parameters calibrated per sea state.

### Swerling Target Fluctuation
| Case | Fluctuation | Distribution |
|------|-------------|--------------|
| 0 | None | Constant |
| 1 | Scan-to-scan | Exponential |
| 2 | Pulse-to-pulse | Exponential |
| 3 | Scan-to-scan | Chi-squared (4 DOF) |
| 4 | Pulse-to-pulse | Chi-squared (4 DOF) |

### Synthetic Land
- Fractal coastline generation
- Configurable bays and peninsulas
- Radar shadow/occlusion modeling
- Smooth azimuthal termination

## Module Structure

```
├── pipeline.py          # Main simulation pipeline + CLI
├── config/
│   └── default.yaml     # Example YAML configuration
├── models/              # Statistical physics models
│   ├── clutter.py       # K-distribution, Weibull sea clutter
│   ├── targets.py       # Swerling I-IV RCS fluctuation
│   ├── land.py          # Synthetic coastline generation
│   └── propagation.py   # Radar equation, 2-ray multipath
├── radar/               # Radar system configuration
│   ├── config.py        # RadarConfig dataclass
│   └── waveform.py      # LFM chirp generation
├── core/                # Core data structures
│   ├── csv_handler.py   # FURUNO CSV I/O
│   ├── converter.py     # Polar <-> Cartesian conversion
│   └── ...
├── scene/               # Scene generation utilities
└── output/              # Output pipeline
```

## Citation

```bibtex
@article{vaught2025multifidelity,
  title={A Multi-Fidelity Simulation Framework for X-Band Pulse Compression Radar},
  author={Vaught, J.C.},
  year={2025}
}
```

## License

MIT License
