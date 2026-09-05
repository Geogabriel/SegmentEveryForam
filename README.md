# SegmentEveryForam

**SegmentEveryForam** is a Python package for automated segmentation, interactive correction, and image-based morphometric analysis of foraminifera.

The package is adapted from [Segmenteverygrain](https://github.com/zsylvester/segmenteverygrain), developed by Zoltán Sylvester and collaborators. Segmenteverygrain combines a U-Net-style convolutional neural network with the Segment Anything Model (SAM) to identify and segment individual objects in images.

SegmentEveryForam extends this framework toward foraminiferal image analysis while retaining the core segmentation architecture of Segmenteverygrain.

> **Development status:** SegmentEveryForam is currently under active development.

## Overview

SegmentEveryForam uses a two-stage segmentation workflow:

1. A U-Net model generates an initial semantic segmentation of the image.
2. The initial segmentation is used to generate prompts for SAM 2.1, which produces individual object masks.

The resulting segmentations can then be inspected and manually corrected using the interactive `ForamPlot` interface.

The general workflow is:

```text
Microscope image
      |
      v
U-Net segmentation
      |
      v
SAM prompt generation
      |
      v
SAM 2.1 instance segmentation
      |
      v
Interactive correction
      |
      v
Foraminifera masks and measurements
```
## Segmentation examples

<p align="center">
  <img src="docs/source/_static/images/g_bulloides_original.jpeg" width="48%">
  <img src="docs/source/_static/images/g_bulloides_segmented.png" width="48%">
</p>

<p align="center">
  <em>Globigerina bulloides</em>: original microscope image (left) and SegmentEveryForam segmentation output (right).
</p>

<p align="center">
  <img src="docs/source/_static/images/g_ruber_original.jpeg" width="48%">
  <img src="docs/source/_static/images/g_ruber_segmented.png" width="48%">
</p>

<p align="center">
  <em>Globigerinoides ruber</em>: original microscope image (left) and SegmentEveryForam segmentation output (right).
</p>

<p align="center">
  <img src="docs/source/_static/images/orbulina_original.png" width="48%">
  <img src="docs/source/_static/images/orbulina_segmented.png" width="48%">
</p>

<p align="center">
  <em>Orbulina universa</em>: original microscope image (left) and SegmentEveryForam segmentation output (right).
</p>

## Current features

SegmentEveryForam currently provides tools for:

- U-Net-based semantic segmentation
- SAM 2.1-based instance segmentation
- Interactive correction of segmentation results
- Manual creation of missed foraminifera using SAM prompts
- Deletion of incorrectly segmented objects
- Addition and merging of touching foram segments
- Scale calibration
- Extraction of segmentation masks and object measurements

Additional foraminifera-specific functionality is under development.

## Requirements

Python 3.10 or higher is recommended.

Major dependencies include:

- TensorFlow
- Keras
- PyTorch
- SAM 2
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- OpenCV
- Shapely
- Rasterio

## Installation

Clone the SegmentEveryForam repository:

```bash
git clone https://github.com/Geogabriel/SegmentEveryForam.git
cd SegmentEveryForam
```

SegmentEveryForam is currently under development, so installation instructions and environment files may continue to change as the package is developed.

For development use, install the package from the repository in editable mode:

```bash
pip install -e .
```

The package can then be imported using:

```python
import segmenteveryforam as sef
```

The interactive module can be imported using:

```python
import segmenteveryforam.interactions as sfi
```

## Models

SegmentEveryForam combines a U-Net segmentation model with SAM 2.1.

The repository currently retains the original Segmenteverygrain U-Net model files:

```text
models/
├── seg_model.keras
└── seg_model_smooth_labels.keras
```

These models originate from the Segmenteverygrain project and were trained for general grain segmentation. They are retained for compatibility and development purposes.

Foraminifera-specific U-Net models are being developed separately for SegmentEveryForam.

SAM 2.1 model checkpoints are developed and distributed by Meta and are not SegmentEveryForam models.

See `models/README.md` for additional information about model provenance and usage.

## Interactive editing

SegmentEveryForam provides the `ForamPlot` interactive interface for inspecting and correcting segmentation results.

A typical interface can be created with:

```python
plot = sfi.ForamPlot(
    forams,
    image=image,
    predictor=predictor
)

plot.activate()
```

Current interactive controls include:

- **Left click on an existing foram:** Select or unselect the segmentation
- **Left click in an unsegmented area:** Create a foram using a SAM foreground prompt
- **Alt + Left click:** Add a foreground SAM prompt
- **Alt + Right click:** Add a background SAM prompt
- **C:** Create a foram from existing SAM prompts
- **D:** Delete selected foram segmentations
- **A:** Add or merge selected touching foram segments
- **M:** Legacy shortcut for merging selected segments
- **Z:** Remove the most recently created segmentation
- **H:** Toggle the coverage mask
- **Esc:** Clear selections and prompts
- **Ctrl (hold):** Temporarily hide selected masks
- **Shift + drag:** Draw a scale bar for unit conversion

`GrainPlot` remains available as a backward-compatible alias for code originally written for Segmenteverygrain.

## Morphometric analysis

Segmented objects can be measured using image-based morphometric properties such as:

- area
- major axis length
- minor axis length
- perimeter
- orientation
- centroid

Additional derived measurements and foraminifera-specific analytical workflows are under development.

## Relationship to Segmenteverygrain

SegmentEveryForam is a derivative of the open-source Segmenteverygrain project.

The original Segmenteverygrain software provides the foundation for much of the segmentation workflow, including U-Net-based segmentation, SAM integration, image-processing utilities, and interactive segmentation functionality.

SegmentEveryForam is intended to extend this framework for applications involving foraminifera, including microscopy-based segmentation and morphometric analysis.

The original Segmenteverygrain project is available at:

https://github.com/zsylvester/segmenteverygrain

## Acknowledgements

SegmentEveryForam builds upon work by the developers and contributors of Segmenteverygrain.

The original `interactions` module was developed by Dave Matthews for Segmenteverygrain. The Segmenteverygrain project was developed by Zoltán Sylvester and collaborators and benefited from contributions and discussions involving Danny Stockli, Nick Howes, Kalinda Roberts, Jake Covault, Matt Malkowski, Raymond Luong, Wilson Bai, Rowan Martindale, Sergey Fomel, and others.

See the original Segmenteverygrain repository and publication for complete project acknowledgements.

## Citation

Because SegmentEveryForam is derived from Segmenteverygrain, users should cite the original Segmenteverygrain publication when using functionality derived from that software:

Sylvester, Z., Stockli, D. F., Howes, N., Roberts, K., Malkowski, M. A., Poros, Z., Martindale, R. C., & Bai, W. (2025). Segmenteverygrain: A Python module for segmentation of grains in images. *Journal of Open Source Software*, 10(112), 7953. https://doi.org/10.21105/joss.07953

```bibtex
@article{Sylvester2025,
  doi = {10.21105/joss.07953},
  url = {https://doi.org/10.21105/joss.07953},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {112},
  pages = {7953},
  author = {Sylvester, Zoltán and Stockli, Daniel F. and Howes, Nick and Roberts, Kalinda and Malkowski, Matthew A. and Poros, Zsófia and Martindale, Rowan C. and Bai, Wilson},
  title = {Segmenteverygrain: A Python module for segmentation of grains in images},
  journal = {Journal of Open Source Software}
}
```

A dedicated citation for SegmentEveryForam will be added if and when the software receives its own archived release or publication.

## License

SegmentEveryForam is distributed under the Apache License 2.0, consistent with the license of the original Segmenteverygrain project.

The original Segmenteverygrain source code is copyright its respective authors and contributors. Modifications and additions made for SegmentEveryForam retain the applicable Apache 2.0 licensing requirements.

See `LICENSE.txt` for the full license text.