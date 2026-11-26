# Changelog

All notable changes to SegmentEveryForam are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/).

SegmentEveryForam was created from
[Segmenteverygrain](https://github.com/zsylvester/segmenteverygrain) at commit
`1ab65cc` (2026-08-14, after Segmenteverygrain 0.5.0). For changes before that
point, see the
[Segmenteverygrain changelog](https://github.com/zsylvester/segmenteverygrain/blob/main/CHANGELOG.md).

## [Unreleased]

### Changed

- `requirements.txt` now matches the package's dependencies and no longer
  installs the original `segmenteverygrain` package.
- `environment_macos.yml` now creates a `segmenteveryforam` environment and
  installs SegmentEveryForam from the repository. The README explains when to
  use it and how to install with pip.

### Removed

- The Segmenteverygrain documentation site sources in `docs/`, the example
  images used only by those pages, and the workflow that built them. The
  README images in `docs/source/_static/images/` are kept.

## [0.1.0] - 2026-11-26

First release of SegmentEveryForam.

### Added

- Two U-Net checkpoints fine-tuned for foraminifera:
  `seg_model_foram_v1_30epochs.keras` (51 image–mask pairs, 30 epochs) and the
  default `best_val_loss.keras` (fine-tuned from v1 on 40 *Globigerinoides
  ruber* pairs, split by image, lowest validation loss over 100 epochs). The
  training notebooks are in `notebooks/`.
- `notebooks/Evaluate_saved_UNet_checkpoints.ipynb`, which compares the saved
  checkpoints on the same reference images, and its results in
  `models/evaluation/`. On three *G. ruber* reference images, the default
  checkpoint scores a boundary IoU of 0.804 against 0.794 for v1.
- Automatic SAM 2.1 box prompts from the U-Net prediction, returning one
  editable polygon per prompt.
- Foraminifera morphometrics: area, perimeter, major, minor and equivalent
  diameters, aspect ratio, elongation, roundness, circularity,
  perimeter-to-area ratio, orientation and centroid, with sample- and
  species-level summaries and plotting functions.
- Export of individual specimen crops and masks (`specimen_export` module),
  with an optional corrected training mask.
- A Google Colab notebook that runs the full workflow with interactive
  `ForamPlot` editing.
- A Gradio browser prototype (`web_app.py`) for upload, review, scale
  calibration and export on a Colab GPU or a GPU workstation.
- The `A` key in `ForamPlot` adds or merges selected touching segments.
- A Windows installation guide (`INSTALLATION.md`).
- `NOTICE` and `CITATION.cff`.

### Changed

- The package is now `segmenteveryforam`, and the interactive interface is
  `ForamPlot`. `GrainPlot` remains as an alias for code written for
  Segmenteverygrain.
- A drawn scale line is reused when its length is entered at export.

### Fixed

- Species-level morphometric aggregation.
- NumPy scalar serialization in the U-Net evaluation export.

### Removed

- Segmenteverygrain's general-grain checkpoints, example datasets and Colab
  notebook. They remain available upstream.
- The Segmenteverygrain JOSS paper sources (`paper/`) and the workflow that
  built them. The paper is cited in the README and `CITATION.cff`.

[0.1.0]: https://github.com/Geogabriel/SegmentEveryForam/releases/tag/v0.1.0
