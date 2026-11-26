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

### Added

- The `ForamPlot` editor labels the scale line with its length in pixels
  while drawing and, when the printed length is known, with the resulting
  calibration (for example `257.3 px = 500 µm`). `set_scale_length()`
  sets the printed length after drawing.
- The Colab notebook's export can skip specimen crops and masks
  (`export_crops_and_masks = False`) and then downloads only the
  measurements CSV.
- The browser prototype has the same option: untick **Include specimen crops
  and masks** to export only the measurements CSV.
- `load_unet()` loads a U-Net checkpoint so that it gives the same results on
  every Keras backend. On PyTorch, plain `keras.saving.load_model` shifts the
  U-Net's upsampling layers by one pixel, which changed 3–4% of predicted
  pixels and most of the boundary class. The Colab notebook and the browser
  prototype now use `load_unet()`.
- `tools/compare_keras_backends.py` compares U-Net predictions between the
  TensorFlow and PyTorch backends, and `tools/compare_workflow_backends.py`
  compares the full U-Net and SAM 2.1 segmentation step.

### Changed

- The Colab notebook checks for a GPU before installing, requests a T4 GPU,
  and restarts the session automatically after installation so the browser
  editor loads without a manual restart.
- The Colab export prints the scale-bar length and calibration in µm. The
  browser prototype's export message does the same.
- The Colab notebook and the browser prototype run Keras on PyTorch, which
  gives the same specimen outlines as TensorFlow in about 12 s per image
  instead of 22 s on a Colab T4 GPU, and uses about 1.2 GB less GPU memory.
  The browser prototype keeps any `KERAS_BACKEND` already set.
- The local workflow notebook also runs Keras on PyTorch and loads the U-Net
  with `load_unet()`. On a CPU this gives the same results in about the same
  time as TensorFlow.
- The browser prototype now uses Gradio 6, the version Colab includes, so
  installing it in Colab no longer downgrades Gradio or prints dependency
  errors.
- `requirements.txt` now matches the package's dependencies and no longer
  installs the original `segmenteverygrain` package.
- `environment_macos.yml` now creates a `segmenteveryforam` environment and
  installs SegmentEveryForam from the repository. The README explains when to
  use it and how to install with pip.

### Removed

- The Segmenteverygrain documentation site sources in `docs/`, the example
  images used only by those pages, and the workflow that built them. The
  README images in `docs/source/_static/images/` are kept.

### Fixed

- The Colab notebook's browser-editor check no longer shows a traceback with
  current Matplotlib versions.
- Browser prototype: specimen numbers and scale-bar markers are sized to the
  image, so they are legible on large microscope images.
- Browser prototype: Depth, Age and the calibration fields start blank
  instead of showing 0, and blank fields are exported as missing values.

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
