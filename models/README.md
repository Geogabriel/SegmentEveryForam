# SegmentEveryForam Models

This directory contains U-Net models used by the SegmentEveryForam segmentation workflow.

## Foraminifera-specific models

SegmentEveryForam includes U-Net models fine-tuned specifically for foraminiferal segmentation.

### `seg_model_foram_v1_30epochs.keras`

First SegmentEveryForam U-Net model fine-tuned for foraminiferal segmentation.

Training details:

- training images: 51 image-mask pairs
- training patches: 7,140
- patch size: 256 × 256 pixels
- stride: 128 pixels
- training epochs: 30
- segmentation classes:
  - 0: background
  - 1: foraminiferal test interior
  - 2: external boundary

This model was trained using multiple foraminiferal images and serves as the first foraminifera-specific U-Net model developed for SegmentEveryForam.

### `best_val_loss.keras`

Default workflow checkpoint. Fine-tuned from the v1 model on *Globigerinoides ruber* images and selected by lowest validation loss during the 100-epoch run. The dataset was split by original microscope image before patch extraction.

## SAM 2.1

SegmentEveryForam also uses SAM 2.1 for instance segmentation.

SAM 2.1 is developed by Meta and its model checkpoints are not developed or distributed as SegmentEveryForam U-Net models. Refer to the official SAM 2 repository for information about SAM checkpoints and licensing.

## Model provenance

Model provenance should be preserved when adding new models to this directory. Documentation for each SegmentEveryForam model should include, where applicable:

- model name and version
- target foraminiferal taxon or taxa
- number of training images
- training and validation strategy
- image type
- training epochs
- segmentation classes
- evaluation metrics
- known limitations