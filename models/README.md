# SegmentEveryForam Models

This directory contains U-Net models used by the SegmentEveryForam segmentation workflow.

## Inherited Segmenteverygrain models

The following model files originate from the Segmenteverygrain project:

- `seg_model.keras`
- `seg_model_smooth_labels.keras`

These models were developed for general grain segmentation and are retained in SegmentEveryForam for compatibility, testing, and development.

For information about their original development and training, see the Segmenteverygrain project:

https://github.com/zsylvester/segmenteverygrain

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

### `seg_model_foram_v2_G_ruber_30epochs.keras`

Second SegmentEveryForam U-Net model, fine-tuned specifically for *Globigerinoides ruber* segmentation.

Training details:

- training images: 40
- training patches: 5,600
- patch size: 256 × 256 pixels
- stride: 128 pixels
- training epochs: 30
- target taxon: *Globigerinoides ruber*
- segmentation classes:
  - 0: background
  - 1: foraminiferal test interior
  - 2: external boundary

This model was developed to improve segmentation performance for *G. ruber* relative to the first foraminifera-specific model.

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