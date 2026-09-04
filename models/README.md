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

Foraminifera-specific U-Net models are being developed for SegmentEveryForam.

These models are intended to improve the initial semantic segmentation of foraminifera before SAM-based instance segmentation.

Foraminifera-specific model files and documentation will be added as the models are prepared for distribution.

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