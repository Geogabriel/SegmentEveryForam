"""
segmenteveryforam: A Python package for automated foraminifera segmentation
and image-based analysis using U-Net and SAM.

This package provides tools for:
- Semantic segmentation of foraminifera using U-Net
- Instance segmentation using the Segment Anything Model (SAM)
- Interactive editing of foraminiferal segmentations
- Extraction of individual foraminifera images for analysis
"""

# Core segmentation functionality
from .segmenteveryforam import *

# Interactive editing tools
from .interactions import (
    ForamPlot,
    GrainPlot,
    Grain,
    load_image,
    polygons_to_grains,
    save_grains,
    save_summary,
    save_histogram,
    save_mask,
    get_summary,
)

# Foraminifera extraction utilities
from .grain_utils import (
    extract_grain_image,
    extract_all_grains,
    make_square,
    extract_vgg16_features,
    cluster_grains,
    create_grain_panel_cluster,
    create_grain_panel,
    create_clustered_grain_montage,
    ClusterMontageSelector,
    ClusterMontageLabeler,
    plot_classified_grains,
    extract_color_features,
)

# Foraminifera morphometrics analysis
from .morphometrics import (
    get_foram_morphometrics,
    summarize_foram_morphometrics,
)

__version__ = "0.1.0"
