"""Export individual specimen images and their aligned segmentation masks."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine


def export_specimens(
    image, grains, foram_data, current_image, foram_data_path,
    *, output_dir="foram_specimen_outputs", padding_px=12,
):
    """Write PNG pairs and a manifest, then return the export directory.

    Grains must be in the same order as the rows of foram_data. Manifest
    bounds are zero-based pixels with exclusive right and bottom edges.
    """
    if len(grains) != len(foram_data) or not foram_data["foram_id"].is_unique:
        raise ValueError("Regenerate morphometrics from the finalized grains before exporting crops.")

    source_image = Path(current_image["image_path"])
    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    export_dir = output_root / f"{source_image.stem}_{run_id}_{uuid4().hex[:8]}"
    height, width = image.shape[:2]
    manifest_rows = []

    with TemporaryDirectory(dir=output_root) as temporary_dir:
        staging = Path(temporary_dir) / "specimens"
        staging.mkdir()

        for grain, foram_id in zip(grains, foram_data["foram_id"]):
            polygon = grain.polygon
            left, top, right, bottom = polygon.bounds
            x0 = max(0, int(np.floor(left)) - padding_px)
            y0 = max(0, int(np.floor(top)) - padding_px)
            x1 = min(width, int(np.ceil(right)) + padding_px + 1)
            y1 = min(height, int(np.ceil(bottom)) + padding_px + 1)
            if x0 >= x1 or y0 >= y1:
                raise ValueError(f"Foram {foram_id} has no crop inside the source image.")

            # Grain coordinates use pixel centers at integer x and y.
            mask = rasterize(
                [(polygon, 255)],
                out_shape=(y1 - y0, x1 - x0),
                transform=Affine.translation(x0 - 0.5, y0 - 0.5),
                dtype="uint8",
            )
            if not mask.any():
                raise ValueError(f"Foram {foram_id} has an empty rasterized mask.")

            crop_name = f"foram_{foram_id:04d}.png"
            mask_name = f"foram_{foram_id:04d}_mask.png"
            Image.fromarray(image[y0:y1, x0:x1]).save(staging / crop_name)
            Image.fromarray(mask).save(staging / mask_name)
            manifest_rows.append({
                "foram_id": foram_id,
                "sample_id": current_image["sample_id"],
                "image_id": current_image["image_id"],
                "source_image": str(source_image.resolve()),
                "morphometrics_csv": str(Path(foram_data_path).resolve()),
                "crop_file": crop_name,
                "mask_file": mask_name,
                "x_min": x0,
                "y_min": y0,
                "x_max_exclusive": x1,
                "y_max_exclusive": y1,
            })

        pd.DataFrame(manifest_rows, columns=[
            "foram_id", "sample_id", "image_id", "source_image", "morphometrics_csv",
            "crop_file", "mask_file", "x_min", "y_min", "x_max_exclusive", "y_max_exclusive",
        ]).to_csv(staging / "manifest.csv", index=False)
        staging.rename(export_dir)
    return export_dir
