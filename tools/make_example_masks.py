"""Segment the web app's example images and store their masks.

Run this on a machine with a CUDA GPU (for example a Colab GPU runtime) from
the repository root:

    python tools/make_example_masks.py

For each image listed in examples/examples.json, it writes
examples/<image name>.geojson with the masks found by the U-Net and SAM. The
web app then shows these masks immediately when someone opens the example,
without running the models. Run it again after the models change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web_app  # noqa: E402  (sets KERAS_BACKEND before Keras is imported)


def main():
    import segmenteveryforam as sef

    unet, sam = web_app.models()
    for name in web_app.example_info():
        image_path = web_app.EXAMPLES_DIR / name
        if not image_path.exists():
            print(f"Skipping {name}: the image is missing.")
            continue
        image = sef.load_image(str(image_path))
        polygons, _, _ = sef.predict_with_box_prompts(image, unet, sam)
        output = image_path.with_suffix(".geojson")
        web_app.save_masks(polygons, output)
        print(f"{name}: saved {len(polygons)} masks to {output.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
