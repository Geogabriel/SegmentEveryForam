"""Run the SegmentEveryForam browser interface on a GPU workstation."""

import math
import os
import re
from functools import lru_cache
from pathlib import Path
from shutil import make_archive, rmtree
from tempfile import TemporaryDirectory, mkdtemp
from urllib.request import urlretrieve
from uuid import uuid4
from zipfile import is_zipfile

# Run Keras on PyTorch unless KERAS_BACKEND is already set. PyTorch gives the
# same results as TensorFlow with load_unet and is about twice as fast. This
# must happen before Keras is first imported.
os.environ.setdefault("KERAS_BACKEND", "torch")

import gradio as gr  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


UNET_URL = (
    "https://raw.githubusercontent.com/Geogabriel/SegmentEveryForam/"
    "1f311172938d2265de0b2d54f05d1244f619057b/models/best_val_loss.keras"
)
SAM_URL = (
    "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/"
    "sam2.1_hiera_large.pt"
)
COLORS = [(74, 222, 167), (255, 172, 80), (115, 173, 255), (234, 151, 226)]
REVIEW = "Inspect masks"
ADD = "Add a missed specimen"
REMOVE = "Remove a mask"
CALIBRATE = "Mark scale-bar endpoints"


def checkpoint(directory, name, url, valid):
    path = directory / name
    if not valid(path):
        with TemporaryDirectory(dir=directory) as temp:
            pending = Path(temp) / name
            urlretrieve(url, pending)
            if not valid(pending):
                raise gr.Error(f"The download for {name} is incomplete or invalid.")
            pending.replace(path)
    return path


@lru_cache(maxsize=1)
def models():
    import torch

    if not torch.cuda.is_available():
        raise gr.Error("Run the app in an environment where PyTorch can access a CUDA GPU.")

    from sam2.build_sam import build_sam2

    import segmenteveryforam as sef

    directory = Path(os.environ.get("SEF_MODEL_DIR", Path.home() / ".cache" / "segmenteveryforam"))
    directory.mkdir(parents=True, exist_ok=True)
    unet_path = checkpoint(directory, "best_val_loss.keras", UNET_URL, is_zipfile)
    sam_path = checkpoint(directory, "sam2.1_hiera_large.pt", SAM_URL, is_zipfile)
    return (
        sef.load_unet(unet_path),
        build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(sam_path), device="cuda"),
    )


def label_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only a small bitmap font.
        return ImageFont.load_default()


def review_image(session):
    canvas = Image.fromarray(session["image"]).convert("RGBA")
    # Scale labels and markers with the image so they stay legible when the
    # preview is shrunk to fit the page.
    unit = max(1, round(max(canvas.size) / 1000))
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    for index, polygon in enumerate(session["polygons"]):
        color = COLORS[index % len(COLORS)]
        draw.polygon(list(polygon.exterior.coords), fill=(*color, 65), outline=(*color, 255), width=2 * unit)
        for ring in polygon.interiors:
            draw.polygon(list(ring.coords), fill=(0, 0, 0, 0), outline=(*color, 255), width=2 * unit)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = label_font(16 * unit)
    for index, polygon in enumerate(session["polygons"], start=1):
        point = polygon.representative_point()
        draw.text((point.x, point.y), str(index), font=font, anchor="mm", fill="white",
                  stroke_width=2 * unit, stroke_fill="black")
    points = session["points"]
    if len(points) == 2:
        draw.line(points, fill="yellow", width=3 * unit)
    radius = 5 * unit
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="black", width=unit)
    return canvas


def clear_session(session):
    if session and session.get("directory"):
        rmtree(session["directory"], ignore_errors=True)


def segment(path, previous):
    if not path:
        raise gr.Error("Upload a microscope image first.")
    path = Path(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise gr.Error("Choose a PNG, JPEG, or TIFF image.")
    import segmenteveryforam as sef

    image = sef.load_image(str(path))
    unet, sam = models()
    polygons, _, _ = sef.predict_with_box_prompts(image, unet, sam)
    session = {"image": image, "name": path.name, "polygons": polygons, "points": [], "history": [],
               "directory": None}
    preview = review_image(session)
    clear_session(previous)
    return session, preview, path.stem, None, None, None, (
        f"Detected {len(polygons)} specimens. Choose a review tool to edit masks or mark the scale bar."
    )


UNDO_LIMIT = 50


def remember(session):
    """Save the masks and scale-bar points before an edit so it can be undone."""
    history = session.setdefault("history", [])
    history.append((list(session["polygons"]), list(session["points"])))
    del history[:-UNDO_LIMIT]


def undo(session):
    if not session or not session.get("history"):
        raise gr.Error("There is nothing to undo.")
    session["polygons"], session["points"] = session["history"].pop()
    remaining = len(session["history"])
    return session, review_image(session), None, (
        f"Undid the last edit. {len(session['polygons'])} specimens in the current review"
        + (f"; {remaining} more edit{'s' if remaining != 1 else ''} can be undone." if remaining else ".")
    )


def edit(session, mode, event: gr.SelectData):
    if not session or mode == REVIEW:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    x, y = event.index
    image = session["image"]
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    polygons = session["polygons"]
    if mode == CALIBRATE:
        points = session["points"]
        if len(points) != 1:
            remember(session)
            session["points"] = [(x, y)]
            status = "Click the other end of the scale bar."
        elif (x, y) == points[0]:
            raise gr.Error("Choose a different point for the other end of the scale bar.")
        else:
            remember(session)
            points.append((x, y))
            status = (f"Scale bar spans {math.dist(*points):.1f} pixels. "
                      "Enter its printed length in Printed scale-bar length (µm).")
    else:
        from shapely.geometry import Point, Polygon

        point = Point(x, y)
        hits = [i for i, polygon in enumerate(polygons) if polygon.covers(point)]
        if mode == REMOVE:
            if not hits:
                raise gr.Error("Click inside the mask you want to remove.")
            remember(session)
            polygons.pop(hits[-1])
        else:
            if hits:
                raise gr.Error("Click an unmasked specimen to add it.")
            import segmenteveryforam as sef
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            predictor = SAM2ImagePredictor(models()[1])
            predictor.set_image(image)
            sx, sy, _ = sef.one_point_prompt(x, y, image, predictor)
            if len(sx) < 3:
                raise gr.Error("SAM did not find a specimen at that point.")
            polygon = Polygon(zip(sx, sy)).buffer(0)
            if polygon.geom_type == "MultiPolygon":
                polygon = max(polygon.geoms, key=lambda part: part.area)
            if polygon.geom_type != "Polygon" or polygon.is_empty or not polygon.covers(point):
                raise gr.Error("SAM did not find a specimen containing that point.")
            remember(session)
            polygons.append(polygon)
        status = f"{len(polygons)} specimens in the current review."
    return session, review_image(session), None, status


def positive(value, label):
    if value is None or not math.isfinite(value) or value <= 0:
        raise gr.Error(f"Enter a positive {label}.")
    return value


def export(session, sample, species, depth, age, scale_um, pixels_per_meter, crops_and_masks=True):
    if not session or not session["polygons"]:
        raise gr.Error("Segment an image and keep at least one specimen before exporting.")
    if pixels_per_meter is None:
        if len(session["points"]) != 2:
            raise gr.Error("Mark both scale-bar endpoints or enter a known calibration in pixels per meter.")
        bar_px = math.dist(*session["points"])
        scale_um = positive(scale_um, "scale-bar length")
        pixels_per_meter = bar_px / (scale_um * 1e-6)
        scale_note = f"Scale bar: {bar_px:.1f} px = {scale_um:g} µm. "
    else:
        scale_note = "Using the known calibration. "
    pixels_per_meter = positive(pixels_per_meter, "calibration in pixels per meter")

    import pandas as pd
    import segmenteveryforam as sef
    from segmenteveryforam.specimen_export import export_specimens

    image = session["image"]
    name = session["name"]
    image_id = Path(name).stem
    metadata = dict(image_path=name, image_id=image_id, sample_id=sample.strip() or image_id,
                    species=species.strip(), depth_ccsf=depth, Age_Ma=age)
    grains = sef.polygons_to_grains(session["polygons"], image=image)
    for grain in grains:
        grain.measure()
    data = sef.get_foram_morphometrics(sef.get_summary(grains, pixels_per_meter), metadata=metadata)
    if not session["directory"]:
        session["directory"] = mkdtemp(prefix="sef-web-")
    directory = Path(session["directory"])
    destination = directory / f"{uuid4().hex[:8]}"
    destination.mkdir()
    if not crops_and_masks:
        safe_name = re.sub(r"[^\w.-]+", "_", metadata["sample_id"]).strip("._") or "sample"
        download = destination / f"{safe_name}_foram_data.csv"
        data.to_csv(download, index=False)
        contents = "measurements"
    else:
        with TemporaryDirectory(dir=directory) as temp:
            result = Path(temp) / "results"
            result.mkdir()
            csv = result / "foram_data.csv"
            data.to_csv(csv, index=False)
            output = export_specimens(image, grains, data, metadata, csv, output_dir=result / "specimens")
            manifest_file = output / "manifest.csv"
            manifest = pd.read_csv(manifest_file)
            manifest["source_image"] = name
            manifest["morphometrics_csv"] = csv.name
            manifest.to_csv(manifest_file, index=False)
            pending = make_archive(str(Path(temp) / "pending"), "zip", root_dir=result)
            download = destination / "SegmentEveryForam_results.zip"
            Path(pending).replace(download)
        contents = "measurements, crops, and masks"
    return session, str(download), (
        f"Exported {contents} for {len(data)} specimens. {scale_note}"
        f"Calibration: {pixels_per_meter * 1e-6:.4g} px per µm."
    )


def build_app():
    with gr.Blocks(title="SegmentEveryForam", delete_cache=(3600, 7200)) as app:
        session = gr.State(time_to_live=7200, delete_callback=clear_session)
        gr.Markdown("# SegmentEveryForam\nSegment one microscope image and export calibrated specimen measurements.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Microscope image", file_types=[".png", ".jpeg", ".jpg", ".tif", ".tiff"])
                run = gr.Button("Run segmentation", variant="primary")
                mode = gr.Radio([REVIEW, ADD, REMOVE, CALIBRATE], value=REVIEW, label="Review tool")
                undo_button = gr.Button("Undo last edit")
                scale = gr.Number(label="Printed scale-bar length (µm)", value=None)
                known = gr.Number(label="Known calibration (pixels per meter, optional override)", value=None)
            with gr.Column(scale=3):
                preview = gr.Image(label="Specimen masks", interactive=False, height=625, format="png")
                status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            sample = gr.Textbox(label="Sample ID")
            species = gr.Textbox(label="Species")
            depth = gr.Number(label="Depth (m CCSF)", value=None)
            age = gr.Number(label="Age (Ma)", value=None)
        crops = gr.Checkbox(value=True, label="Include specimen crops and masks (ZIP). Untick to export only the measurements CSV.")
        save = gr.Button("Export", variant="primary")
        download = gr.File(label="Results", interactive=False)
        run.click(segment, [upload, session], [session, preview, sample, scale, known, download, status], concurrency_id="workflow")
        preview.select(edit, [session, mode], [session, preview, download, status], concurrency_id="workflow")
        undo_button.click(undo, [session], [session, preview, download, status], concurrency_id="workflow")
        save.click(export, [session, sample, species, depth, age, scale, known, crops], [session, download, status], concurrency_id="workflow")
        # Gradio's number boxes show an empty initial value as 0, which users
        # could export as a real depth or age. Clearing them on load shows
        # them as blank.
        app.load(lambda: (None, None, None, None), outputs=[scale, known, depth, age])
    return app.queue()


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="50mb")
