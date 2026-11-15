"""Run the SegmentEveryForam browser interface on a GPU workstation."""

import math
import os
from functools import lru_cache
from pathlib import Path
from shutil import make_archive, rmtree
from tempfile import TemporaryDirectory, mkdtemp
from urllib.request import urlretrieve
from uuid import uuid4
from zipfile import is_zipfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


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

    import keras
    from sam2.build_sam import build_sam2

    directory = Path(os.environ.get("SEF_MODEL_DIR", Path.home() / ".cache" / "segmenteveryforam"))
    directory.mkdir(parents=True, exist_ok=True)
    unet_path = checkpoint(directory, "best_val_loss.keras", UNET_URL, is_zipfile)
    sam_path = checkpoint(directory, "sam2.1_hiera_large.pt", SAM_URL, is_zipfile)
    return (
        keras.saving.load_model(unet_path, compile=False),
        build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(sam_path), device="cuda"),
    )


def review_image(session):
    canvas = Image.fromarray(session["image"]).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    for index, polygon in enumerate(session["polygons"]):
        color = COLORS[index % len(COLORS)]
        draw.polygon(list(polygon.exterior.coords), fill=(*color, 65), outline=(*color, 255), width=2)
        for ring in polygon.interiors:
            draw.polygon(list(ring.coords), fill=(0, 0, 0, 0), outline=(*color, 255), width=2)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, polygon in enumerate(session["polygons"], start=1):
        point = polygon.representative_point()
        draw.text((point.x, point.y), str(index), anchor="mm", fill="white", stroke_width=2, stroke_fill="black")
    points = session["points"]
    if len(points) == 2:
        draw.line(points, fill="yellow", width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="yellow", outline="black")
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
    session = {"image": image, "name": path.name, "polygons": polygons, "points": [], "directory": None}
    preview = review_image(session)
    clear_session(previous)
    return session, preview, path.stem, None, None, None, (
        f"Detected {len(polygons)} specimens. Choose a review tool to edit masks or mark the scale bar."
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
            session["points"] = [(x, y)]
            status = "Click the other end of the scale bar."
        elif (x, y) == points[0]:
            raise gr.Error("Choose a different point for the other end of the scale bar.")
        else:
            points.append((x, y))
            status = f"Scale bar spans {math.dist(*points):.2f} pixels. Enter its printed length below."
    else:
        from shapely.geometry import Point, Polygon

        point = Point(x, y)
        hits = [i for i, polygon in enumerate(polygons) if polygon.covers(point)]
        if mode == REMOVE:
            if not hits:
                raise gr.Error("Click inside the mask you want to remove.")
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
            polygons.append(polygon)
        status = f"{len(polygons)} specimens in the current review."
    return session, review_image(session), None, status


def positive(value, label):
    if value is None or not math.isfinite(value) or value <= 0:
        raise gr.Error(f"Enter a positive {label}.")
    return value


def export(session, sample, species, depth, age, scale_um, pixels_per_meter):
    if not session or not session["polygons"]:
        raise gr.Error("Segment an image and keep at least one specimen before exporting.")
    if pixels_per_meter is None:
        if len(session["points"]) != 2:
            raise gr.Error("Mark both scale-bar endpoints or enter a known calibration in pixels per meter.")
        pixels_per_meter = math.dist(*session["points"]) / (positive(scale_um, "scale-bar length") * 1e-6)
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
        archive = directory / f"{uuid4().hex[:8]}" / "SegmentEveryForam_results.zip"
        archive.parent.mkdir()
        Path(pending).replace(archive)
    return session, str(archive), f"Exported {len(data)} specimens at {pixels_per_meter:,.2f} pixels per meter."


def build_app():
    with gr.Blocks(title="SegmentEveryForam", delete_cache=(3600, 7200)) as app:
        session = gr.State(time_to_live=7200, delete_callback=clear_session)
        gr.Markdown("# SegmentEveryForam\nSegment one microscope image and export calibrated specimen measurements.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Microscope image", file_types=[".png", ".jpeg", ".jpg", ".tif", ".tiff"])
                run = gr.Button("Run segmentation", variant="primary")
                mode = gr.Radio([REVIEW, ADD, REMOVE, CALIBRATE], value=REVIEW, label="Review tool")
                scale = gr.Number(label="Printed scale-bar length (µm)")
                known = gr.Number(label="Known calibration (pixels per meter, optional override)")
            with gr.Column(scale=3):
                preview = gr.Image(label="Specimen masks", interactive=False, height=625, format="png")
                status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            sample = gr.Textbox(label="Sample ID")
            species = gr.Textbox(label="Species")
            depth = gr.Number(label="Depth (m CCSF)")
            age = gr.Number(label="Age (Ma)")
        save = gr.Button("Export measurements, crops, and masks", variant="primary")
        download = gr.File(label="Results ZIP", interactive=False)
        run.click(segment, [upload, session], [session, preview, sample, scale, known, download, status], concurrency_id="workflow")
        preview.select(edit, [session, mode], [session, preview, download, status], concurrency_id="workflow")
        save.click(export, [session, sample, species, depth, age, scale, known], [session, download, status], concurrency_id="workflow")
    return app.queue()


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="50mb")
