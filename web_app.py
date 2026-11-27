"""Run the SegmentEveryForam browser interface on a GPU workstation."""

import json
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
# Example images, with their scale bars and (optionally) stored masks, so
# visitors can try the app without their own image.
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
HIGHLIGHT = (255, 214, 0)
# Columns shown in the measurement table: get_foram_morphometrics column,
# heading, and whether the value is an area, a length or a ratio.
TABLE_COLUMNS = [
    ("area_mm2", "Area", "area"),
    ("perimeter_mm", "Perimeter", "length"),
    ("major_diameter_mm", "Major axis", "length"),
    ("minor_diameter_mm", "Minor axis", "length"),
    ("equivalent_diameter_mm", "Eq. diameter", "length"),
    ("aspect_ratio", "Aspect ratio", "ratio"),
    ("circularity", "Circularity", "ratio"),
    ("roundness", "Roundness", "ratio"),
]


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
    selected = session.get("selected")
    for index, polygon in enumerate(session["polygons"]):
        color = COLORS[index % len(COLORS)]
        if index == selected:
            outline, fill, width = (*HIGHLIGHT, 255), (*HIGHLIGHT, 90), 5 * unit
        else:
            outline, fill, width = (*color, 255), (*color, 65), 2 * unit
        draw.polygon(list(polygon.exterior.coords), fill=fill, outline=outline, width=width)
        for ring in polygon.interiors:
            draw.polygon(list(ring.coords), fill=(0, 0, 0, 0), outline=outline, width=width)
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


def valid_number(value):
    return value is not None and math.isfinite(value) and value > 0


def current_calibration(session, scale_um, pixels_per_meter):
    """Return pixels per meter from the known value or the scale bar, or None."""
    if valid_number(pixels_per_meter):
        return pixels_per_meter
    points = session["points"] if session else []
    if len(points) == 2 and valid_number(scale_um):
        return math.dist(*points) / (scale_um * 1e-6)
    return None


def pixel_measurements(session):
    """Measure each mask in pixels, reusing results for masks measured before."""
    import segmenteveryforam as sef

    cache = session.setdefault("measured", {})
    rows = []
    for polygon in session["polygons"]:
        key = polygon.wkb
        if key not in cache:
            grains = sef.polygons_to_grains([polygon], image=session["image"])
            cache[key] = grains[0].measure().to_dict() if grains else {}
        rows.append(cache[key])
    return rows


def measurement_table(session, scale_um=None, pixels_per_meter=None):
    """Build the measurement table, in millimetres once the scale is known."""
    import pandas as pd

    ppm = current_calibration(session, scale_um, pixels_per_meter)
    length, area = ("mm", "mm²") if ppm else ("px", "px²")
    units = {"area": f" ({area})", "length": f" ({length})", "ratio": ""}
    headings = ["#"] + [name + units[kind] for _, name, kind in TABLE_COLUMNS]
    if not session or not session["polygons"]:
        return gr.Dataframe(value=pd.DataFrame(columns=headings), label="Measurements")

    import segmenteveryforam as sef

    summary = pd.DataFrame(pixel_measurements(session)).reindex(
        columns=["area", "perimeter", "major_axis_length", "minor_axis_length",
                 "orientation", "centroid-0", "centroid-1"])
    # get_foram_morphometrics expects meters and reports millimetres. Without a
    # calibration, 1000 px per meter makes its millimetre columns pixels.
    meters_per_px = 1 / (ppm or 1000)
    summary["area"] *= meters_per_px ** 2
    for column in ["perimeter", "major_axis_length", "minor_axis_length"]:
        summary[column] *= meters_per_px
    data = sef.get_foram_morphometrics(summary)
    table = pd.DataFrame({"#": range(1, len(data) + 1)})
    formats = {}
    for (column, _, kind), heading in zip(TABLE_COLUMNS, headings[1:]):
        if kind == "ratio":
            decimals = 3
        elif ppm:
            decimals = 5 if kind == "area" else 4
        else:
            decimals = 1
        table[heading] = data[column].to_numpy(dtype=float).round(decimals)
        formats[heading] = f"{{:.{decimals}f}}"
    selected = session.get("selected")
    styled = table.style.format(formats, na_rep="–").apply(
        lambda row: [f"background-color: rgba{(*HIGHLIGHT, 0.35)}" if row.name == selected else ""] * len(row),
        axis=1)
    if ppm:
        label = f"Measurements ({ppm * 1e-6:.4g} px per µm)"
    else:
        label = ("Measurements in pixels. Set the scale bar length or a known calibration "
                 "to see millimetres.")
    return gr.Dataframe(value=styled, label=label)


def clear_session(session):
    if session and session.get("directory"):
        rmtree(session["directory"], ignore_errors=True)


def example_info():
    """Read examples/examples.json: file name -> title and scale bar."""
    try:
        return json.loads((EXAMPLES_DIR / "examples.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_masks(path):
    """Load polygons from a GeoJSON file written by save_masks."""
    from shapely.geometry import shape

    features = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    return [shape(feature["geometry"]) for feature in features]


def save_masks(polygons, path):
    """Save polygons as a GeoJSON FeatureCollection in image pixel coordinates."""
    from shapely.geometry import mapping

    collection = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"specimen": index}, "geometry": mapping(polygon)}
        for index, polygon in enumerate(polygons, start=1)]}
    Path(path).write_text(json.dumps(collection), encoding="utf-8")


def segment(path, previous, polygons=None, points=(), scale_um=None):
    if not path:
        raise gr.Error("Upload a microscope image first.")
    path = Path(path)
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise gr.Error("Choose a PNG, JPEG, or TIFF image.")
    import segmenteveryforam as sef

    image = sef.load_image(str(path))
    if polygons is None:
        unet, sam = models()
        polygons, _, _ = sef.predict_with_box_prompts(image, unet, sam)
    session = {"image": image, "name": path.name, "polygons": list(polygons),
               "points": [tuple(point) for point in points], "history": [], "selected": None,
               "directory": None}
    preview = review_image(session)
    clear_session(previous)
    if scale_um:
        status = (f"Detected {len(polygons)} specimens. The {scale_um:g} µm scale bar is already marked, "
                  "so the table is in millimetres. Click a mask to find it in the table, "
                  "or choose a review tool to edit masks.")
    else:
        status = (f"Detected {len(polygons)} specimens. Click a mask to find it in the table, "
                  "or choose a review tool to edit masks or mark the scale bar.")
    return (session, preview, path.stem, scale_um, None, None, status,
            measurement_table(session, scale_um))


def run_example(path, previous):
    """Open an example image with its scale bar, using stored masks if there are any."""
    path = Path(path)
    info = example_info().get(path.name, {})
    stored = EXAMPLES_DIR / f"{path.stem}.geojson"
    polygons = load_masks(stored) if stored.exists() else None
    return segment(path, previous, polygons, info.get("scale_bar_px", ()), info.get("scale_bar_um"))


UNDO_LIMIT = 50


def remember(session):
    """Save the masks and scale-bar points before an edit so it can be undone."""
    history = session.setdefault("history", [])
    history.append((list(session["polygons"]), list(session["points"])))
    del history[:-UNDO_LIMIT]
    # Mask numbers can change, so drop the highlight.
    session["selected"] = None


def undo(session, scale_um=None, pixels_per_meter=None):
    if not session or not session.get("history"):
        raise gr.Error("There is nothing to undo.")
    session["polygons"], session["points"] = session["history"].pop()
    session["selected"] = None
    remaining = len(session["history"])
    return session, review_image(session), None, (
        f"Undid the last edit. {len(session['polygons'])} specimens in the current review"
        + (f"; {remaining} more edit{'s' if remaining != 1 else ''} can be undone." if remaining else ".")
    ), measurement_table(session, scale_um, pixels_per_meter)


def highlight(session, index, scale_um, pixels_per_meter):
    """Highlight one specimen in the image and the table (None clears it)."""
    session["selected"] = index
    if index is None:
        status = "No specimen selected."
    else:
        status = f"Specimen {index + 1} is highlighted in the image and the table."
    return session, review_image(session), status, measurement_table(session, scale_um, pixels_per_meter)


def select_row(session, scale_um, pixels_per_meter, event: gr.SelectData):
    if not session or not session["polygons"]:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    row = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
    if row is None or not 0 <= row < len(session["polygons"]):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    return highlight(session, row, scale_um, pixels_per_meter)


def check_scale_length(session, scale_um):
    """Warn when the printed length typed in equals the bar's length in pixels."""
    points = session["points"] if session else []
    if len(points) == 2 and valid_number(scale_um) and abs(scale_um - math.dist(*points)) < 1:
        gr.Warning(
            f"{scale_um:g} µm is the scale bar's length in pixels. Enter the length printed on "
            "the bar instead (for example 500). If the bar really is labelled "
            f"{scale_um:g} µm, ignore this message.", duration=15)


def refresh_table(session, scale_um, pixels_per_meter):
    check_scale_length(session, scale_um)
    return measurement_table(session, scale_um, pixels_per_meter)


def edit(session, mode, scale_um=None, pixels_per_meter=None, event: gr.SelectData = None):
    skip = (gr.skip(),) * 5
    if not session:
        return skip
    x, y = event.index
    image = session["image"]
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return skip
    polygons = session["polygons"]
    if mode == REVIEW:
        from shapely.geometry import Point

        hits = [i for i, polygon in enumerate(polygons) if polygon.covers(Point(x, y))]
        session, preview, status, table = highlight(
            session, hits[-1] if hits else None, scale_um, pixels_per_meter)
        return session, preview, gr.skip(), status, table
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
            check_scale_length(session, scale_um)
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
    return session, review_image(session), None, status, measurement_table(session, scale_um, pixels_per_meter)


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
        check_scale_length(session, scale_um)
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
                examples = {name: info for name, info in example_info().items()
                            if (EXAMPLES_DIR / name).exists()}
                if examples:
                    # A hidden image component makes the examples show as
                    # thumbnails; choosing one opens it with its scale bar.
                    example_image = gr.Image(type="filepath", visible=False)
                    example_picker = gr.Examples(
                        [[str(EXAMPLES_DIR / name)] for name in examples], [example_image],
                        label="Or try an example (click a picture)", examples_per_page=len(examples))
                mode = gr.Radio([REVIEW, ADD, REMOVE, CALIBRATE], value=REVIEW, label="Review tool")
                undo_button = gr.Button("Undo last edit")
                scale = gr.Number(label="Printed scale-bar length (µm)", value=None)
                known = gr.Number(label="Known calibration (pixels per meter, optional override)", value=None)
            with gr.Column(scale=3):
                preview = gr.Image(label="Specimen masks", interactive=False, height=625, format="png")
                status = gr.Textbox(label="Status", interactive=False)
                table = gr.Dataframe(label="Measurements", interactive=False, max_height=400)
        with gr.Row():
            sample = gr.Textbox(label="Sample ID")
            species = gr.Textbox(label="Species")
            depth = gr.Number(label="Depth (m CCSF)", value=None)
            age = gr.Number(label="Age (Ma)", value=None)
        crops = gr.Checkbox(value=True, label="Include specimen crops and masks (ZIP). Untick to export only the measurements CSV.")
        save = gr.Button("Export", variant="primary")
        download = gr.File(label="Results", interactive=False)
        run.click(segment, [upload, session], [session, preview, sample, scale, known, download, status, table],
                  concurrency_id="workflow")
        if examples:
            example_picker.load_input_event.then(
                lambda path: path, [example_image], [upload]).then(
                run_example, [example_image, session],
                [session, preview, sample, scale, known, download, status, table], concurrency_id="workflow")
        preview.select(edit, [session, mode, scale, known], [session, preview, download, status, table],
                       concurrency_id="workflow")
        undo_button.click(undo, [session, scale, known], [session, preview, download, status, table],
                          concurrency_id="workflow")
        table.select(select_row, [session, scale, known], [session, preview, status, table],
                     concurrency_id="workflow")
        for number in (scale, known):
            number.input(refresh_table, [session, scale, known], [table], concurrency_id="workflow")
        save.click(export, [session, sample, species, depth, age, scale, known, crops], [session, download, status], concurrency_id="workflow")
        # Gradio's number boxes show an empty initial value as 0, which users
        # could export as a real depth or age. Clearing them on load shows
        # them as blank.
        app.load(lambda: (None, None, None, None), outputs=[scale, known, depth, age])
    return app.queue()


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, max_file_size="50mb")
