"""Compare U-Net predictions from Keras's TensorFlow and PyTorch backends.

Runs the saved checkpoints on the example images once with each backend,
using the package's own tiling code (``predict_image``), and reports how
closely the class probabilities and class maps agree.

Usage, from the repository root::

    python tools/compare_keras_backends.py

Each backend runs in its own process, because Keras picks its backend when it
is first imported.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["models/best_val_loss.keras", "models/seg_model_foram_v1_30epochs.keras"]
IMAGES = [
    "docs/source/_static/images/g_ruber_original.jpeg",
    "docs/source/_static/images/g_bulloides_original.jpeg",
]
CLASSES = ["background", "interior", "boundary"]


def load_predict_image():
    """Load predict_image without importing the package (and SAM 2)."""
    source = (ROOT / "segmenteveryforam" / "segmenteveryforam.py").read_text()
    wanted = {"predict_image_tile", "predict_image"}
    body = [node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {}
    exec("import numpy as np\n"
         "from tqdm import trange\n"
         "from scipy.special import softmax as scipy_softmax\n", namespace)
    exec(compile(ast.Module(body=body, type_ignores=[]), "segmenteveryforam.py", "exec"), namespace)
    return namespace["predict_image"]


def load_keras_compat():
    """Load segmenteveryforam/keras_compat.py without importing the package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "keras_compat", ROOT / "segmenteveryforam" / "keras_compat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(path):
    """Load a checkpoint as selected by SEF_LOADER: plain, compat or forced."""
    import keras

    loader = os.environ.get("SEF_LOADER", "plain")
    if loader == "plain":
        return keras.saving.load_model(path, compile=False)
    return load_keras_compat().load_unet(path, force=loader == "forced")


def predict_all(output):
    """Run every model on every image with the current backend."""
    import keras
    from PIL import Image

    predict_image = load_predict_image()
    timings = {"backend": keras.backend.backend()}
    arrays = {}
    for model_path in MODELS:
        model = load_model(ROOT / model_path)
        for image_path in IMAGES:
            image = np.array(Image.open(ROOT / image_path).convert("RGB"))
            predict_image(image[:512, :512], model, I=256)  # warm-up
            start = time.perf_counter()
            prediction = predict_image(image, model, I=256)
            key = f"{Path(model_path).stem}|{Path(image_path).stem}"
            timings[key] = time.perf_counter() - start
            arrays[key] = prediction.astype(np.float32)
    np.savez_compressed(output, **arrays)
    print(json.dumps(timings))


def layer_outputs(output):
    """Save every layer's output for one image tile with the current backend."""
    import keras
    from PIL import Image

    model = keras.saving.load_model(ROOT / MODELS[0], compile=False)
    probe = keras.Model(model.inputs, [layer.output for layer in model.layers[1:]])
    image = np.array(Image.open(ROOT / IMAGES[0]).convert("RGB"))
    tile = image[512:768, 512:768][None].astype("float32") / 255.0
    results = probe.predict(tile, verbose=0)
    np.savez_compressed(output, **{f"{i:02d}_{layer.name}": np.asarray(r)
                                   for i, (layer, r) in enumerate(zip(model.layers[1:], results))})
    print(json.dumps({"backend": keras.backend.backend()}))


def run_backend(backend, output, mode="--predict", loader="plain"):
    env = dict(os.environ, KERAS_BACKEND=backend, TF_CPP_MIN_LOG_LEVEL="2", SEF_LOADER=loader)
    result = subprocess.run(
        [sys.executable, __file__, mode, str(output)],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.exit(f"The {backend} run failed:\n{result.stderr[-3000:]}")
    timings = json.loads(result.stdout.strip().splitlines()[-1])
    if timings["backend"] != backend:
        sys.exit(f"Expected the {backend} backend but Keras used {timings['backend']}.")
    return timings


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def compare_layers():
    """Report where the two backends' layer outputs first diverge."""
    with tempfile.TemporaryDirectory() as temp:
        outputs = {b: Path(temp) / f"{b}.npz" for b in ("tensorflow", "torch")}
        for backend, path in outputs.items():
            run_backend(backend, path, "--layers")
        a, b = np.load(outputs["tensorflow"]), np.load(outputs["torch"])
        for key in a.files:
            diff = np.abs(a[key] - b[key])
            scale = np.abs(a[key]).max() or 1.0
            note = ""
            if diff.max() / scale > 1e-3:
                # Test for a one-pixel shift between the backends.
                shifts = {(dy, dx): np.abs(np.roll(b[key], (dy, dx), axis=(1, 2))[:, 1:-1, 1:-1]
                                           - a[key][:, 1:-1, 1:-1]).max()
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
                best = min(shifts, key=shifts.get)
                note = f"  best shift (dy, dx)={best}: {shifts[best]:.2e}"
            print(f"{key:28s} shape {str(a[key].shape):20s} max diff {diff.max():.2e} "
                  f"(relative {diff.max() / scale:.2e}){note}")


def compare(reference, candidate, timings, labels):
    """Print agreement between two sets of predictions; return True if they agree."""
    all_agree = True
    for key in reference.files:
        a, b = reference[key], candidate[key]
        diff = np.abs(a - b)
        a_map, b_map = a.argmax(-1), b.argmax(-1)
        agreement = (a_map == b_map).mean()
        # The specimen detector thresholds the interior probability at 0.5.
        interior_iou = iou(a[..., 1] >= 0.5, b[..., 1] >= 0.5)
        class_iou = ", ".join(f"{name} {iou(a_map == k, b_map == k):.6f}"
                              for k, name in enumerate(CLASSES))
        print(f"\n{key}")
        print(f"  max |prob difference|:  {diff.max():.2e}  (mean {diff.mean():.2e})")
        print(f"  class map agreement:    {agreement:.6%}")
        print(f"  class IoU between runs: {class_iou}")
        print(f"  interior >= 0.5 IoU:    {interior_iou:.6f}")
        print(f"  seconds: {labels[0]} {timings[0][key]:.1f}, {labels[1]} {timings[1][key]:.1f}")
        all_agree &= bool(agreement > 0.9999 and interior_iou > 0.9999)

    return all_agree


def main():
    import torch

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    runs = {
        "TensorFlow": ("tensorflow", "plain"),
        "PyTorch + load_unet": ("torch", "compat"),
        "TensorFlow + forced load_unet": ("tensorflow", "forced"),
    }
    with tempfile.TemporaryDirectory() as temp:
        results = {}
        for label, (backend, loader) in runs.items():
            path = Path(temp) / f"{backend}-{loader}.npz"
            timings = run_backend(backend, path, loader=loader)
            results[label] = (np.load(path), timings)
        reference, ref_timings = results["TensorFlow"]
        verdicts = {}
        for label in list(runs)[1:]:
            print(f"\n=== TensorFlow vs {label} ===")
            verdicts[label] = compare(reference, results[label][0],
                                      (ref_timings, results[label][1]), ("TensorFlow", label))
    print()
    for label, agree in verdicts.items():
        print(f"Result, {label}:", "agrees with TensorFlow." if agree else "DIFFERS from TensorFlow; see above.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--predict":
        predict_all(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--layers":
        layer_outputs(sys.argv[2])
    elif sys.argv[1:] == ["layers"]:
        compare_layers()
    else:
        main()
