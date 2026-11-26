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


def predict_all(output):
    """Run every model on every image with the current backend."""
    import keras
    from PIL import Image

    predict_image = load_predict_image()
    timings = {"backend": keras.backend.backend()}
    arrays = {}
    for model_path in MODELS:
        model = keras.saving.load_model(ROOT / model_path, compile=False)
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


def run_backend(backend, output):
    env = dict(os.environ, KERAS_BACKEND=backend, TF_CPP_MIN_LOG_LEVEL="2")
    result = subprocess.run(
        [sys.executable, __file__, "--predict", str(output)],
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


def main():
    import torch

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    with tempfile.TemporaryDirectory() as temp:
        outputs = {b: Path(temp) / f"{b}.npz" for b in ("tensorflow", "torch")}
        timings = {b: run_backend(b, path) for b, path in outputs.items()}
        tf_pred = np.load(outputs["tensorflow"])
        torch_pred = np.load(outputs["torch"])

        all_agree = True
        for key in tf_pred.files:
            a, b = tf_pred[key], torch_pred[key]
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
            print(f"  seconds: TensorFlow {timings['tensorflow'][key]:.1f}, "
                  f"PyTorch {timings['torch'][key]:.1f}")
            all_agree &= bool(agreement > 0.9999 and interior_iou > 0.9999)

    print("\nResult:", "the backends agree." if all_agree else "the backends DIFFER; see above.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--predict":
        predict_all(sys.argv[2])
    else:
        main()
