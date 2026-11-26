"""Compare the full segmentation workflow between Keras's TensorFlow and PyTorch backends.

Runs ``predict_with_box_prompts`` (U-Net, then SAM 2.1 box prompts), as used by
the Colab notebook and the browser prototype, on the example images once with
each backend. Reports the specimen outlines' agreement, the time per image and
the GPU memory each run used.

Needs a CUDA GPU and SAM 2 installed. Usage, from the repository root::

    python tools/compare_workflow_backends.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNET = ROOT / "models" / "best_val_loss.keras"
IMAGES = [
    "docs/source/_static/images/g_ruber_original.jpeg",
    "docs/source/_static/images/g_bulloides_original.jpeg",
]
SAM_URL = ("https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/"
           "sam2.1_hiera_large.pt")


def sam_checkpoint():
    directory = Path(os.environ.get("SEF_MODEL_DIR", Path.home() / ".cache" / "segmenteveryforam"))
    path = directory / "sam2.1_hiera_large.pt"
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".part")
        urlretrieve(SAM_URL, pending)
        pending.replace(path)
    return path


def process_gpu_memory_mb():
    """GPU memory used by this process, as nvidia-smi reports it."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    for line in result.stdout.splitlines():
        pid, used = (part.strip() for part in line.split(","))
        if pid == str(os.getpid()) and used.isdigit():
            return int(used)
    return None


def run_workflow(output):
    """Segment every example image with the current backend."""
    sys.path.insert(0, str(ROOT))
    import keras
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2

    import segmenteveryforam as sef

    start = time.perf_counter()
    unet = sef.load_unet(UNET)
    sam = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(sam_checkpoint()), device="cuda")
    report = {"backend": keras.backend.backend(), "load_seconds": time.perf_counter() - start,
              "images": {}}
    warm_up = np.array(Image.open(ROOT / IMAGES[0]).convert("RGB"))[:768, :768]
    sef.predict_with_box_prompts(warm_up, unet, sam)

    polygons = {}
    for image_path in IMAGES:
        image = np.array(Image.open(ROOT / image_path).convert("RGB"))
        torch.cuda.synchronize()
        start = time.perf_counter()
        result, _, _ = sef.predict_with_box_prompts(image, unet, sam)
        torch.cuda.synchronize()
        name = Path(image_path).stem
        report["images"][name] = {"seconds": time.perf_counter() - start, "count": len(result)}
        polygons[name] = [polygon.wkt for polygon in result]

    report["torch_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    report["torch_reserved_mb"] = torch.cuda.memory_reserved() / 2**20
    report["process_gpu_mb"] = process_gpu_memory_mb()
    report["tensorflow_imported"] = "tensorflow" in sys.modules
    Path(output).write_text(json.dumps({"report": report, "polygons": polygons}))


def run_backend(backend, output):
    env = dict(os.environ, KERAS_BACKEND=backend, TF_CPP_MIN_LOG_LEVEL="2")
    result = subprocess.run([sys.executable, __file__, "--run", str(output)],
                            env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"The {backend} run failed:\n{result.stderr[-4000:]}")
    data = json.loads(Path(output).read_text())
    if data["report"]["backend"] != backend:
        sys.exit(f"Expected the {backend} backend but Keras used {data['report']['backend']}.")
    return data


def match_polygons(reference, candidate):
    """IoU of each reference polygon with its best match among the candidates."""
    from shapely import wkt

    candidates = [wkt.loads(text) for text in candidate]
    ious = []
    for text in reference:
        polygon = wkt.loads(text)
        best = 0.0
        for other in candidates:
            if polygon.intersects(other):
                union = polygon.union(other).area
                best = max(best, polygon.intersection(other).area / union if union else 0.0)
        ious.append(best)
    return np.array(ious)


def main():
    with tempfile.TemporaryDirectory() as temp:
        runs = {backend: run_backend(backend, Path(temp) / f"{backend}.json")
                for backend in ("tensorflow", "torch")}

    tf_report, torch_report = runs["tensorflow"]["report"], runs["torch"]["report"]
    agree = True
    for name in tf_report["images"]:
        tf_polygons = runs["tensorflow"]["polygons"][name]
        torch_polygons = runs["torch"]["polygons"][name]
        ious = match_polygons(tf_polygons, torch_polygons)
        reverse = match_polygons(torch_polygons, tf_polygons)
        print(f"\n{name}")
        print(f"  specimens: TensorFlow {len(tf_polygons)}, PyTorch {len(torch_polygons)}")
        if len(ious):
            print(f"  outline IoU with best match: mean {ious.mean():.6f}, min {ious.min():.6f}; "
                  f"identical outlines: {int((ious > 0.999999).sum())} of {len(ious)}")
        print(f"  unmatched (IoU < 0.5): TensorFlow {int((ious < 0.5).sum())}, "
              f"PyTorch {int((reverse < 0.5).sum())}")
        print(f"  seconds: TensorFlow {tf_report['images'][name]['seconds']:.1f}, "
              f"PyTorch {torch_report['images'][name]['seconds']:.1f}")
        agree &= (len(tf_polygons) == len(torch_polygons)
                  and (not len(ious) or ious.min() > 0.99))

    for label, report in (("TensorFlow", tf_report), ("PyTorch", torch_report)):
        print(f"\n{label}: model loading {report['load_seconds']:.1f} s; "
              f"GPU memory used by the process {report['process_gpu_mb']} MB "
              f"(PyTorch peak {report['torch_peak_mb']:.0f} MB, reserved {report['torch_reserved_mb']:.0f} MB); "
              f"TensorFlow imported: {report['tensorflow_imported']}")
    print("\nResult:", "the workflows agree." if agree else "the workflows DIFFER; see above.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--run":
        run_workflow(sys.argv[2])
    else:
        main()
