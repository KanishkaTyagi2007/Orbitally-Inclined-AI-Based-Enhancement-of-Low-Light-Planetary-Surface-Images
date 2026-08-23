"""
app.py
=======
AURA-NET dashboard -- a minimal local web front end for the pipeline.

Upload a scene, watch it run, read the statistics, download the products.

    python app.py                 # then open http://127.0.0.1:5000

Design notes
------------
*Local tool, not a service.* It binds to localhost, runs one job at a time in a
worker thread, and keeps job state in memory. There is no auth, no database and
no multi-user isolation, because none of that is needed to look at your own
imagery -- and pretending otherwise would invite deploying it somewhere it does
not belong.

*The caveats travel with the numbers.* The pipeline's whole point is that you
can tell recovered signal from invented structure, so the dashboard surfaces
run conditions that qualify the result -- physics-only runs, an uninformative
trust map, a cropped input, ungated guardrails -- as prominently as the metrics
themselves. A dashboard that showed only the pretty numbers would undo the work
the pipeline does.

*Big scenes are cropped, loudly.* The pipeline is full-frame; a 592-megapixel
strip needs tens of GB. Oversized uploads are centre-cropped with a windowed
read (never a full decode) and the result is labelled as a crop everywhere it
is reported.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from aura_pipeline import AuraNetPipeline

PACKAGE_DIR = Path(__file__).resolve().parent
WORK_DIR = PACKAGE_DIR / "data" / "output" / "_dashboard"
CONFIG_DIR = PACKAGE_DIR / "config"

# A PDS4 product is a detached XML label plus a raw binary; both must land in
# the same directory and the label is the file GDAL is pointed at.
LABEL_SUFFIXES = {".xml", ".lbl"}
DATA_SUFFIXES = {".tif", ".tiff", ".img", ".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg"}

# The pipeline's preview dict also carries non-image entries (the output
# directory, the stretch method). Only these keys name an actual file.
IMAGE_KINDS = ("raw", "enhanced", "trust", "comparison")

app = Flask(__name__, static_folder=str(PACKAGE_DIR / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3      # 4 GB


# =============================================================================
# Job state
# =============================================================================
@dataclass
class Job:
    job_id: str
    name: str
    state: str = "queued"              # queued | running | done | error
    percent: int = 0
    stage: str = "Queued"
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started
        return {
            "job_id": self.job_id, "name": self.name, "state": self.state,
            "percent": self.percent, "stage": self.stage,
            "elapsed": round(elapsed, 1), "error": self.error,
            "notes": self.notes, "result": self.result,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()            # one pipeline run at a time


# =============================================================================
# Input handling
# =============================================================================
def probe_shape(path: Path) -> Optional[tuple[int, int]]:
    """(height, width) without decoding the whole raster, or None if unknown."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.height, src.width
    except Exception:
        pass
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            return int(page.imagelength), int(page.imagewidth)
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[1], im.size[0]
    except Exception:
        return None


def centre_crop(path: Path, max_dim: int, out_dir: Path) -> tuple[Path, dict]:
    """
    Windowed centre crop, so an oversized scene never has to be fully decoded.

    Returns (new_path, info). Georeferencing is carried across when the source
    has it: the window's affine transform is derived from the parent, so the
    crop stays correctly located rather than silently losing its position.
    """
    shape = probe_shape(path)
    if shape is None:
        return path, {"cropped": False}
    h, w = shape
    if max(h, w) <= max_dim:
        return path, {"cropped": False, "height": h, "width": w}

    ch, cw = min(h, max_dim), min(w, max_dim)
    row0, col0 = (h - ch) // 2, (w - cw) // 2
    out = out_dir / f"{path.stem}_crop{ch}x{cw}.tif"

    info = {
        "cropped": True, "source_height": h, "source_width": w,
        "height": ch, "width": cw, "row_offset": row0, "col_offset": col0,
    }

    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path) as src:
            window = Window(col0, row0, cw, ch)
            data = src.read(1, window=window)
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff", "height": ch, "width": cw, "count": 1,
                "dtype": data.dtype.name, "compress": "deflate",
            })
            transform = src.window_transform(window)
            if src.crs is not None:
                profile["transform"] = transform
            else:
                profile.pop("transform", None)
                profile.pop("crs", None)
            for key in ("photometric", "interleave", "blockxsize", "blockysize", "nodata"):
                profile.pop(key, None)
            tags = dict(src.tags())
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
            tags.update({f"CROP_{k.upper()}": str(v) for k, v in info.items()})
            dst.update_tags(**tags)
        return out, info
    except Exception:
        pass

    # Fallback: memory-mapped read for plain uncompressed TIFFs.
    import tifffile
    try:
        mm = tifffile.memmap(str(path), mode="r")
    except Exception:
        mm = tifffile.imread(str(path))
    arr = np.asarray(mm[row0:row0 + ch, col0:col0 + cw])
    tifffile.imwrite(str(out), arr,
                     metadata={f"CROP_{k}": str(v) for k, v in info.items()})
    return out, info


def choose_primary(paths: list[Path]) -> Path:
    """PDS4 ships a detached label; that is what the reader must be given."""
    labels = [p for p in paths if p.suffix.lower() in LABEL_SUFFIXES]
    if labels:
        return labels[0]
    data = [p for p in paths if p.suffix.lower() in DATA_SUFFIXES]
    return data[0] if data else paths[0]


def check_pds4_pair(primary: Path, uploaded: list[Path]) -> None:
    """
    A PDS4 `.img` is a headerless binary: its dimensions, data type and byte
    order live only in the detached `.xml` label. Handed the binary alone, GDAL
    declines it and the reader chain falls through to tifffile, which reports
    "not a TIFF file" -- an error that says nothing about the actual problem.

    Catch that here and say what to do instead, since selecting only the `.img`
    out of a PDS4 directory is the natural mistake to make.
    """
    if primary.suffix.lower() != ".img":
        return
    if any(p.suffix.lower() in LABEL_SUFFIXES for p in uploaded):
        return
    sibling = primary.with_suffix(".xml").name
    raise ValueError(
        f"'{primary.name}' is a PDS4 data file with no label. Its dimensions and "
        f"data type live in the detached label, so it cannot be read on its own. "
        f"Select BOTH '{sibling}' and '{primary.name}' together (Ctrl-click, or "
        f"drop both at once) and upload them in one go."
    )


# =============================================================================
# Worker
# =============================================================================
def run_job(job_id: str, primary: Path, job_dir: Path, config_path: Path,
            max_dim: int, oversize: str, uploaded: Optional[list[Path]] = None) -> None:
    job = JOBS[job_id]
    try:
        with RUN_LOCK:
            job.state = "running"
            job.stage = "Inspecting input"
            job.percent = 2

            check_pds4_pair(primary, uploaded or [primary])

            shape = probe_shape(primary)
            if shape:
                megapixels = shape[0] * shape[1] / 1e6
                job.notes.append(f"Input {shape[1]} x {shape[0]} px ({megapixels:.1f} Mpx)")
                if max(shape) > max_dim:
                    if oversize == "reject":
                        raise ValueError(
                            f"Scene is {shape[1]} x {shape[0]} px, larger than the "
                            f"{max_dim} px limit. Re-run allowing a centre crop, "
                            "raise the limit, or use the CLI on a pre-cut tile."
                        )
                    job.stage = "Centre-cropping oversized scene"
                    primary, info = centre_crop(primary, max_dim, job_dir)
                    if info.get("cropped"):
                        job.notes.append(
                            f"CROPPED to a centre {info['width']} x {info['height']} px "
                            f"window at offset ({info['col_offset']}, {info['row_offset']}). "
                            "Metrics describe the crop, not the full scene."
                        )

            def progress(pct: int, label: str) -> None:
                job.percent = int(pct)
                job.stage = label

            pipeline = AuraNetPipeline(config_path=str(config_path))
            if not pipeline.checkpoint_loaded:
                job.notes.append(
                    "PHYSICS-ONLY run: no trained weights loaded, so the learned "
                    "stages are identity pass-throughs and the trust map carries "
                    "only the physical masks."
                )

            result = pipeline.process(
                str(primary),
                output_dir=str(job_dir / "products"),
                preview_dir=str(job_dir / "previews"),
                progress=progress,
            )

            metrics = result["metrics"]
            gc = metrics.get("gradient_consistency_gated")
            if gc is False:
                job.notes.append(
                    "Gradient consistency is REPORTED, not gated -- no fixed "
                    "threshold survived calibration across scene sizes. Flux "
                    "conservation and the zero-synthesis guarantee are what gate."
                )
            if metrics.get("trust_map_informative") is False:
                job.notes.append(
                    "Trust map is uniform: without trained weights the variance "
                    "head has nothing to say. Only saturated, cosmic-ray and "
                    "nodata pixels are marked untrusted."
                )

            job.result = {
                "metrics": metrics,
                "config": config_path.name,
                "images": {kind: Path(path).name
                           for kind, path in (
                               (k.removeprefix("preview_"), v)
                               for k, v in result.get("previews", {}).items())
                           if kind in IMAGE_KINDS},
                "downloads": {
                    "enhanced_geotiff": Path(result["enhanced_path"]).name,
                    "trust_geotiff": (Path(result["trust_path"]).name
                                      if result.get("trust_path") else None),
                    "metrics_json": Path(result["metrics_path"]).name,
                },
            }
            job.state = "done"
            job.percent = 100
            job.stage = "Complete"
    except Exception as exc:                                # surfaced to the UI
        job.state = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Failed"
        traceback.print_exc()
    finally:
        job.finished = time.time()


# =============================================================================
# Routes
# =============================================================================
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/configs")
def list_configs():
    configs = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    return jsonify({
        "configs": configs,
        "default": "default_config.yaml" if "default_config.yaml" in configs
                   else (configs[0] if configs else None),
    })


@app.post("/api/run")
def start_run():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify({"error": "No file uploaded."}), 400

    config_name = secure_filename(request.form.get("config", "default_config.yaml"))
    config_path = CONFIG_DIR / config_name
    if not config_path.is_file():
        return jsonify({"error": f"Unknown config: {config_name}"}), 400

    try:
        max_dim = max(64, min(int(request.form.get("max_dim", 2048)), 16384))
    except ValueError:
        max_dim = 2048
    oversize = request.form.get("oversize", "crop")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    (job_dir / "input").mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        name = secure_filename(f.filename) or "upload.bin"
        dest = job_dir / "input" / name
        f.save(dest)
        saved.append(dest)

    primary = choose_primary(saved)
    job = Job(job_id=job_id, name=primary.name)
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_job,
        args=(job_id, primary, job_dir, config_path, max_dim, oversize, saved),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job.public())


def _serve(job_id: str, subdir: str, filename: str, download: bool):
    if job_id not in JOBS:
        abort(404)
    safe = secure_filename(filename)
    path = (WORK_DIR / job_id / subdir / safe).resolve()
    # Containment check: never serve outside this job's own directory.
    if not str(path).startswith(str((WORK_DIR / job_id).resolve())) or not path.is_file():
        abort(404)
    return send_file(path, as_attachment=download, download_name=safe)


@app.get("/api/jobs/<job_id>/preview/<filename>")
def job_preview(job_id: str, filename: str):
    return _serve(job_id, "previews", filename, download=False)


@app.get("/api/jobs/<job_id>/product/<filename>")
def job_product(job_id: str, filename: str):
    return _serve(job_id, "products", filename, download=True)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Upload exceeds the 4 GB limit."}), 413


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("[AURA-NET] Dashboard: http://127.0.0.1:5000")
    print(f"[AURA-NET] Job workspace: {WORK_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
