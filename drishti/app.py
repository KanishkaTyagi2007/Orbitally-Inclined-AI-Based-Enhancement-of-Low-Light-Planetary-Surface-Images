"""
app.py
=======
AURA-NET dashboard -- a minimal local web front end for the pipeline.

Point it at a scene, watch it run, read the statistics, download the products.

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

*Big scenes are cropped, loudly.* The pipeline is full-frame; a 977-megapixel
strip needs tens of GB. Oversized inputs are centre-cropped with a windowed
read (never a full decode) and the result is labelled as a crop everywhere it
is reported.

Three ways in
-------------
An ISSDC download is a directory tree whose science raster is a ~2 GB `.img`
next to the `.xml` label that gives it a shape. All three entry points below
converge on `pds4_bundle.discover_products`, so the browse quicklook PNG is
never mistaken for the data and the label's own illumination geometry always
reaches the photometric stage.

  1. *Local path* -- paste the extracted folder. Nothing is copied or uploaded;
     the pipeline reads the label in place. This is the only practical route
     for a full-size product, and it is why this endpoint exists at all.
  2. *Bundle ZIP* -- upload the archive as downloaded; it is extracted into the
     job directory and searched.
  3. *Loose files* -- the original path, for a GeoTIFF or a hand-picked
     `.xml` + `.img` pair.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

from aura_pipeline import AuraNetPipeline
from pds4_bundle import LABEL_SUFFIXES, choose_best, discover_products, extract_archive

PACKAGE_DIR = Path(__file__).resolve().parent
WORK_DIR = PACKAGE_DIR / "data" / "output" / "_dashboard"
CONFIG_DIR = PACKAGE_DIR / "config"
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"

DATA_SUFFIXES = {".tif", ".tiff", ".img", ".fits", ".fit", ".fts", ".png", ".jpg", ".jpeg"}

# The pipeline's preview dict also carries non-image entries (the output
# directory, the stretch method). Only these keys name an actual file.
IMAGE_KINDS = ("raw", "enhanced", "trust", "comparison")

# Requests that start work or read the filesystem must carry this header.
# A cross-origin page can forge a plain form POST to localhost, but it cannot
# set a custom header without a CORS preflight this server never answers -- so
# this is what stops a random browser tab from driving the dashboard into
# reading arbitrary local paths and rendering them back.
CLIENT_HEADER = "X-Aura-Client"
ALLOWED_ORIGIN_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

app = Flask(__name__, static_folder=str(PACKAGE_DIR / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3      # 8 GB


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
    source: Optional[dict] = None      # what the input resolved to

    def public(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started
        return {
            "job_id": self.job_id, "name": self.name, "state": self.state,
            "percent": self.percent, "stage": self.stage,
            "elapsed": round(elapsed, 1), "error": self.error,
            "notes": self.notes, "result": self.result, "source": self.source,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()            # one pipeline run at a time


# =============================================================================
# Request guard
# =============================================================================
def reject_cross_origin() -> Optional[tuple]:
    """Returns an error response for a request that did not come from the UI."""
    if request.headers.get(CLIENT_HEADER) is None:
        return jsonify({"error": f"Missing {CLIENT_HEADER} header."}), 403
    origin = request.headers.get("Origin")
    if origin:
        from urllib.parse import urlparse
        if (urlparse(origin).hostname or "") not in ALLOWED_ORIGIN_HOSTS:
            return jsonify({"error": f"Origin {origin} is not allowed."}), 403
    return None


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


def centre_crop(path: Path, max_dim: int, out_dir: Path,
                extra_tags: Optional[dict] = None) -> tuple[Path, dict]:
    """
    Windowed centre crop, so an oversized scene never has to be fully decoded.

    Returns (new_path, info). Georeferencing is carried across when the source
    has it: the window's affine transform is derived from the parent, so the
    crop stays correctly located rather than silently losing its position.

    `extra_tags` carries the PDS4 label's observing geometry onto the crop.
    Without it the crop is a plain GeoTIFF with no `.xml` beside it, the
    ingestor's label lookup finds nothing, and the photometric stage silently
    falls back to the config's scene-constant incidence angle -- which is the
    wrong angle for every product but the one it was written from.
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
            tags.update(extra_tags or {})
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
    metadata = {f"CROP_{k}": str(v) for k, v in info.items()}
    metadata.update({k: str(v) for k, v in (extra_tags or {}).items()})
    tifffile.imwrite(str(out), arr, metadata=metadata)
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
        f"Either paste the path of the extracted product folder, upload the "
        f"bundle ZIP, or select BOTH '{sibling}' and '{primary.name}' together."
    )


@dataclass
class ResolvedInput:
    """What the pipeline will actually be handed, and what it was found in."""

    primary: Path
    product: Optional[object] = None      # Pds4Product, when one was found
    notes: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "primary": str(self.primary),
            "product": self.product.summary() if self.product is not None else None,
            "candidates": self.candidates,
            "notes": self.notes,
        }


def resolve_bundle(root: Path) -> ResolvedInput:
    """Picks the science product out of a bundle tree (or a bare directory)."""
    products = discover_products(root)
    best = choose_best(products)
    notes: list[str] = []

    if best is None:
        # No label anywhere. Fall back to any raster that is not a browse
        # quicklook, so a folder of plain GeoTIFFs still works.
        rasters = [p for p in sorted(root.rglob("*"))
                   if p.suffix.lower() in DATA_SUFFIXES and p.is_file()
                   and "browse" not in {x.lower() for x in p.relative_to(root).parts[:-1]}]
        if not rasters:
            raise ValueError(
                f"No readable product under '{root.name}'. Expected a PDS4 "
                "label (.xml) beside its .img, or a GeoTIFF/FITS raster."
            )
        notes.append(f"No PDS4 label found; using raster '{rasters[0].name}' directly.")
        return ResolvedInput(primary=rasters[0], notes=notes)

    others = [p for p in products if p.readable and p.label_path != best.label_path]
    if others:
        notes.append(
            f"Bundle holds {len(others) + 1} products; selected the "
            f"{best.processing_level.lower()} {best.sensor.lower()} scene "
            f"'{best.name}'. Point at a single .xml to choose another."
        )
    size = (f"{best.samples} x {best.lines} px "
            f"({best.megapixels:.1f} Mpx)" if best.megapixels < 10
            else f"{best.samples} x {best.lines} px ({best.megapixels:.0f} Mpx)")
    identity = f"{best.instrument.upper()} {best.sensor}, {best.processing_level.lower()}"
    notes.append(
        f"{identity}, {size}, incidence {best.incidence_deg:.2f} deg"
        if best.incidence_deg is not None else f"{identity}, {size}"
    )
    for warning in best.warnings:
        notes.append(warning)

    return ResolvedInput(
        primary=best.label_path, product=best, notes=notes,
        candidates=[p.summary() for p in products if p.readable],
    )


def resolve_local_path(raw: str, extract_dir: Path) -> ResolvedInput:
    """
    Resolves a filesystem path the user typed: a bundle folder, a bundle ZIP,
    a `.xml` label, or a plain raster.
    """
    path = Path(os.path.expandvars(raw.strip().strip('"').strip("'"))).expanduser()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if path.is_file() and path.suffix.lower() == ".zip":
        root = extract_archive(path, extract_dir)
        resolved = resolve_bundle(root)
        resolved.notes.insert(0, f"Extracted '{path.name}' into the job directory.")
        return resolved

    if path.is_dir():
        return resolve_bundle(path)

    if path.suffix.lower() in LABEL_SUFFIXES:
        return resolve_bundle(path.parent) if not _label_is_product(path) \
            else _single_label(path)

    return ResolvedInput(primary=path,
                         notes=[f"Reading '{path.name}' directly."])


def _label_is_product(path: Path) -> bool:
    from pds4_bundle import parse_label
    product = parse_label(path)
    return product is not None and product.readable


def _single_label(path: Path) -> ResolvedInput:
    from pds4_bundle import parse_label
    product = parse_label(path)
    return ResolvedInput(primary=path, product=product,
                         candidates=[product.summary()],
                         notes=[f"Using label '{path.name}'."])


def resolve_uploads(saved: list[Path], job_dir: Path) -> ResolvedInput:
    """Resolves what was uploaded: a bundle ZIP, a directory tree, or loose files."""
    archives = [p for p in saved if p.suffix.lower() == ".zip"]
    if archives:
        root = extract_archive(archives[0], job_dir / "extracted")
        resolved = resolve_bundle(root)
        resolved.notes.insert(0, f"Extracted '{archives[0].name}'.")
        return resolved

    if len(saved) > 2:
        # A directory upload: let bundle discovery pick, rather than taking
        # whichever label happened to sort first (which for an ISSDC tree is
        # the browse quicklook's).
        try:
            return resolve_bundle(job_dir / "input")
        except ValueError:
            pass

    primary = choose_primary(saved)
    check_pds4_pair(primary, saved)
    if primary.suffix.lower() in LABEL_SUFFIXES and _label_is_product(primary):
        return _single_label(primary)
    return ResolvedInput(primary=primary)


# =============================================================================
# Worker
# =============================================================================
def run_job(job_id: str, resolved: ResolvedInput, job_dir: Path,
            config_path: Path, max_dim: int, oversize: str,
            checkpoint: Optional[Path]) -> None:
    job = JOBS[job_id]
    primary = resolved.primary
    try:
        with RUN_LOCK:
            job.state = "running"
            job.stage = "Inspecting input"
            job.percent = 2
            job.notes.extend(resolved.notes)

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
                    label_tags = (resolved.product.label_metadata()
                                  if resolved.product is not None else {})
                    primary, info = centre_crop(primary, max_dim, job_dir, label_tags)
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
            if checkpoint is not None:
                job.stage = "Loading trained weights"
                loaded = pipeline.load_checkpoint(str(checkpoint))
                job.notes.append(
                    f"Trained weights from '{checkpoint.name}': "
                    f"{', '.join(loaded) if loaded else 'no matching sections'}."
                )
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
                    "Trust map is uniform: the variance head has nothing to say "
                    "for this scene. Only saturated, cosmic-ray and nodata pixels "
                    "are marked untrusted."
                )
            if metrics.get("incidence_source") == "config_default":
                job.notes.append(
                    "Illumination geometry came from the CONFIG, not the product. "
                    "The Lommel-Seeliger correction divides by cos(i)/(cos(i)+cos(e)), "
                    "so this scene's photometry is only as right as that constant."
                )

            job.result = {
                "metrics": metrics,
                "config": config_path.name,
                "checkpoint": checkpoint.name if checkpoint else None,
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
    checkpoints = sorted(p.name for p in CHECKPOINT_DIR.glob("*.pt")) \
        if CHECKPOINT_DIR.is_dir() else []
    return jsonify({
        "configs": configs,
        "default": "default_config.yaml" if "default_config.yaml" in configs
                   else (configs[0] if configs else None),
        "checkpoints": checkpoints,
    })


@app.post("/api/inspect")
def inspect_path():
    """
    Reports what a local path resolves to, without running anything.

    Separate from `/api/run` on purpose: a full-size product takes minutes, and
    finding out only then that the bundle resolved to the fore-sensor raw strip
    instead of the calibrated nadir one is an expensive way to learn it.
    """
    rejection = reject_cross_origin()
    if rejection:
        return rejection

    raw = (request.json or {}).get("path", "") if request.is_json \
        else request.form.get("path", "")
    if not raw.strip():
        return jsonify({"error": "No path given."}), 400
    try:
        resolved = resolve_local_path(raw, WORK_DIR / f"inspect_{uuid.uuid4().hex[:8]}")
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    shape = probe_shape(resolved.primary)
    payload = resolved.public()
    payload["readable"] = shape is not None
    payload["shape"] = list(shape) if shape else None
    return jsonify(payload)


@app.post("/api/run")
def start_run():
    rejection = reject_cross_origin()
    if rejection:
        return rejection

    files = [f for f in request.files.getlist("files") if f and f.filename]
    local_path = request.form.get("path", "").strip()
    if not files and not local_path:
        return jsonify({"error": "Give a local path or upload a file."}), 400

    config_name = secure_filename(request.form.get("config", "default_config.yaml"))
    config_path = CONFIG_DIR / config_name
    if not config_path.is_file():
        return jsonify({"error": f"Unknown config: {config_name}"}), 400

    checkpoint = None
    checkpoint_name = secure_filename(request.form.get("checkpoint", ""))
    if checkpoint_name:
        checkpoint = CHECKPOINT_DIR / checkpoint_name
        if not checkpoint.is_file():
            return jsonify({"error": f"Unknown checkpoint: {checkpoint_name}"}), 400

    try:
        max_dim = max(64, min(int(request.form.get("max_dim", 2048)), 16384))
    except ValueError:
        max_dim = 2048
    oversize = request.form.get("oversize", "crop")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    (job_dir / "input").mkdir(parents=True, exist_ok=True)

    try:
        if local_path:
            resolved = resolve_local_path(local_path, job_dir / "extracted")
        else:
            saved = []
            for f in files:
                # `webkitRelativePath` arrives as the field name for a folder
                # upload; keeping the tree matters because a PDS4 label
                # references its .img by a path relative to itself.
                relative = Path(f.filename.replace("\\", "/"))
                parts = [secure_filename(p) for p in relative.parts
                         if p not in ("", ".", "..")]
                dest = job_dir / "input" / Path(*(parts or ["upload.bin"]))
                dest.parent.mkdir(parents=True, exist_ok=True)
                f.save(dest)
                saved.append(dest)
            resolved = resolve_uploads(saved, job_dir)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    job = Job(job_id=job_id, name=resolved.primary.name, source=resolved.public())
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(
        target=run_job,
        args=(job_id, resolved, job_dir, config_path, max_dim, oversize, checkpoint),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "source": resolved.public()})


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
    return jsonify({"error": "Upload exceeds the 8 GB limit. For a full-size "
                             "product, paste its folder path instead."}), 413


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("[AURA-NET] Dashboard: http://127.0.0.1:5000")
    print(f"[AURA-NET] Job workspace: {WORK_DIR}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
