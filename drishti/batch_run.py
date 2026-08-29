"""
batch_run.py
=============
Runs the pipeline across every product in a PDS4 bundle folder and collects the
results into one comparable table.

    python batch_run.py --data <bundle folder>
    python batch_run.py --data <bundle folder> --checkpoint checkpoints/aura_net_tmc.pt

Why windows rather than whole strips
------------------------------------
These products are pushbroom strips: 4000 samples across-track by up to 245 000
lines along-track, which is 977 megapixels, and the pipeline is full-frame --
the stationary wavelet transform alone holds 6 undecimated float32 sub-bands at
native resolution, so one strip would need well over 100 GB. Every run here is
therefore a set of windows, and every number reported describes those windows.

The windows are stratified along-track rather than taken from the centre. A
single centre crop of a strip that spans 40 degrees of latitude would
characterise one patch of terrain under one illumination and say nothing about
the rest, which is exactly the kind of quiet generalisation this pipeline is
built to avoid making.

What the summary is for
-----------------------
`summary.csv` puts each window's guardrail verdicts beside its metrics, so a
failure is visible as a row rather than buried in one of ninety JSON files. The
columns that matter for trusting a product are `all_guardrails_passed`,
`flux_conservation_passed`, `zero_synthesis_guarantee_held` and
`incidence_source` -- the last one because a scene whose geometry came from the
config rather than its own label has been photometrically normalised with an
angle that belongs to a different scene.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from aura_pipeline import AuraNetPipeline
from pds4_bundle import Pds4Product, discover_products

PACKAGE_DIR = Path(__file__).resolve().parent

# Columns lifted into summary.csv, in the order a reviewer wants to read them:
# what it was, whether to trust it, then how it scored.
SUMMARY_COLUMNS = [
    "product", "instrument", "sensor", "processing_level", "window",
    "row_offset", "col_offset", "window_px", "status",
    "all_guardrails_passed", "flux_conservation_passed",
    "structure_guardrail_passed", "zero_synthesis_guarantee_held",
    "trust_map_informative", "trained_weights_loaded",
    "incidence_source", "incidence_deg_applied", "emission_deg_applied",
    "ssim", "psnr", "niqe", "brisque", "entropy_gain",
    "mean_trust", "low_trust_pixel_fraction",
    "gradient_correlation", "flux_drift_coarsest_scale",
    "craters_detected_raw", "craters_detected_enhanced", "craters_matched",
    "cosmic_ray_hit_fraction", "saturated_pixel_fraction",
    "radiance_units", "runtime_seconds",
]


# =============================================================================
# Window extraction
# =============================================================================
def stratified_windows(product: Pds4Product, count: int, size: int,
                       seed: int = 0) -> list[tuple[int, int]]:
    """
    `count` along-track bands, one window each, centred across-track.

    Across-track is not randomised: a pushbroom's optical and radiometric
    behaviour varies systematically with field angle, so holding the column
    fixed keeps the windows comparable to each other. Along-track is where the
    terrain and illumination actually change, so that is what gets spread.
    """
    height = max(product.lines - size, 0)
    col = max((product.samples - size) // 2, 0)
    if count <= 1:
        return [(height // 2, col)]
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, height, count + 1)
    rows = (edges[:-1] + rng.random(count) * np.diff(edges)).astype(int)
    return [(int(r), col) for r in rows]


def extract_window(product: Pds4Product, row: int, col: int, size: int,
                   out_path: Path) -> tuple[Path, dict]:
    """
    Windowed read into a GeoTIFF, carrying the label's geometry as tags.

    The tags are the point. Once the window is a bare `.tif` there is no `.xml`
    beside it for the ingestor to read, so without them the photometric stage
    would fall back to the config's incidence angle -- which across these
    products is wrong by up to 44 degrees.
    """
    import rasterio
    from rasterio.windows import Window

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(product.label_path) as src:
        height = min(size, src.height - row)
        width = min(size, src.width - col)
        window = Window(col, row, width, height)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update({"driver": "GTiff", "height": height, "width": width,
                        "count": 1, "dtype": data.dtype.name, "compress": "deflate"})
        if src.crs is not None:
            profile["transform"] = src.window_transform(window)
        else:
            profile.pop("transform", None)
            profile.pop("crs", None)
        for key in ("photometric", "interleave", "blockxsize", "blockysize", "nodata"):
            profile.pop(key, None)
        tags = dict(src.tags())

    tags.update(product.label_metadata())
    tags.update({"WINDOW_ROW_OFFSET": str(row), "WINDOW_COL_OFFSET": str(col),
                 "WINDOW_SOURCE_LINES": str(product.lines),
                 "WINDOW_SOURCE_SAMPLES": str(product.samples)})

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)

    return out_path, {"row_offset": row, "col_offset": col,
                      "height": height, "width": width}


# =============================================================================
# Batch
# =============================================================================
def run_batch(data_dir: str | Path, output_dir: str | Path, config: str,
              windows: int = 3, window_px: int = 1024,
              checkpoint: Optional[str] = None, device: Optional[str] = None,
              limit: Optional[int] = None, seed: int = 0) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    products = [p for p in discover_products(data_dir) if p.readable]
    if limit:
        products = products[:limit]
    if not products:
        raise SystemExit(f"No readable PDS4 products under {data_dir}")

    pipeline = AuraNetPipeline(config_path=config, device=device)
    if checkpoint:
        loaded = pipeline.load_checkpoint(checkpoint)
        print(f"[batch] checkpoint '{Path(checkpoint).name}' -> "
              f"{', '.join(loaded) if loaded else 'no matching sections'}")
    else:
        print("[batch] physics-only run (no checkpoint): the learned stages are "
              "identity pass-throughs.")

    print(f"[batch] {len(products)} products x {windows} window(s) of "
          f"{window_px} px  ->  {output_dir}")

    rows: list[dict] = []
    started = time.perf_counter()

    for index, product in enumerate(products, start=1):
        product_dir = output_dir / product.name
        instrument_note = ""
        # The config's calibration constants are per instrument. A TMC config
        # applied to an OHRC product would silently convert DN with the wrong
        # gain and exposure, so say so rather than quietly reporting the number.
        if product.instrument and product.instrument not in Path(config).stem.lower():
            instrument_note = (f"config '{Path(config).name}' does not name "
                               f"instrument '{product.instrument.upper()}'")

        for window_index, (row, col) in enumerate(
                stratified_windows(product, windows, window_px, seed)):
            tag = f"w{window_index}"
            label = f"[{index:2d}/{len(products)}] {product.name[:42]:44s} {tag}"
            record = {
                "product": product.name,
                "instrument": product.instrument.upper(),
                "sensor": product.sensor,
                "processing_level": product.processing_level,
                "window": window_index,
                "row_offset": row,
                "col_offset": col,
                "window_px": window_px,
            }
            if instrument_note:
                record["instrument_mismatch"] = instrument_note

            try:
                window_path, info = extract_window(
                    product, row, col, window_px,
                    product_dir / "windows" / f"{product.name}_{tag}.tif")
                result = pipeline.process(
                    str(window_path),
                    output_dir=str(product_dir / "products"),
                    preview_dir=str(product_dir / "previews"),
                )
                record.update(info)
                record.update(result["metrics"])
                record["status"] = "ok"
                record["metrics_path"] = result["metrics_path"]
                record["preview_image"] = result.get("preview_image")

                verdict = "PASS" if record.get("all_guardrails_passed") else "FAIL"
                print(f"  {label} {verdict}  ssim={record.get('ssim')}  "
                      f"trust={record.get('mean_trust')}  "
                      f"{record.get('runtime_seconds')}s")
            except Exception as exc:
                record["status"] = f"{type(exc).__name__}: {exc}"
                print(f"  {label} ERROR  {record['status']}")
                traceback.print_exc()

            rows.append(record)

    elapsed = time.perf_counter() - started
    summary = summarize(rows, products, config, checkpoint, elapsed)

    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    write_csv(rows, output_dir / "summary.csv")

    print(f"\n[batch] {summary['windows_ok']}/{summary['windows_total']} windows "
          f"processed, {summary['guardrails_passed']} passed all guardrails "
          f"({elapsed / 60:.1f} min)")
    print(f"[batch] summary -> {output_dir / 'summary.csv'}")
    if summary["windows_failed"]:
        print(f"[batch] {summary['windows_failed']} window(s) errored -- see "
              "summary.json for the messages.")
    return summary


def summarize(rows: list[dict], products: list[Pds4Product], config: str,
              checkpoint: Optional[str], elapsed: float) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]

    def mean(key: str) -> Optional[float]:
        values = [r[key] for r in ok
                  if isinstance(r.get(key), (int, float)) and not isinstance(r[key], bool)]
        return round(float(np.mean(values)), 5) if values else None

    return {
        "products": len(products),
        "windows_total": len(rows),
        "windows_ok": len(ok),
        "windows_failed": len(rows) - len(ok),
        "guardrails_passed": sum(1 for r in ok if r.get("all_guardrails_passed")),
        "flux_conservation_passed": sum(1 for r in ok if r.get("flux_conservation_passed")),
        "zero_synthesis_held": sum(1 for r in ok if r.get("zero_synthesis_guarantee_held")),
        "trust_informative": sum(1 for r in ok if r.get("trust_map_informative")),
        "geometry_from_label": sum(1 for r in ok
                                   if r.get("incidence_source") == "product_metadata"),
        "trained_weights_loaded": bool(ok and ok[0].get("trained_weights_loaded")),
        "mean_ssim": mean("ssim"),
        "mean_psnr": mean("psnr"),
        "mean_entropy_gain": mean("entropy_gain"),
        "mean_trust": mean("mean_trust"),
        "mean_gradient_correlation": mean("gradient_correlation"),
        "mean_runtime_seconds": mean("runtime_seconds"),
        "config": Path(config).name,
        "checkpoint": Path(checkpoint).name if checkpoint else None,
        "elapsed_minutes": round(elapsed / 60, 2),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    """
    Writes the curated columns first, then any extra key some row happened to
    carry, so a metric added to the pipeline later still lands in the table
    instead of being silently dropped.
    """
    extra = sorted({k for r in rows for k in r} - set(SUMMARY_COLUMNS)
                   - {"histogram_raw", "histogram_enhanced",
                      "flux_drift_per_scale", "flux_drift_after_tone_mapping"})
    columns = SUMMARY_COLUMNS + extra
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AURA-NET across every product in a PDS4 bundle folder.")
    parser.add_argument("--data", type=str, required=True,
                        help="Bundle folder (searched recursively for PDS4 labels).")
    parser.add_argument("--output-dir", type=str,
                        default=str(PACKAGE_DIR / "data" / "output" / "batch"))
    parser.add_argument("--config", type=str, default="config/ch2_tmc.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--windows", type=int, default=3,
                        help="Along-track windows per product.")
    parser.add_argument("--window-px", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N products (smoke testing).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_batch(args.data, args.output_dir, args.config, windows=args.windows,
              window_px=args.window_px, checkpoint=args.checkpoint,
              device=args.device, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()
