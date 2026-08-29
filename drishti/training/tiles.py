"""
training/tiles.py
==================
Builds the training tile cache from real Chandrayaan-2 PDS4 products.

Why a cache at all
------------------
Every learned module in this pipeline is trained *in the exact domain it sees
at inference* -- the de-quantizer on VST-stabilized radiance, Zero-DCE on the
normalized LL band, the detail restorer on symmetric-normalized SWT detail
bands. Producing those domains means running the real stage-1 physics for every
tile, which is dominated by the cosmic-ray scrubber's median filters (~90 ms per
192 px tile). Doing that inside the training loop would spend most of an epoch
in scipy instead of in the optimizer, so stage 1 is run once, up front, and its
output is what the loop consumes.

The consequence that matters scientifically: the training distribution is not an
approximation of the inference distribution, it is byte-identical to it, because
the same `PhysicsFrontend` object produced both.

The noise pair
--------------
There is no clean ground truth for a lunar pushbroom strip, so each tile is
cached twice:

    stab_clean   the measured scene, stage-1 processed
    stab_noisy   the same scene with one extra sensor-model noise realization

Training the restorer to map noisy -> measured (Noisier2Noise, Moran et al.
2020) anchors its target to data that was actually recorded. The extra noise is
drawn in the *electron* domain from the config's own gain and read-noise
constants, so the augmentation is the sensor's noise model rather than a
convenient-looking Gaussian:

    e_noisy = Poisson(e / k) * k + N(0, read_noise_e)

which has mean e and variance k*e + read_noise^2 -- k = 1 reproduces the
sensor's natural shot noise, k > 1 makes the input progressively harder while
leaving the target untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Cache layout. `_PLANES` is the channel order inside prepared.npy and is
# written into the manifest so a cache can never be silently misread by a
# later version that reordered them.
_PLANES = ("stab_clean", "stab_noisy", "rad_clean")
MANIFEST_VERSION = 2


# =============================================================================
# Window sampling
# =============================================================================
@dataclass
class TileRef:
    """Where one cached tile came from, so any sample is traceable to a pixel."""

    product: str
    label_path: str
    row: int
    col: int
    size: int
    processing_level: str
    sensor: str
    instrument: str
    incidence_deg: Optional[float]
    mean_dn: float
    std_dn: float


def _stratified_rows(n: int, height: int, size: int, rng: np.random.Generator
                     ) -> np.ndarray:
    """
    One row offset per along-track band, rather than n uniform draws.

    A TMC strip is ~245 000 lines of continuously changing terrain and slowly
    changing illumination. Uniform sampling clumps by chance and can leave whole
    latitude ranges unrepresented; stratifying guarantees the cache spans the
    strip, which is what stops the models from over-fitting one type of terrain.
    """
    limit = max(height - size, 0)
    edges = np.linspace(0, limit, n + 1)
    return (edges[:-1] + rng.random(n) * np.diff(edges)).astype(np.int64)


def sample_windows(height: int, width: int, count: int, size: int,
                   rng: np.random.Generator) -> list[tuple[int, int]]:
    rows = _stratified_rows(count, height, size, rng)
    cols = rng.integers(0, max(width - size, 0) + 1, size=count)
    return list(zip(rows.tolist(), cols.tolist()))


def tile_is_usable(tile: np.ndarray, nodata_value: Optional[float],
                   max_nodata_fraction: float = 0.01,
                   min_std: float = 0.75) -> tuple[bool, str]:
    """
    Rejects windows that would teach the models something false.

    Padding/fill regions (a pushbroom strip is ragged at its ends) carry no
    terrain, and a flat tile has no structure to preserve -- training on either
    pushes the restorer toward outputting a constant, which is the failure mode
    that later reads as smoothed-away detail.
    """
    if nodata_value is not None:
        if float(np.mean(tile == nodata_value)) > max_nodata_fraction:
            return False, "nodata"
    if float(tile.std()) < min_std:
        return False, "flat"
    return True, "ok"


# =============================================================================
# Noise augmentation (sensor model, not a convenient Gaussian)
# =============================================================================
def add_sensor_noise(dn: np.ndarray, gain_e_per_dn: float, read_noise_e: float,
                     bias_dn: float, black_dn: float, boost: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Adds one extra realization of the sensor's own Poisson-Gaussian noise.

    Works in electrons, because that is the domain the Poisson model is valid
    in -- the same reason stage 1.3 scrubs there. Returns DN, clipped at zero
    but deliberately not at any upper bound: clipping the bright tail would
    hand the restorer a biased target near saturation.
    """
    electrons = np.clip((dn.astype(np.float64) - bias_dn - black_dn)
                        * gain_e_per_dn, 0.0, None)
    k = max(float(boost), 1e-6)
    noisy_e = rng.poisson(electrons / k).astype(np.float64) * k
    noisy_e += rng.normal(0.0, read_noise_e, size=electrons.shape)
    return np.clip(noisy_e / gain_e_per_dn + bias_dn + black_dn, 0.0, None)


# =============================================================================
# Cache construction
# =============================================================================
def build_cache(products: Iterable, out_dir: str | Path, config: dict,
                frontend, tiles_per_product: int = 48, tile_size: int = 192,
                noise_boost: float = 2.0, seed: int = 42,
                max_attempts_factor: int = 6, verbose: bool = True) -> dict:
    """
    Samples windows from every product, runs stage 1 on each twice (measured and
    noise-augmented), and writes the cache.

    Args:
        products:  `Pds4Product` objects from `pds4_bundle.discover_products`.
        frontend:  a configured `PhysicsFrontend` -- the *same class* the
                   pipeline uses, so the cached domain is the inference domain.
        max_attempts_factor: how many windows to try per accepted tile before
                   giving up on a product. Ragged strips reject a lot.

    Writes:
        prepared.npy   (N, 3, S, S) float32 -- see `_PLANES`
        masks.npy      (N, 2, S, S) uint8   -- saturation, cosmic-ray
        raw.npy        (N, S, S)    uint16  -- measured DN, for inspection
        manifest.json  provenance for every tile
    """
    import rasterio
    from rasterio.windows import Window

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cal = config["calibration"]
    gain = float(cal["gain_e_per_dn"])
    read_noise = float(cal["read_noise_e"])
    bias = float(cal["bias_offset_dn"])
    black = float(cal["black_level_dn"])
    nodata = config["io"].get("nodata_value")

    prepared: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    refs: list[TileRef] = []
    rejected = {"nodata": 0, "flat": 0, "read_error": 0}
    started = time.perf_counter()

    products = list(products)
    for index, product in enumerate(products, start=1):
        if not product.readable:
            continue
        accepted = 0
        attempts = tiles_per_product * max_attempts_factor
        candidates = sample_windows(product.lines, product.samples,
                                    attempts, tile_size, rng)

        with rasterio.open(product.label_path) as src:
            for row, col in candidates:
                if accepted >= tiles_per_product:
                    break
                try:
                    tile = src.read(1, window=Window(col, row, tile_size, tile_size))
                except Exception:
                    rejected["read_error"] += 1
                    continue
                if tile.shape != (tile_size, tile_size):
                    rejected["read_error"] += 1
                    continue

                ok, reason = tile_is_usable(tile, nodata)
                if not ok:
                    rejected[reason] += 1
                    continue

                dn_clean = tile.astype(np.float64)
                dn_noisy = add_sensor_noise(dn_clean, gain, read_noise, bias,
                                            black, noise_boost, rng)

                # The one place the inference domain is defined: same object,
                # same config, same code path the dashboard will run.
                fe_clean = frontend.process(_scene(dn_clean))
                fe_noisy = frontend.process(_scene(dn_noisy))

                prepared.append(np.stack([
                    fe_clean.stabilized.astype(np.float32),
                    fe_noisy.stabilized.astype(np.float32),
                    fe_clean.radiance_linear.astype(np.float32),
                ]))
                masks.append(np.stack([
                    fe_clean.saturation_mask.astype(np.uint8),
                    fe_clean.cosmic_ray_mask.astype(np.uint8),
                ]))
                raws.append(tile.astype(np.uint16))
                refs.append(TileRef(
                    product=product.name, label_path=str(product.label_path),
                    row=int(row), col=int(col), size=tile_size,
                    processing_level=product.processing_level,
                    sensor=product.sensor, instrument=product.instrument.upper(),
                    incidence_deg=product.incidence_deg,
                    mean_dn=round(float(tile.mean()), 2),
                    std_dn=round(float(tile.std()), 3),
                ))
                accepted += 1

        if verbose:
            print(f"  [{index:2d}/{len(products)}] {product.name[:44]:46s} "
                  f"{accepted:3d} tiles  ({time.perf_counter() - started:5.1f}s)")

    if not prepared:
        raise RuntimeError(
            "No usable tiles. Every sampled window was nodata or flat -- check "
            "that the products are readable and that io.nodata_value is right."
        )

    np.save(out_dir / "prepared.npy", np.stack(prepared).astype(np.float32))
    np.save(out_dir / "masks.npy", np.stack(masks).astype(np.uint8))
    np.save(out_dir / "raw.npy", np.stack(raws).astype(np.uint16))

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "planes": list(_PLANES),
        "tile_size": tile_size,
        "tile_count": len(refs),
        "tiles_per_product": tiles_per_product,
        "noise_boost": noise_boost,
        "seed": seed,
        "rejected": rejected,
        "build_seconds": round(time.perf_counter() - started, 1),
        "calibration": {"gain_e_per_dn": gain, "read_noise_e": read_noise,
                        "bias_offset_dn": bias, "black_level_dn": black},
        "products": sorted({r.product for r in refs}),
        "tiles": [asdict(r) for r in refs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    return manifest


def _scene(dn: np.ndarray):
    """Wraps a bare array as an IngestedScene, bypassing file I/O."""
    from models.physics_frontend import IngestedScene
    return IngestedScene(dn=dn, source_format="tile")


# =============================================================================
# Loading
# =============================================================================
class TileCache:
    """
    Read-only view over a built cache.

    `prepared.npy` is memory-mapped: at 192 px and a few thousand tiles it is
    several hundred MB, and mapping it keeps training's resident set flat
    whatever the cache size.
    """

    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No tile cache at {self.dir}. Build one first:\n"
                f"    python train.py prepare --data <bundle folder>"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        planes = tuple(self.manifest.get("planes", ()))
        if planes != _PLANES:
            raise ValueError(
                f"Cache plane order {planes} does not match this version's "
                f"{_PLANES}. Rebuild the cache."
            )

        self.prepared = np.load(self.dir / "prepared.npy", mmap_mode="r")
        self.masks = np.load(self.dir / "masks.npy", mmap_mode="r")
        self.tile_size = int(self.manifest["tile_size"])

    def __len__(self) -> int:
        return int(self.prepared.shape[0])

    def plane(self, index: int, name: str) -> np.ndarray:
        return np.array(self.prepared[index, _PLANES.index(name)], dtype=np.float32)

    def batch(self, indices: np.ndarray, name: str) -> np.ndarray:
        """(B, 1, S, S) float32 -- the shape every module here expects."""
        p = _PLANES.index(name)
        return np.asarray(self.prepared[indices, p], dtype=np.float32)[:, None]

    def mask_batch(self, indices: np.ndarray, which: int) -> np.ndarray:
        return np.asarray(self.masks[indices, which], dtype=np.float32)[:, None]

    def split(self, val_fraction: float = 0.1, seed: int = 42
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Held-out split by *product*, not by tile.

        Tiles from one strip overlap in terrain and illumination, so a random
        tile split would put near-duplicates on both sides and report a
        validation loss that flatters the model. Splitting by product keeps the
        validation scenes genuinely unseen.
        """
        tiles = self.manifest["tiles"]
        products = sorted({t["product"] for t in tiles})
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(products) * val_fraction)))
        val_products = set(rng.permutation(products)[:n_val].tolist())

        val = np.array([i for i, t in enumerate(tiles)
                        if t["product"] in val_products], dtype=np.int64)
        train = np.array([i for i, t in enumerate(tiles)
                          if t["product"] not in val_products], dtype=np.int64)
        return train, val
