"""
pds4_bundle.py
===============
PDS4 bundle discovery and label parsing for Chandrayaan-2 ISSDC products.

An ISSDC download is a *directory tree*, not a file:

    ch2_tmc_ncn_20260810T0544521099_d_img_d18/
      browse/calibrated/20260810/..._b_brw_d18.png     <- 8-bit quicklook
      data/calibrated/20260810/..._d_img_d18.img       <- the actual raster
                               ..._d_img_d18.xml       <- its detached label
      geometry/calibrated/20260810/..._g_grd_d18.csv   <- lat/lon grid
      miscellaneous/calibrated/20260810/*.oat|.spm|.lbr

Only one of those files is the science product, and it is the `.xml` label --
the `.img` beside it is a headerless binary whose shape, dtype and byte order
exist nowhere else. Hand a user's extracted folder to `discover_products` and
it finds the label, skipping the quicklook PNG that would otherwise look like
a perfectly good image and silently get enhanced instead of the real data.

Two things this module exists to prevent
----------------------------------------
1. *Processing the browse PNG.* It is an 8-bit, contrast-stretched, already
   lossy rendering. Enhancing it and reporting the metrics as if they described
   the science product would be a fabricated result.

2. *Applying another scene's illumination geometry.* `solar_incidence` varies
   from product to product (39.06 deg in one of these scenes, 41.40 deg in
   another). The Lommel-Seeliger stage divides by cos(i)/(cos(i)+cos(e)), so a
   wrong incidence angle is a wrong radiometric normalization applied to real
   data. `label_metadata` lifts the per-product angles out of the label into
   the tags the photometric stage already knows how to read, which also makes
   it report `incidence_source: product_metadata` instead of `config_default`.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree

# PDS4 namespaces. `isda` is ISSDC's mission-specific dictionary and is where
# every Chandrayaan-2 observing-geometry parameter actually lives.
NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "https://isda.issdc.gov.in/pds4/isda/v1",
}

LABEL_SUFFIXES = {".xml", ".lbl"}

# Sub-trees of an ISSDC bundle that never contain the science raster. `browse`
# is the trap: it holds a real, openable PNG of the same scene.
NON_DATA_DIRS = {"browse", "geometry", "miscellaneous", "document", "context"}

# ch2_<instrument>_<n><level><sensor>_<timestamp>_<kind>_<...>
#   level : c = calibrated, r = raw
#   sensor: n = nadir, f = fore, a = aft, p = OHRC pan
PRODUCT_RE = re.compile(
    r"^ch2_(?P<instrument>[a-z]{3})_(?P<code>[a-z]{3})_(?P<timestamp>\d{8}T\d+)", re.I
)

SENSOR_NAMES = {"n": "Nadir", "f": "Fore", "a": "Aft", "p": "Pan"}
LEVEL_NAMES = {"c": "Calibrated", "r": "Raw"}

# TMC-2 carries three sensors on fixed mounts. The label states the geometry in
# prose ("Fore, Nadir and Aft placed with angles of +25 deg, 0 deg and -25 deg")
# but gives no numeric emission-angle element, so the boresight tilt is taken
# from that statement rather than measured per product. `label_metadata` records
# EMISSION_ANGLE_SOURCE alongside it so the provenance stays visible.
SENSOR_BORESIGHT_DEG = {"Nadir": 0.0, "Fore": 25.0, "Aft": 25.0, "Pan": 0.0}


# =============================================================================
# Parsed product
# =============================================================================
@dataclass
class Pds4Product:
    """One observational product: its label, its raster, and what they say."""

    label_path: Path
    data_path: Optional[Path] = None
    lines: int = 0                       # rows (along-track for a pushbroom)
    samples: int = 0                     # columns (across-track)
    data_type: str = ""                  # PDS4 element type, e.g. UnsignedLSB2
    processing_level: str = ""           # Raw | Calibrated
    instrument: str = ""                 # tmc | ohr
    sensor: str = ""                     # Nadir | Fore | Aft | Pan
    start_time: str = ""
    incidence_deg: Optional[float] = None
    emission_deg: Optional[float] = None
    line_exposure_ms: Optional[float] = None
    pixel_resolution_m: Optional[float] = None
    spacecraft_altitude_km: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    orbit_number: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def megapixels(self) -> float:
        return self.lines * self.samples / 1e6

    @property
    def name(self) -> str:
        return self.label_path.stem

    @property
    def readable(self) -> bool:
        """A label whose raster is missing cannot be opened by any reader."""
        return self.data_path is not None and self.data_path.is_file()

    def label_metadata(self) -> dict:
        """
        Label facts rendered as raster tags, in the spelling the pipeline's
        photometric stage looks for (`photometric.incidence_key` /
        `emission_key`, default INCIDENCE_ANGLE / EMISSION_ANGLE).

        Returned as strings because that is what `src.tags()` yields for a real
        raster, and the consumer parses with `float(str(v).split()[0])`.
        """
        meta: dict[str, str] = {}
        if self.incidence_deg is not None:
            meta["INCIDENCE_ANGLE"] = f"{self.incidence_deg:.6f}"
            meta["INCIDENCE_ANGLE_SOURCE"] = "pds4_label:isda:solar_incidence"
        if self.emission_deg is not None:
            meta["EMISSION_ANGLE"] = f"{self.emission_deg:.6f}"
            meta["EMISSION_ANGLE_SOURCE"] = (
                f"pds4_label:sensor_boresight({self.sensor})+roll/pitch"
            )
        if self.line_exposure_ms is not None:
            meta["LINE_EXPOSURE_DURATION_MS"] = f"{self.line_exposure_ms:.6f}"
        if self.pixel_resolution_m is not None:
            meta["PIXEL_RESOLUTION_M"] = f"{self.pixel_resolution_m:.4f}"
        if self.sun_elevation_deg is not None:
            meta["SUN_ELEVATION"] = f"{self.sun_elevation_deg:.6f}"
        meta.update({
            "PDS4_PROCESSING_LEVEL": self.processing_level,
            "PDS4_SENSOR": self.sensor,
            "PDS4_INSTRUMENT": self.instrument.upper(),
            "PDS4_START_TIME": self.start_time,
            "PDS4_LABEL": self.label_path.name,
        })
        return {k: v for k, v in meta.items() if v}

    def summary(self) -> dict:
        """JSON-safe description, for the API and the batch manifest."""
        return {
            "name": self.name,
            "label_path": str(self.label_path),
            "data_path": str(self.data_path) if self.data_path else None,
            "lines": self.lines,
            "samples": self.samples,
            "megapixels": round(self.megapixels, 1),
            "data_type": self.data_type,
            "processing_level": self.processing_level,
            "instrument": self.instrument.upper(),
            "sensor": self.sensor,
            "start_time": self.start_time,
            "incidence_deg": self.incidence_deg,
            "emission_deg": self.emission_deg,
            "line_exposure_ms": self.line_exposure_ms,
            "pixel_resolution_m": self.pixel_resolution_m,
            "orbit_number": self.orbit_number,
            "readable": self.readable,
            "warnings": self.warnings,
        }


# =============================================================================
# Label parsing
# =============================================================================
def _text(root: ElementTree.Element, path: str) -> Optional[str]:
    node = root.find(path, NS)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def _float(root: ElementTree.Element, path: str) -> Optional[float]:
    raw = _text(root, path)
    if raw is None:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def _axis_lengths(root: ElementTree.Element) -> tuple[int, int]:
    """
    Reads the Array_2D_Image axes by *name* rather than by document order.

    PDS4 orders axes by `sequence_number`, and the fast axis is declared
    separately by `axis_index_order`. Trusting position would silently
    transpose a product whose label happens to list Sample first.
    """
    lines = samples = 0
    for axis in root.iter(f"{{{NS['pds']}}}Axis_Array"):
        name = (axis.findtext(f"{{{NS['pds']}}}axis_name") or "").strip().lower()
        try:
            elements = int((axis.findtext(f"{{{NS['pds']}}}elements") or "0").strip())
        except ValueError:
            continue
        if name == "line":
            lines = elements
        elif name == "sample":
            samples = elements
    return lines, samples


def parse_label(label_path: str | Path) -> Optional[Pds4Product]:
    """
    Parses one PDS4 observational label. Returns None when the file is not a
    `Product_Observational` (bundle/collection labels and the browse, geometry
    and miscellaneous labels all parse fine as XML but describe other things).
    """
    label_path = Path(label_path)
    try:
        root = ElementTree.parse(label_path).getroot()
    except (ElementTree.ParseError, OSError):
        return None

    if not root.tag.endswith("Product_Observational"):
        return None

    product = Pds4Product(label_path=label_path)

    # -- the raster this label describes -----------------------------------
    file_name = _text(root, ".//pds:File_Area_Observational/pds:File/pds:file_name")
    if file_name:
        candidate = label_path.parent / file_name
        product.data_path = candidate
        if not candidate.is_file():
            product.warnings.append(
                f"Label references '{file_name}', which is not in the same "
                "directory. The .img must sit beside its .xml."
            )
    else:
        product.warnings.append("Label declares no File_Area_Observational file name.")

    product.lines, product.samples = _axis_lengths(root)
    product.data_type = _text(
        root, ".//pds:Element_Array/pds:data_type") or ""
    product.processing_level = _text(
        root, ".//pds:Primary_Result_Summary/pds:processing_level") or ""
    product.start_time = _text(
        root, ".//pds:Time_Coordinates/pds:start_date_time") or ""

    # -- ISSDC mission-area geometry ---------------------------------------
    product.incidence_deg = _float(root, ".//isda:solar_incidence")
    product.line_exposure_ms = _float(root, ".//isda:line_exposure_duration")
    product.pixel_resolution_m = _float(root, ".//isda:pixel_resolution")
    product.spacecraft_altitude_km = _float(root, ".//isda:spacecraft_altitude")
    product.sun_elevation_deg = _float(root, ".//isda:sun_elevation")
    product.orbit_number = _text(root, ".//isda:imaging_orbit_number")

    # -- identity from the file name, corrected by the label ---------------
    match = PRODUCT_RE.match(label_path.stem)
    if match:
        product.instrument = match.group("instrument").lower()
        code = match.group("code").lower()
        product.sensor = SENSOR_NAMES.get(code[2], "")
        if not product.processing_level:
            product.processing_level = LEVEL_NAMES.get(code[1], "")

    # Emission angle: boresight tilt for this sensor, plus the off-nadir
    # pointing the label does state. Never guessed for an unknown sensor --
    # left None so the config fallback applies and says so.
    if product.sensor in SENSOR_BORESIGHT_DEG:
        roll = _float(root, ".//isda:roll") or 0.0
        pitch = _float(root, ".//isda:pitch") or 0.0
        off_nadir = max(abs(roll), abs(pitch))
        product.emission_deg = SENSOR_BORESIGHT_DEG[product.sensor] + off_nadir

    return product


# =============================================================================
# Discovery
# =============================================================================
def _is_data_label(path: Path, root: Path) -> bool:
    """
    Excludes labels living under a non-data sub-tree. Checked on the path
    *relative to the bundle root* so a user whose extraction directory happens
    to be named 'browse' is not locked out of their own data.
    """
    try:
        parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
    except ValueError:
        parts = {p.lower() for p in path.parts[:-1]}
    return not (parts & NON_DATA_DIRS)


def _rank(product: Pds4Product) -> tuple:
    """
    Sort key for `choose_best`, most preferred first.

    Calibrated over raw, because the raw product has had no radiometric look-up
    table applied and the pipeline's calibration constants are already an
    approximation. Nadir over fore/aft, because at 25 degrees off-boresight the
    Lommel-Seeliger correction is doing considerably more work. Larger over
    smaller only as a final tie-break.
    """
    level_rank = {"calibrated": 0, "raw": 1}.get(product.processing_level.lower(), 2)
    sensor_rank = {"Nadir": 0, "Pan": 1, "Fore": 2, "Aft": 3}.get(product.sensor, 4)
    return (0 if product.readable else 1, level_rank, sensor_rank, -product.megapixels)


def discover_products(root: str | Path, include_non_data: bool = False
                      ) -> list[Pds4Product]:
    """
    Walks an extracted bundle (or any directory) and returns every readable
    observational product, best first.

    `include_non_data=True` lifts the browse/geometry/miscellaneous exclusion.
    It exists for inspection, not for processing.
    """
    root = Path(root)
    if root.is_file():
        product = parse_label(root)
        return [product] if product else []

    products: list[Pds4Product] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in LABEL_SUFFIXES or not path.is_file():
            continue
        if not include_non_data and not _is_data_label(path, root):
            continue
        product = parse_label(path)
        if product is not None:
            products.append(product)

    return sorted(products, key=_rank)


def choose_best(products: Iterable[Pds4Product]) -> Optional[Pds4Product]:
    """The product a bundle most plausibly 'is'. None if nothing is readable."""
    readable = [p for p in products if p.readable]
    return sorted(readable, key=_rank)[0] if readable else None


# =============================================================================
# Archive extraction
# =============================================================================
def extract_archive(archive_path: str | Path, dest_dir: str | Path) -> Path:
    """
    Extracts a bundle ZIP into `dest_dir` and returns the directory to search.

    Entries are filtered rather than trusted: absolute paths and `..` segments
    are dropped instead of being written outside `dest_dir`. Python's own
    `extractall` sanitizes since 3.12 via `filter="data"`, but this code has to
    behave identically on the interpreter the user actually has, and a path
    traversal here would write attacker-chosen bytes anywhere the dashboard
    process can reach.
    """
    archive_path, dest_dir = Path(archive_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest_dir.resolve()

    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/")
                     if p not in ("", ".", "..") and not Path(p).is_absolute()]
            if not parts:
                continue
            target = (dest_dir / Path(*parts)).resolve()
            if not str(target).startswith(str(resolved_dest)):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)

    # A bundle usually zips with its own folder as the single top-level entry;
    # descend through those so callers see the bundle root, not its wrapper.
    current = dest_dir
    while True:
        entries = [p for p in current.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
        else:
            return current


def resolve_input(path: str | Path, extract_dir: Optional[str | Path] = None
                  ) -> tuple[Optional[Pds4Product], list[Pds4Product], Path]:
    """
    Turns whatever the user pointed at -- a label, a bundle directory, or a
    bundle ZIP -- into (chosen product, all products found, search root).

    A ZIP needs `extract_dir`; without one it is reported as unresolvable
    rather than being written somewhere the caller did not choose.
    """
    path = Path(path)

    if path.is_file() and path.suffix.lower() == ".zip":
        if extract_dir is None:
            raise ValueError("A bundle ZIP needs an extraction directory.")
        path = extract_archive(path, extract_dir)

    products = discover_products(path)
    return choose_best(products), products, path
