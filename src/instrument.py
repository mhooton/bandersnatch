"""Instrument abstraction.

Everything the pipeline knows about a particular camera lives here, driven by
``configs/{INSTRUMENT}.yaml``.  This is the only module that opens a raw FITS
file: every other stage receives a :class:`Frame`, which carries a 2-D float
array for the selected detector plus normalised metadata.

Adding an instrument should mean writing one YAML file.  See
``docs/instrument_abstraction.md`` for the schema and the reasoning.

Every new configuration key has a default equal to the pipeline's historical
behaviour, so instrument configs written before this module existed keep
working untouched.
"""

from __future__ import annotations

import glob
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from astropy.io import fits

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# frame types
# --------------------------------------------------------------------------

class FrameType(str, Enum):
    BIAS = "bias"
    DARK = "dark"
    FLAT = "flat"
    FRINGE = "fringe"
    SCIENCE = "science"
    IGNORE = "ignore"


#: Frame classification used before this module existed: substring match on
#: IMAGETYP.  Reproduces ``create_lists.classify_file`` exactly.
LEGACY_CLASSIFICATION = [
    {"type": "science", "where": {"IMAGETYP": {"contains": "LIGHT"}}},
    {"type": "dark", "where": {"IMAGETYP": {"contains": "DARK"}}},
    {"type": "bias", "where": {"IMAGETYP": {"contains": "BIAS"}}},
    {"type": "flat", "where": {"IMAGETYP": {"contains": "FLAT"}}},
]

LEGACY_KEYWORDS = {
    "exptime": "EXPTIME",
    "object": "OBJECT",
    "filter": "FILTER",
    "airmass": {"keywords": ["AIRMASS"]},
    "altitude": {"keywords": ["ALTITUDE"]},
}

#: SPECULOOS frames carry BJD-OBS already barycentred, so the default is a
#: pass-through and the recorded time is bit-identical to the legacy pipeline.
LEGACY_TIME = {
    "keyword": "BJD-OBS",
    "format": "jd",
    "scale": "tdb",
    "reference": "mid",
    "barycentric": True,
}

LEGACY_RAW_DIR = "{topdir}/Observations/{inst}/images/{date}"
LEGACY_STEPS = ["bias", "dark", "flat"]

VALID_STEPS = ("overscan", "bias", "dark", "flat", "bpm", "fringe")


# --------------------------------------------------------------------------
# FITS section handling
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^\s*\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")


def parse_fits_section(section: str) -> tuple[slice, slice]:
    """Convert an IRAF/FITS section string into numpy ``(row, col)`` slices.

    ``"[54:2101,1:4096]"`` is 1-based and inclusive on both ends, x first.
    numpy wants 0-based, exclusive-stop, rows first, so that becomes
    ``(slice(0, 4096), slice(53, 2101))``.

    Trailing text after the closing bracket is ignored, because INT/WFC writes
    things like ``"[0:0,0:0], disabled"``.
    """
    if not isinstance(section, str):
        raise ValueError(f"FITS section must be a string, got {section!r}")
    m = _SECTION_RE.match(section)
    if not m:
        raise ValueError(f"Cannot parse FITS section {section!r}")
    x1, x2, y1, y2 = (int(g) for g in m.groups())
    if x2 < x1 or y2 < y1:
        raise ValueError(f"FITS section {section!r} has a reversed range")
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


# --------------------------------------------------------------------------
# keyword resolution
# --------------------------------------------------------------------------

_EXPR_ALLOWED = {
    "abs": abs, "min": min, "max": max, "round": round,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")


def _eval_expr(expr: str, header) -> Any:
    """Evaluate a small arithmetic expression over header values.

    Only header keywords, a handful of maths helpers and arithmetic are
    available.  Builtins are removed.  This exists so that, for example, an
    instrument with no ALTITUDE keyword can declare
    ``altitude: {expr: "90 - 0.5*(ZDSTART + ZDEND)"}`` in YAML rather than
    requiring a code branch.
    """
    names = {}
    for tok in set(_IDENT_RE.findall(expr)):
        if tok in _EXPR_ALLOWED:
            continue
        if tok in header:
            names[tok] = header[tok]
    safe = dict(_EXPR_ALLOWED)
    safe.update(names)
    # Header keywords may contain '-', which is not a Python identifier.
    # Substitute those before evaluating.
    prepared = expr
    for key in sorted((k for k in header.keys() if "-" in k), key=len, reverse=True):
        if key in prepared:
            alias = "_kw_" + key.replace("-", "_")
            safe[alias] = header[key]
            prepared = prepared.replace(key, alias)
    try:
        return eval(prepared, {"__builtins__": {}}, safe)  # noqa: S307
    except Exception as exc:
        raise KeyError(f"Could not evaluate expression {expr!r}: {exc}") from exc


def resolve_keyword(spec, header, default=None, required=False, name=""):
    """Resolve one metadata item from a header given its YAML specification.

    ``spec`` may be:
      * a string, naming a single keyword;
      * ``{"keyword": NAME}`` (``strip`` is accepted and ignored: string
        values are always stripped, because FITS pads them);
      * ``{"keywords": [NAME, ...]}``, first present wins;
      * ``{"expr": "..."}``, arithmetic over header values.
    """
    value = None

    if spec is None:
        value = None
    elif isinstance(spec, str):
        value = header.get(spec)
    elif isinstance(spec, dict):
        if "expr" in spec:
            try:
                value = _eval_expr(spec["expr"], header)
            except KeyError:
                value = None
        elif "keywords" in spec:
            for kw in spec["keywords"]:
                if kw in header:
                    value = header[kw]
                    break
        elif "keyword" in spec:
            value = header.get(spec["keyword"])
        else:
            raise ValueError(f"Unrecognised keyword spec for {name!r}: {spec!r}")
    else:
        raise ValueError(f"Unrecognised keyword spec for {name!r}: {spec!r}")

    if value is None:
        if required:
            raise KeyError(
                f"Required metadata {name!r} not found in header "
                f"(specification: {spec!r})")
        return default
    if isinstance(value, str):
        # Always stripped: FITS pads string values to 8 characters, so
        # 'z       ' and 'z' must compare equal. `strip` is accepted for
        # explicitness in the YAML but changes nothing.
        value = value.strip()
    return value


# --------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------

@dataclass
class Detector:
    """One readout of one instrument: where the pixels are and what they mean."""
    name: str
    hdu: Any = 0                       # int index or EXTNAME string
    trim: Optional[str] = None         # FITS section of the illuminated area
    overscan: Optional[str] = None     # FITS section of the overscan strip
    gain: float = 1.0                  # electrons per ADU
    read_noise: Optional[float] = None  # electrons
    phpadu: float = 1.0
    saturation: Optional[float] = None
    bad_columns: list = field(default_factory=list)

    @property
    def trim_slices(self):
        return parse_fits_section(self.trim) if self.trim else None

    @property
    def overscan_slices(self):
        return parse_fits_section(self.overscan) if self.overscan else None

    @property
    def trim_offset(self) -> tuple[int, int]:
        """``(dx, dy)`` to subtract from a raw position to get a trimmed one."""
        if not self.trim:
            return (0, 0)
        rows, cols = self.trim_slices
        return (cols.start, rows.start)


@dataclass
class FrameMeta:
    path: Path
    frame_type: FrameType
    object: str
    filter: str
    exptime: float
    bjd_mid: float
    airmass: float
    altitude: Optional[float]
    time_start: Any = None
    header: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class Frame:
    meta: FrameMeta
    data: np.ndarray                  # full, untrimmed array for the detector


@dataclass
class FringeParams:
    highpass_sigma: float = 120.0
    template_smooth_sigma: float = 2.0
    star_mask: dict = field(default_factory=lambda: {
        "smooth": 3, "threshold_mad": 4.0, "dilate": 12, "border": 40})
    fit: dict = field(default_factory=lambda: {
        "plane": True, "clip_sigma": 3.0, "iterations": 3, "subsample": 7})


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------

class Instrument:
    """Everything instrument-specific, assembled from one YAML file."""

    def __init__(self, config: dict, name: Optional[str] = None,
                 detector_name: Optional[str] = None):
        self.config = config or {}
        self.name = name or self.config.get("instrument_name", "UNKNOWN")

        # --- detectors -----------------------------------------------------
        det_cfg = self.config.get("detectors")
        if not det_cfg:
            # Legacy single-detector instrument: the top-level keys describe it.
            det_cfg = {"default": {
                "hdu": self.config.get("image_extension", 0),
                "gain": self.config.get("gain", 1.0),
                "phpadu": self.config.get("phpadu", 1.0),
                "saturation": self.config.get("saturation_threshold"),
            }}
            self._legacy_detector = True
        else:
            self._legacy_detector = False

        self.detectors = {}
        for dname, d in det_cfg.items():
            d = dict(d or {})
            self.detectors[dname] = Detector(
                name=dname,
                hdu=d.get("hdu", 0),
                trim=d.get("trim"),
                overscan=d.get("overscan"),
                gain=float(d.get("gain", self.config.get("gain", 1.0))),
                read_noise=(float(d["read_noise"])
                            if d.get("read_noise") is not None else None),
                phpadu=float(d.get("phpadu", self.config.get("phpadu", 1.0))),
                saturation=(float(d["saturation"])
                            if d.get("saturation") is not None else None),
                bad_columns=list(d.get("bad_columns", []) or []),
            )

        chosen = (detector_name or self.config.get("default_detector")
                  or next(iter(self.detectors)))
        if chosen not in self.detectors:
            raise KeyError(
                f"{self.name}: detector {chosen!r} not defined. "
                f"Available: {sorted(self.detectors)}")
        self.detector = self.detectors[chosen]

        # --- discovery -----------------------------------------------------
        self.raw_dir_template = self.config.get("raw_dir", LEGACY_RAW_DIR)
        self.file_patterns = list(self.config.get(
            "file_patterns", ["*.fits", "*.fits.fz", "*.fts", "*.fit"]))
        self.search_subdirectories = bool(
            self.config.get("search_subdirectories", True))

        # --- headers and metadata ------------------------------------------
        self.header_hdus = list(self.config.get("header_hdus", [0]))
        self.keywords = dict(LEGACY_KEYWORDS)
        self.keywords.update(self.config.get("keywords", {}) or {})
        # Legacy defaults apply only when the config declares no time block.
        # Merging them would leak 'barycentric: true' into an instrument that
        # records a plain UTC timestamp, silently skipping the correction.
        if self.config.get("time"):
            self.time_cfg = {"format": "jd", "scale": "utc", "reference": "mid",
                             "barycentric": False}
            self.time_cfg.update(self.config["time"])
        else:
            self.time_cfg = dict(LEGACY_TIME)
        self.classification = list(
            self.config.get("classification", LEGACY_CLASSIFICATION))
        self.dome_flat_rule = self.config.get("dome_flat")
        self.flat_sources = self.config.get("flat_sources")

        # --- reduction -----------------------------------------------------
        red = self.config.get("reduction", {}) or {}
        self.steps = list(red.get("steps", LEGACY_STEPS))
        self.overscan_cfg = dict(red.get("overscan", {"method": "median"}))
        self.validate_steps()

        fr = self.config.get("fringe", {}) or {}
        self.fringe = FringeParams(
            highpass_sigma=float(fr.get("highpass_sigma", 120.0)),
            template_smooth_sigma=float(fr.get("template_smooth_sigma", 2.0)),
            star_mask={**FringeParams().star_mask, **(fr.get("star_mask") or {})},
            fit={**FringeParams().fit, **(fr.get("fit") or {})},
        )

        # --- coordinates ----------------------------------------------------
        self.coordinates = str(self.config.get("coordinates", "trimmed")).lower()
        if self.coordinates not in ("raw", "trimmed"):
            raise ValueError(
                f"{self.name}: coordinates must be 'raw' or 'trimmed', "
                f"got {self.coordinates!r}")

        # --- site -----------------------------------------------------------
        self.observatory_cfg = self.config.get("observatory")
        self.plate_scale = self.config.get("plate_scale")
        self._location = None

    # -- convenience -------------------------------------------------------

    def legacy_config(self) -> dict:
        """The instrument config with per-detector values resolved.

        Modules that predate this class read ``config['instrument_config']``
        and expect flat ``gain``, ``phpadu``, ``read_noise`` keys. A
        multi-detector instrument has no single value for those, so the
        selected detector's are filled in here. Without this a WFC run reaches
        the analysis stage and fails on a missing 'gain'.
        """
        merged = dict(self.config)
        merged.update(
            instrument_name=self.name,
            detector=self.detector.name,
            gain=self.detector.gain,
            phpadu=self.detector.phpadu,
        )
        if self.detector.read_noise is not None:
            merged["read_noise"] = self.detector.read_noise
        if self.detector.saturation is not None:
            merged["saturation_threshold"] = self.detector.saturation
        if self.plate_scale is not None:
            merged["plate_scale"] = self.plate_scale
        if self.observatory_cfg and "observatory_altitude" not in merged:
            merged["observatory_altitude"] = self.observatory_cfg.get("height", 0.0)
        return merged

    def gain(self) -> float:
        return self.detector.gain

    def phpadu(self) -> float:
        return self.detector.phpadu

    def read_noise(self) -> Optional[float]:
        return self.detector.read_noise

    def saturation(self) -> Optional[float]:
        return self.detector.saturation

    @property
    def location(self):
        """Observatory position, or ``None`` when not configured."""
        if self._location is None and self.observatory_cfg:
            from astropy.coordinates import EarthLocation
            import astropy.units as u
            o = self.observatory_cfg
            self._location = EarthLocation(
                lat=o["lat"] * u.deg, lon=o["lon"] * u.deg,
                height=o.get("height", 0.0) * u.m)
        return self._location

    def validate_steps(self):
        """Reject an impossible recipe before any frame is read."""
        bad = [s for s in self.steps if s not in VALID_STEPS]
        if bad:
            raise ValueError(
                f"{self.name}: unknown reduction step(s) {bad}. "
                f"Valid steps: {list(VALID_STEPS)}")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError(f"{self.name}: duplicate reduction steps in {self.steps}")
        if "overscan" in self.steps and self.steps[0] != "overscan":
            raise ValueError(
                f"{self.name}: 'overscan' must be the first reduction step, "
                f"got {self.steps}")
        if "overscan" in self.steps and not self.detector.overscan:
            raise ValueError(
                f"{self.name}: reduction includes 'overscan' but detector "
                f"{self.detector.name!r} defines no overscan section")
        if "dark" in self.steps and "bias" not in self.steps:
            raise ValueError(
                f"{self.name}: 'dark' requires 'bias' in the reduction steps")
        if "fringe" in self.steps and "flat" in self.steps:
            if self.steps.index("fringe") < self.steps.index("flat"):
                raise ValueError(
                    f"{self.name}: 'fringe' must follow 'flat', got {self.steps}")

    # -- discovery ---------------------------------------------------------

    def raw_dir(self, topdir, date: str, template: Optional[str] = None) -> Path:
        tmpl = template or self.raw_dir_template
        return Path(str(tmpl).format(
            topdir=str(topdir), inst=self.name, date=date,
            detector=self.detector.name)).expanduser()

    def find_raw_files(self, raw_dir) -> list:
        raw_dir = Path(raw_dir)
        found = []
        for pat in self.file_patterns:
            found.extend(glob.glob(str(raw_dir / pat)))
            if self.search_subdirectories:
                found.extend(glob.glob(str(raw_dir / "*" / pat)))
        # A '*.fits' pattern also matches '*.fits.fz' on some systems; dedupe.
        return sorted(set(found))

    # -- reading -----------------------------------------------------------

    def _detector_hdu_index(self, hdul) -> int:
        """Resolve the configured HDU to an index in this file."""
        hdu = self.detector.hdu
        if isinstance(hdu, str):
            for i, h in enumerate(hdul):
                if (h.header.get("EXTNAME", "") or "").strip() == hdu.strip():
                    return i
            raise KeyError(
                f"{self.name}: no HDU named {hdu!r} in file "
                f"(found {[h.header.get('EXTNAME') for h in hdul]})")
        idx = int(hdu)
        if idx == 0 and hdul[0].data is None and len(hdul) > 1:
            # fpacked or MEF file with an empty primary: the pixels are in the
            # first extension. This is what the legacy '.fz' special case did.
            return 1
        if idx >= len(hdul):
            raise IndexError(
                f"{self.name}: HDU {idx} requested but file has {len(hdul)}")
        return idx

    def _merged_header(self, hdul, data_index: int):
        merged = fits.Header()
        for spec in self.header_hdus:
            if spec == "data":
                idx = data_index
            elif isinstance(spec, str):
                idx = None
                for i, h in enumerate(hdul):
                    if (h.header.get("EXTNAME", "") or "").strip() == spec.strip():
                        idx = i
                        break
                if idx is None:
                    continue
            else:
                idx = int(spec)
            if idx < len(hdul):
                merged.update(hdul[idx].header)
        if data_index != 0 and 0 not in self.header_hdus and "data" not in self.header_hdus:
            # Legacy compressed-file behaviour: fall back to the extension
            # header when the primary carried nothing useful.
            merged.update(hdul[data_index].header)
        return merged

    def read_meta(self, path) -> FrameMeta:
        """Read headers only. No pixel data is decompressed."""
        path = Path(path)
        with fits.open(path, memmap=False) as hdul:
            idx = self._detector_hdu_index(hdul)
            header = self._merged_header(hdul, idx)
        return self._meta_from_header(path, header)

    def read_processed(self, path) -> FrameMeta:
        """Metadata for a frame this pipeline already wrote.

        Processed frames are always a single primary HDU carrying the merged
        header, whatever the raw file looked like, so the detector's HDU
        specification does not apply to them.
        """
        path = Path(path)
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header.copy()
        return self._meta_from_header(path, header)

    def open_frame(self, path) -> Frame:
        """Read the selected detector's pixels plus normalised metadata."""
        path = Path(path)
        with fits.open(path, memmap=False) as hdul:
            idx = self._detector_hdu_index(hdul)
            header = self._merged_header(hdul, idx)
            data = np.asarray(hdul[idx].data, dtype=np.float64)
        return Frame(meta=self._meta_from_header(path, header), data=data)

    def _meta_from_header(self, path: Path, header) -> FrameMeta:
        obj = resolve_keyword(self.keywords.get("object"), header,
                              default="", name="object")
        filt = resolve_keyword(self.keywords.get("filter"), header,
                               default="", name="filter")
        exptime = resolve_keyword(self.keywords.get("exptime"), header,
                                  default=0.0, name="exptime")
        airmass = resolve_keyword(self.keywords.get("airmass"), header,
                                  default=None, name="airmass")
        altitude = resolve_keyword(self.keywords.get("altitude"), header,
                                   default=None, name="altitude")

        obj = str(obj).strip()
        filt = str(filt).strip()
        try:
            exptime = float(exptime)
        except (TypeError, ValueError):
            exptime = 0.0

        if airmass is None and altitude is not None:
            try:
                airmass = 1.0 / math.sin(math.radians(float(altitude)))
            except (ValueError, ZeroDivisionError):
                airmass = None
        airmass = float(airmass) if airmass is not None else -1.0
        altitude = float(altitude) if altitude is not None else None

        extra = {}
        for key, spec in (self.config.get("extra_keywords", {}) or {}).items():
            extra[key] = resolve_keyword(spec, header, name=key)

        meta = FrameMeta(
            path=path, frame_type=FrameType.IGNORE, object=obj, filter=filt,
            exptime=exptime, bjd_mid=float("nan"), airmass=airmass,
            altitude=altitude, header=header, extra=extra,
        )
        meta.frame_type = self.classify(meta)
        return meta

    # -- classification ----------------------------------------------------

    @staticmethod
    def _match_rule(where: dict, header) -> bool:
        for key, want in (where or {}).items():
            if key not in header:
                return False
            have = header[key]
            have_s = str(have).strip()
            if isinstance(want, dict):
                if "regex" in want:
                    if not re.search(want["regex"], have_s):
                        return False
                elif "contains" in want:
                    if str(want["contains"]).upper() not in have_s.upper():
                        return False
                elif "equals" in want:
                    if have_s.upper() != str(want["equals"]).strip().upper():
                        return False
                else:
                    raise ValueError(f"Unrecognised match spec: {want!r}")
            else:
                if have_s.upper() != str(want).strip().upper():
                    return False
        return True

    def classify(self, meta: FrameMeta) -> FrameType:
        for rule in self.classification:
            if self._match_rule(rule.get("where", {}), meta.header):
                return FrameType(rule["type"])
        return FrameType.IGNORE

    def is_dome_flat(self, meta: FrameMeta) -> bool:
        if not self.dome_flat_rule:
            return False
        return self._match_rule(self.dome_flat_rule, meta.header)

    def flat_session(self, meta: FrameMeta) -> str:
        """Split flats into ``dome``, ``dusk``, ``dawn`` or ``unknown``.

        Where an observatory is configured the split is made about local solar
        midnight, which works at any longitude.  Without one, the historical
        fixed UT hour windows are used, which happen to suit Paranal.
        """
        if self.is_dome_flat(meta):
            return "dome"
        hour = self._ut_hour(meta)
        if hour is None:
            return "unknown"
        if self.observatory_cfg:
            # Local solar time; midnight is the dividing line.
            lon = float(self.observatory_cfg["lon"])
            local = (hour + lon / 15.0) % 24.0
            return "dusk" if local >= 12.0 else "dawn"
        if 4 <= hour <= 10:
            return "dawn"
        if 18 <= hour <= 23 or 0 <= hour <= 2:
            return "dusk"
        logger.warning("Flat taken at unusual time %d:xx in %s", hour, meta.path)
        return "unknown"

    def _ut_hour(self, meta: FrameMeta):
        header = meta.header
        for kw in ("UT-MID", "UTSTART", "UT", "DATE-OBS", "UTC", "TIME-OBS"):
            if kw not in header:
                continue
            val = header[kw]
            if not isinstance(val, str):
                continue
            s = val.strip()
            if "T" in s:
                s = s.split("T", 1)[1]
            elif " " in s:
                s = s.split(" ", 1)[1]
            if ":" not in s:
                continue
            try:
                return int(s.split(":")[0])
            except ValueError:
                continue
        return None

    # -- time --------------------------------------------------------------

    def compute_bjd(self, meta: FrameMeta, coord=None) -> float:
        """Mid-exposure time as BJD_TDB.

        When the configured keyword is already barycentric (``barycentric:
        true``, the default for SPECULOOS' BJD-OBS) the value is returned
        unchanged, so historical output is reproduced bit for bit.
        """
        from astropy.time import Time
        import astropy.units as u

        cfg = self.time_cfg
        spec = cfg.get("keyword", cfg.get("keywords"))
        if isinstance(spec, str):
            spec = {"keyword": spec}
        elif isinstance(spec, list):
            spec = {"keywords": spec}
        raw = resolve_keyword(spec, meta.header, default=None, name="time")
        if raw is None:
            raise KeyError(
                f"{self.name}: no time keyword found in {meta.path.name}. "
                f"Configured as {cfg!r}. Add a 'time:' block to the instrument "
                f"config naming a keyword this instrument actually writes.")

        fmt = cfg.get("format", "jd")
        scale = cfg.get("scale", "utc")
        reference = cfg.get("reference", "mid")

        if cfg.get("barycentric", False):
            # Already barycentric; apply only the half-exposure shift if asked.
            value = float(raw)
            if reference == "start":
                value += (meta.exptime / 2.0) / 86400.0
            return value

        t = Time(raw, format=fmt, scale=scale, location=self.location)
        if reference == "start":
            t = t + (meta.exptime / 2.0) * u.s

        if coord is None or self.location is None:
            if coord is None:
                logger.debug("No target coordinates for %s; returning JD_TDB "
                             "without barycentric correction", meta.path.name)
            return float(t.tdb.jd)
        ltt = t.light_travel_time(coord, kind="barycentric")
        return float((t.tdb + ltt).jd)

    # -- geometry ----------------------------------------------------------

    def trim(self, data: np.ndarray) -> np.ndarray:
        sl = self.detector.trim_slices
        return data if sl is None else data[sl[0], sl[1]]

    def to_trimmed_coords(self, positions):
        """Convert configured star positions into trimmed pixel coordinates.

        Positions are eyeballed on raw frames, but the pipeline works on
        trimmed images.  When the instrument declares ``coordinates: raw`` the
        trim offset is removed here, exactly once, at config load.
        """
        arr = np.asarray(positions, dtype=float)
        if self.coordinates != "raw":
            return arr
        dx, dy = self.detector.trim_offset
        if dx == 0 and dy == 0:
            return arr
        out = arr.copy()
        out[..., 0] -= dx
        out[..., 1] -= dy
        return out

    def describe_coord_conversion(self, label, positions) -> str:
        arr = np.asarray(positions, dtype=float)
        conv = self.to_trimmed_coords(arr)
        dx, dy = self.detector.trim_offset
        first_raw = np.atleast_2d(arr)[0]
        first_new = np.atleast_2d(conv)[0]
        return (f"{label} star 0: raw ({first_raw[0]:.1f}, {first_raw[1]:.1f}) "
                f"-> trimmed ({first_new[0]:.1f}, {first_new[1]:.1f}), "
                f"offset (-{dx}, -{dy})")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_instrument(name: str, config_dir, detector: Optional[str] = None,
                    raw_dir_override: Optional[str] = None) -> Instrument:
    """Build an :class:`Instrument` from ``{config_dir}/{name}.yaml``."""
    path = Path(config_dir) / f"{name}.yaml"
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Instrument config not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing instrument config {path}: {exc}") from exc

    inst = Instrument(cfg, name=cfg.get("instrument_name", name),
                      detector_name=detector)
    if raw_dir_override:
        inst.raw_dir_template = raw_dir_override
    logger.info("Instrument %s: detector=%s steps=%s coordinates=%s",
                inst.name, inst.detector.name, inst.steps, inst.coordinates)
    return inst


# --------------------------------------------------------------------------
# dated-entry lookup, shared by master flats and fringe maps
# --------------------------------------------------------------------------

def find_dated_entry(config: dict, keys, observation_date: str,
                     root: str = "filters"):
    """Pick the entry whose ``start_date`` is the latest one <= the date.

    Used for both ``master_flats.yaml`` (keys ``[filter]``) and
    ``fringe_maps.yaml`` (keys ``[filter, detector]``), so the date-range rule
    exists once rather than twice.
    """
    if not config:
        return None
    node = config.get(root)
    if node is None:
        return None
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            logger.debug("No %s entry for key %r", root, key)
            return None
        node = node[key]
    if not isinstance(node, list):
        return None

    best, best_date = None, None
    for entry in node:
        start = str(entry.get("start_date", ""))
        if start and start <= str(observation_date):
            if best_date is None or start > best_date:
                best, best_date = entry, start
    if best is not None:
        logger.info("Using %s entry from %s for %s (date %s)",
                    root, best_date, list(keys), observation_date)
    return best
