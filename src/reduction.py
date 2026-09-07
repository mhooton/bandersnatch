"""Per-frame reduction: an ordered, configurable chain of steps.

The recipe is ``instrument.steps``, for example ``[bias, dark, flat]`` for the
SPECULOOS cameras or ``[overscan, bias, flat, fringe]`` for INT/WFC.  A
:class:`Reducer` is built once per target from the instrument and the loaded
master frames, then applied to each frame in turn.

Each step may contribute per-frame scalars (overscan level, fringe scale, sky
level).  These are returned alongside the reduced image and written to the
per-target ``frames.fits`` table, where they are available later as
decorrelation vectors without re-opening any raw file.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from utils import clean_bad_pixels

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# overscan
# --------------------------------------------------------------------------

def overscan_level(data: np.ndarray, section, method: str = "median",
                   order: int = 1):
    """Measure the bias level from the overscan strip.

    ``median`` returns a scalar, which is what INT/WFC wants: the row-to-row
    scatter of a 53-pixel median is 0.4 ADU, i.e. pure noise, so a per-row
    correction would add noise for no gain.  ``row_median`` reproduces the
    per-row form used by the IDL pipeline; ``polynomial`` fits the row medians.
    """
    rows, cols = section
    strip = np.asarray(data[rows, cols], dtype=np.float64)
    if strip.size == 0:
        raise ValueError("Overscan section selects no pixels")

    if method == "median":
        return float(np.median(strip))
    if method == "row_median":
        return np.median(strip, axis=1)
    if method == "polynomial":
        per_row = np.median(strip, axis=1)
        y = np.arange(per_row.size, dtype=float)
        good = np.isfinite(per_row)
        coeffs = np.polyfit(y[good], per_row[good], order)
        return np.polyval(coeffs, y)
    raise ValueError(f"Unknown overscan method {method!r}")


def apply_overscan(data: np.ndarray, level):
    if np.isscalar(level):
        return data - float(level)
    level = np.asarray(level, dtype=np.float64)
    if level.ndim == 1 and level.size == data.shape[0]:
        return data - level[:, None]
    raise ValueError("Overscan level has the wrong shape for this image")


# --------------------------------------------------------------------------
# fringe correction
# --------------------------------------------------------------------------

def build_star_mask(image: np.ndarray, smooth=3, threshold_mad=4.0, dilate=12,
                    border=40, bad_columns=(), column_pad=6):
    """Boolean mask, True where a pixel is usable sky.

    Deliberately crude and generous: it does not need to find stars, only to
    exclude them.  The whole-image fringe fit is insensitive to the settings
    (varying the threshold from 3 to 10 MAD and the dilation from 6 to 25 px
    moved the fitted scale by under 3 per cent on INT/WFC data), so a wide
    margin costs nothing.
    """
    s = gaussian_filter(np.nan_to_num(image, nan=0.0), smooth)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    if mad <= 0:
        mad = np.std(s) or 1.0
    flagged = s > med + threshold_mad * mad
    if dilate > 0:
        flagged = binary_dilation(flagged, iterations=int(dilate))
    flagged |= ~np.isfinite(image)
    for col in bad_columns or ():
        lo = max(0, int(col) - column_pad)
        hi = min(image.shape[1], int(col) + column_pad + 1)
        flagged[:, lo:hi] = True
    if border > 0:
        b = int(border)
        flagged[:b, :] = True
        flagged[-b:, :] = True
        flagged[:, :b] = True
        flagged[:, -b:] = True
    return ~flagged


def fit_fringe_scale(image: np.ndarray, template: np.ndarray, mask=None,
                     plane=True, clip_sigma=3.0, iterations=3, subsample=7):
    """Fit ``image = scale * template + offset [+ plane]`` over sky pixels.

    Returns ``(scale, offset, n_pixels)``.

    This is the whole-image fit rather than a hand-placed pair of boxes.  On
    INT/WFC data it repeats to 1 per cent between even and odd rows, where a
    single box pair scatters by about 20 per cent and a pair placed on the
    template extremes read half the true scale.

    The template must already be smoothed (see :func:`prepare_fringe_template`).
    Fitting against a noisy template attenuates the recovered slope by the
    ratio of true to measured template variance, which was 15 to 25 per cent on
    real data: a large, purely instrumental error.
    """
    if image.shape != template.shape:
        raise ValueError(
            f"Fringe template shape {template.shape} does not match image "
            f"{image.shape}")
    if mask is None:
        mask = np.isfinite(image)
    sel = mask & np.isfinite(image) & np.isfinite(template)
    idx = np.nonzero(sel.ravel())[0][::max(1, int(subsample))]
    if idx.size < 100:
        raise ValueError(
            f"Only {idx.size} usable sky pixels for the fringe fit; "
            "check the star mask and border settings")

    y = image.ravel()[idx]
    cols = [template.ravel()[idx], np.ones(idx.size)]
    if plane:
        ny, nx = image.shape
        yy, xx = np.unravel_index(idx, image.shape)
        cols.append(xx / nx - 0.5)
        cols.append(yy / ny - 0.5)
    A = np.column_stack(cols)

    good = np.ones(idx.size, dtype=bool)
    coeffs = None
    for _ in range(max(1, int(iterations))):
        coeffs, *_ = np.linalg.lstsq(A[good], y[good], rcond=None)
        resid = y - A @ coeffs
        sigma = 1.4826 * np.median(np.abs(resid[good] - np.median(resid[good])))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        good = np.abs(resid - np.median(resid[good])) < clip_sigma * sigma
        if good.sum() < 100:
            break
    return float(coeffs[0]), float(coeffs[1]), int(good.sum())


def prepare_fringe_template(template: np.ndarray, highpass_sigma=120.0,
                            smooth_sigma=2.0):
    """Separate the fringe pattern and suppress template pixel noise.

    The high pass removes the large-scale sky and illumination component, which
    would otherwise compete with the real sky gradient of a science frame.  The
    smoothing removes the template's own pixel noise, which attenuates the
    fitted scale.  Both matter: fitting the unseparated, unsmoothed map was 30
    per cent wrong in twilight and 15 to 25 per cent low elsewhere.
    """
    t = np.asarray(template, dtype=np.float64)
    if highpass_sigma and highpass_sigma > 0:
        t = t - gaussian_filter(t, highpass_sigma)
    if smooth_sigma and smooth_sigma > 0:
        t = gaussian_filter(t, smooth_sigma)
    return t


# --------------------------------------------------------------------------
# the reducer
# --------------------------------------------------------------------------

class Reducer:
    """Applies an instrument's ordered reduction recipe to single frames."""

    def __init__(self, instrument, calib_frames=None, bad_pixel_map=None,
                 fringe_template=None, steps=None):
        self.instrument = instrument
        self.calib = calib_frames or {}
        self.bad_pixel_map = bad_pixel_map
        # `steps` lets a calibration builder run a truncated chain: a master
        # flat is built from frames taken through everything up to, but not
        # including, the flat step itself.
        self.steps = list(instrument.steps if steps is None else steps)

        self._raw_fringe = fringe_template
        self.fringe_template = None
        if fringe_template is not None:
            p = instrument.fringe
            self.fringe_template = prepare_fringe_template(
                fringe_template, p.highpass_sigma, p.template_smooth_sigma)

        self.validate_products()

    # -- validation --------------------------------------------------------

    def validate_products(self):
        """Fail before processing rather than part-way through a night."""
        missing = []
        for step in self.steps:
            if step == "bias" and self.calib.get("bias") is None:
                missing.append("master bias")
            if step == "dark" and self.calib.get("dark") is None:
                missing.append("master dark")
            if step == "flat" and self.calib.get("flat") is None:
                missing.append("master flat")
            if step == "fringe" and self.fringe_template is None:
                missing.append("fringe map")
        if missing:
            raise ValueError(
                f"{self.instrument.name}: reduction steps {self.steps} require "
                f"products that are not loaded: {', '.join(missing)}")

    # -- the chain ---------------------------------------------------------

    def reduce(self, frame):
        """Return ``(image, record)`` for one :class:`~instrument.Frame`."""
        inst = self.instrument
        data = np.array(frame.data, dtype=np.float64, copy=True)
        record = {}

        if "overscan" in self.steps:
            level = overscan_level(
                data, inst.detector.overscan_slices,
                method=self.instrument.overscan_cfg.get("method", "median"),
                order=int(self.instrument.overscan_cfg.get("order", 1)))
            data = apply_overscan(data, level)
            record["overscan_level"] = (
                float(level) if np.isscalar(level) else float(np.median(level)))

        # Trim is implicit: everything downstream is in trimmed coordinates.
        data = inst.trim(data)

        for step in self.steps:
            if step == "overscan":
                continue
            if step == "bias":
                data = data - self.calib["bias"]
            elif step == "dark":
                data = data - self.calib["dark"] * frame.meta.exptime
            elif step == "flat":
                data = data / self.calib["flat"]
            elif step == "bpm":
                if self.bad_pixel_map is not None:
                    data = clean_bad_pixels(data, self.bad_pixel_map)
            elif step == "fringe":
                data, fr = self.apply_fringe(data)
                record.update(fr)

        if self.bad_pixel_map is not None and "bpm" not in self.steps:
            # Historical behaviour: a bad pixel map, when present, is applied
            # after the calibration chain even though it is not a named step.
            data = clean_bad_pixels(data, self.bad_pixel_map)

        return data, record

    def apply_fringe(self, image):
        """Scale and subtract the fringe template; record the scale."""
        inst = self.instrument
        p = inst.fringe
        mask = build_star_mask(
            image, bad_columns=inst.detector.bad_columns, **p.star_mask)
        sky = float(np.median(image[mask])) if mask.any() else float(np.median(image))
        scale, offset, npix = fit_fringe_scale(
            image - sky, self.fringe_template, mask=mask, **p.fit)
        corrected = image - scale * self.fringe_template
        return corrected, {
            "fringe_scale": scale,
            "fringe_intercept": offset,
            "sky_level": sky,
            "fringe_fit_pixels": npix,
        }


# --------------------------------------------------------------------------
# fringe map construction
# --------------------------------------------------------------------------

def steps_before(instrument, step):
    """The reduction steps that precede ``step``, for calibration building.

    A master flat must be built from frames reduced exactly as far as the flat
    step, and a fringe map from frames reduced exactly as far as the fringe
    step, or the product will not match the frames it is applied to.
    """
    steps = list(instrument.steps)
    return steps[:steps.index(step)] if step in steps else steps


def combine_fringe_frames(images, dtype=np.float32):
    """Median-combine reduced, dithered blank-sky frames, sky removed.

    Each frame has its own median subtracted, then the stack is
    median-combined.  Dithering is what removes the stars: at any pixel a star
    is present in at most one or two frames, so the median rejects it without
    any star finding.

    The stack is assembled in place at ``dtype``, rather than by building a
    list and calling np.stack, because 55 INT/WFC frames are 3.7 GB in float64
    and the list plus the stack would hold two copies of that.
    """
    images = list(images)
    if not images:
        raise ValueError("No frames to combine into a fringe map")
    first = np.asarray(images[0])
    stack = np.empty((len(images),) + first.shape, dtype=dtype)
    for i, im in enumerate(images):
        arr = np.asarray(im, dtype=np.float64)
        stack[i] = arr - np.nanmedian(arr)
    return np.nanmedian(stack, axis=0).astype(np.float64)


def highpass(image, sigma):
    """Remove the large-scale component, leaving the fringe pattern."""
    if not sigma or sigma <= 0:
        return image
    return image - gaussian_filter(image, sigma)


def build_fringe_map(images, highpass_sigma=120.0):
    """Combine dithered blank-sky frames into a fringe template.

    The large-scale component is removed after combining, so what is returned
    is the fringe pattern alone.
    """
    return highpass(combine_fringe_frames(images), highpass_sigma)
