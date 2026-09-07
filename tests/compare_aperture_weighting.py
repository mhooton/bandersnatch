#!/usr/bin/env python3
"""Exact versus approximate aperture weighting in aper().

Builds a long synthetic sequence with the star field, PSF, sky level, gain
and read noise of the SPECULOOS flavour of make_synthetic_night.py, moves the
stars by the same slow drift plus a per-frame guiding jitter so that many
sub-pixel phases are sampled, measures every frame with both weightings, and
reports the scatter of the differential light curve of the brightest star
against the sum of the other two.

Three scenarios:
  A  noise-free frames, true positions   -> pixel-phase systematics only
  B  Poisson + read noise, true positions -> realistic
  C  as B, with 0.05 px errors on the positions handed to aper()

Usage:
    python compare_aperture_weighting.py [--n-frames 400] [--seed 1]
                                         [--aper-min 2 --aper-max 12]
                                         [--skyrad 15 25]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from aper import aper                                  # noqa: E402
from make_synthetic_night import SPEC, gaussian_star  # noqa: E402

READ_NOISE_ADU = 6.0        # as in make_synthetic_night.write_speculoos
JITTER_PX = 0.25            # per-frame guiding jitter, common to all stars
DRIFT_PX = (0.03, -0.02)    # per-frame drift, as in the synthetic night


def build_sequence(n_frames, rng, noise, fwhm):
    c = SPEC
    ny, nx = c["ny"], c["nx"]
    stars = np.array(c["stars"], dtype=float)
    # the synthetic night puts every star on a pixel centre; give each its own
    # fixed sub-pixel phase, as real fields have, so pixel-phase errors do not
    # cancel identically in the ratio
    stars[:, :2] += rng.uniform(0.0, 1.0, (len(stars), 2))
    frames = np.empty((n_frames, ny, nx))
    pos = np.empty((n_frames, len(stars), 2))
    for i in range(n_frames):
        ddx = DRIFT_PX[0] * i + rng.normal(0, JITTER_PX)
        ddy = DRIFT_PX[1] * i + rng.normal(0, JITTER_PX)
        img = np.full((ny, nx), c["sky"], float)
        for j, (x, y, f) in enumerate(stars):
            pos[i, j] = (x + ddx, y + ddy)
            img += gaussian_star(ny, nx, x + ddx, y + ddy, f, fwhm)
        if noise:
            img = rng.poisson(img * c["gain"]) / c["gain"]
            img += rng.normal(0, READ_NOISE_ADU, (ny, nx))
        frames[i] = img
    return frames, pos


def measure(frames, pos, apr, skyrad, exact):
    flux = np.empty((len(frames),) + (len(apr), pos.shape[1]))
    for i, img in enumerate(frames):
        flux[i] = aper(img, pos[i, :, 0], pos[i, :, 1], apr=apr, skyrad=skyrad,
                       flux=True, silent=True, exact=exact)[0]
    return flux                                  # (frame, aperture, star)


def differential(flux, target=0):
    comps = np.delete(flux, target, axis=2).sum(axis=2)
    d = flux[:, :, target] / comps
    return d / np.median(d, axis=0)


def raw_scatter(flux, star=0):
    f = flux[:, :, star]
    return np.std(f / np.median(f, axis=0), axis=0, ddof=1)


def report(label, apr, f_exact, f_approx, unit="ppt"):
    scale = {"ppt": 1e3, "ppm": 1e6}[unit]
    d_exact, d_approx = differential(f_exact), differential(f_approx)
    s_e = np.std(d_exact, axis=0, ddof=1) * scale
    s_a = np.std(d_approx, axis=0, ddof=1) * scale
    delta = np.std(d_exact - d_approx, axis=0, ddof=1) * scale
    raw_e = raw_scatter(f_exact) * scale
    raw_a = raw_scatter(f_approx) * scale
    print(f"\n{label}  [{unit}]")
    print(f"{'r':>4} | {'differential':^33} | {'target raw flux':^19}")
    print(f"{'px':>4} | {'exact':>9} {'approx':>9} {'a/e':>6} {'rms(a-e)':>9} "
          f"| {'exact':>9} {'approx':>9}")
    for r, e, a, dd, re_, ra in zip(apr, s_e, s_a, delta, raw_e, raw_a):
        print(f"{r:4.0f} | {e:9.3f} {a:9.3f} {a / e:6.3f} {dd:9.3f} "
              f"| {re_:9.3f} {ra:9.3f}")
    k = np.argmin(s_e)
    print(f"best exact aperture r={apr[k]:.0f}: {s_e[k]:.3f} {unit}; "
          f"approx at the same r: {s_a[k]:.3f} {unit}; "
          f"best approx r={apr[np.argmin(s_a)]:.0f}: {s_a.min():.3f} {unit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aper-min", type=int, default=2)
    ap.add_argument("--aper-max", type=int, default=12)
    ap.add_argument("--skyrad", type=float, nargs=2, default=(15.0, 25.0))
    ap.add_argument("--fwhm", type=float, default=SPEC["fwhm"],
                    help="PSF FWHM in pixels (default: the synthetic night's)")
    a = ap.parse_args()

    apr = np.arange(a.aper_min, a.aper_max + 1, dtype=float)
    skyrad = np.array(a.skyrad)
    print(f"{a.n_frames} frames, apertures {a.aper_min}-{a.aper_max} px, "
          f"sky annulus {skyrad[0]:.0f}-{skyrad[1]:.0f} px, jitter {JITTER_PX} px, "
          f"FWHM {a.fwhm} px")

    rng = np.random.default_rng(a.seed)
    frames, pos = build_sequence(a.n_frames, rng, noise=False, fwhm=a.fwhm)
    report("A: noise-free, true positions", apr,
           measure(frames, pos, apr, skyrad, True),
           measure(frames, pos, apr, skyrad, False), unit="ppm")

    rng = np.random.default_rng(a.seed)
    frames, pos = build_sequence(a.n_frames, rng, noise=True, fwhm=a.fwhm)
    fe = measure(frames, pos, apr, skyrad, True)
    fa = measure(frames, pos, apr, skyrad, False)
    report("B: Poisson + read noise, true positions", apr, fe, fa)

    noisy_pos = pos + rng.normal(0, 0.05, pos.shape)
    report("C: as B, with 0.05 px centroid errors", apr,
           measure(frames, noisy_pos, apr, skyrad, True),
           measure(frames, noisy_pos, apr, skyrad, False))


if __name__ == "__main__":
    main()
