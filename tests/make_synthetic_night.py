#!/usr/bin/env python3
"""Generate a small, deterministic synthetic observing night.

Two flavours are produced:

  speculoos  single-extension frames with SPECULOOS-style headers
             (IMAGETYP / FILTER / OBJECT / BJD-OBS). Used as the regression
             fixture: the pipeline must produce identical numbers for this
             night before and after the instrument-abstraction refactor.

  wfc        multi-extension frames with INT/WFC-style headers
             (OBSTYPE / WFFBAND / MJD-OBS, four detectors, overscan + trim,
             a fringe pattern in the science and fringe frames).

Usage:
    python make_synthetic_night.py <topdir> [--flavour speculoos|wfc]
"""
import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------- parameters

RNG_SEED = 20231102

SPEC = dict(
    inst="SYNTH", date="20240115", ny=128, nx=128, bias=500.0, dark_rate=0.05,
    gain=1.6, filt="zYJ", target="SynthTarget", exptime=30.0,
    stars=[(64.0, 64.0, 40000.0), (30.0, 90.0, 22000.0), (95.0, 35.0, 15000.0)],
    fwhm=3.0, sky=120.0, n_science=8, n_bias=8, n_dark=8, n_flat_dusk=9, n_flat_dawn=9,
)

WFC = dict(
    inst="WFCSYNTH", date="20240115", ny=200, nx=160, trim_x0=10, trim_x1=140,
    overscan_x0=145, overscan_x1=160, n_det=4,
    bias=[1910.0, 1946.0, 2010.0, 1950.0], gain=[2.8, 3.0, 2.5, 2.9],
    rn=[6.4, 6.9, 5.5, 5.8], filt="z", target="SynthKELT", exptime=60.0,
    stars=[(60.0, 100.0, 30000.0), (25.0, 40.0, 18000.0), (100.0, 150.0, 12000.0)],
    fwhm=3.5, sky=210.0, n_science=6, n_bias=5, n_flat=6, n_fringe=8,
)


def gaussian_star(ny, nx, x, y, flux, fwhm):
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    return flux / (2 * np.pi * sigma ** 2) * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def fringe_pattern(ny, nx, amplitude=8.0):
    """Static, smoothly varying interference-like pattern, mean zero."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    p = (np.sin(xx / 11.0 + 0.35 * np.sin(yy / 17.0))
         + np.cos(yy / 9.0 - 0.25 * np.cos(xx / 21.0)))
    p -= p.mean()
    return amplitude * p / np.abs(p).std()


def base_header(**kw):
    h = fits.Header()
    for k, v in kw.items():
        h[k] = v
    return h


# ---------------------------------------------------------------- speculoos

def write_speculoos(topdir: Path):
    c = SPEC
    rng = np.random.default_rng(RNG_SEED)
    raw = topdir / "Observations" / c["inst"] / "images" / c["date"]
    raw.mkdir(parents=True, exist_ok=True)

    ny, nx = c["ny"], c["nx"]
    flat_true = 1.0 + 0.05 * np.sin(np.mgrid[0:ny, 0:nx][1] / 20.0) \
                    + 0.03 * np.cos(np.mgrid[0:ny, 0:nx][0] / 14.0)
    dark_true = c["dark_rate"] * (1.0 + 0.2 * rng.random((ny, nx)))

    def common(imagetyp, exptime, hours, i, obj):
        h = base_header(
            IMAGETYP=imagetyp, OBJECT=obj, FILTER=c["filt"], EXPTIME=exptime,
            GAIN=c["gain"], INSTRUME=c["inst"],
        )
        h["DATE-OBS"] = f"2024-01-15T{hours:02d}:{(i * 3) % 60:02d}:00.000"
        h["BJD-OBS"] = 2460324.5 + (hours + i * 0.01) / 24.0
        h["MJD-OBS"] = 60324.0 + (hours + i * 0.01) / 24.0
        h["AIRMASS"] = 1.05 + 0.004 * i
        h["ALTITUDE"] = 72.0 - 0.25 * i
        return h

    n = 0
    for i in range(c["n_bias"]):
        d = c["bias"] + rng.normal(0, 3.0, (ny, nx))
        fits.PrimaryHDU(d.astype(np.float32),
                        common("BIAS", 0.0, 19, i, "Bias")).writeto(
            raw / f"bias_{i:03d}.fits", overwrite=True)
        n += 1
    for i in range(c["n_dark"]):
        d = c["bias"] + dark_true * c["exptime"] + rng.normal(0, 3.0, (ny, nx))
        fits.PrimaryHDU(d.astype(np.float32),
                        common("DARK", c["exptime"], 19, i, "Dark")).writeto(
            raw / f"dark_{i:03d}.fits", overwrite=True)
        n += 1
    for label, hours, count in (("dusk", 19, c["n_flat_dusk"]),
                                ("dawn", 6, c["n_flat_dawn"])):
        for i in range(count):
            level = 22000.0 + 500.0 * i
            d = c["bias"] + flat_true * level + rng.normal(0, 30.0, (ny, nx))
            fits.PrimaryHDU(d.astype(np.float32),
                            common("FLAT", 4.0, hours, i, "Flat")).writeto(
                raw / f"flat_{label}_{i:03d}.fits", overwrite=True)
            n += 1
    for i in range(c["n_science"]):
        img = np.full((ny, nx), c["sky"], float)
        for (x, y, f) in c["stars"]:
            img += gaussian_star(ny, nx, x + 0.03 * i, y - 0.02 * i, f, c["fwhm"])
        d = c["bias"] + dark_true * c["exptime"] + flat_true * img
        d += rng.normal(0, 6.0, (ny, nx))
        fits.PrimaryHDU(d.astype(np.float32),
                        common("LIGHT", c["exptime"], 22, i, c["target"])).writeto(
            raw / f"science_{i:03d}.fits", overwrite=True)
        n += 1
    return raw, n


# ---------------------------------------------------------------------- wfc

def write_wfc(topdir: Path):
    c = WFC
    rng = np.random.default_rng(RNG_SEED + 1)
    raw = topdir / "WFCDATA" / c["date"] / "download_dir"
    raw.mkdir(parents=True, exist_ok=True)

    ny, nx = c["ny"], c["nx"]
    tx0, tx1 = c["trim_x0"], c["trim_x1"]
    tw = tx1 - tx0
    flat_true = [1.0 + 0.04 * np.sin(np.mgrid[0:ny, 0:tw][1] / 15.0 + d)
                 + 0.02 * np.cos(np.mgrid[0:ny, 0:tw][0] / 11.0)
                 for d in range(c["n_det"])]
    fringe = [fringe_pattern(ny, tw, amplitude=9.0 - d) for d in range(c["n_det"])]

    def primary(obstype, imagetyp, obj, exptime, hours, i):
        h = base_header(
            OBSTYPE=obstype, IMAGETYP=imagetyp, OBJECT=obj, WFFBAND=c["filt"],
            EXPTIME=exptime, INSTRUME="WFC", DETECTOR="WFC", TELESCOP="INT",
            OBSERVAT="LAPALMA", CCDSPEED="FAST", CCDXBIN=1, CCDYBIN=1,
        )
        h["DATE-OBS"] = "2024-01-15"
        h["MJD-OBS"] = 60324.0 + (hours + i * 0.02) / 24.0
        h["UTSTART"] = f"{hours:02d}:{(i * 5) % 60:02d}:11.4"
        h["ZDSTART"] = 20.0 + 0.5 * i
        h["ZDEND"] = 20.1 + 0.5 * i
        h["AIRMASS"] = 1.06 + 0.01 * i
        h["RA"] = "20:57:02.0"
        h["DEC"] = "+31:36:10.9"
        h["RUN"] = 1400000 + i
        h["RUNSET"] = "1:21:1400000"
        return h

    def write(fname, prim, planes):
        hdul = fits.HDUList([fits.PrimaryHDU(header=prim)])
        for d, plane in enumerate(planes):
            hh = base_header(EXTNAME=f"extension{d+1}", IMAGEID=d + 1,
                             CCDNAME=f"SYNTH-{d+1}", GAIN=c["gain"][d],
                             READNOIS=c["rn"][d], SATURATE=65535.0)
            hh["BIASSEC"] = f"[{c['overscan_x0']+1}:{c['overscan_x1']},1:{ny}]"
            hh["TRIMSEC"] = f"[{tx0+1}:{tx1},1:{ny}]"
            hh["INHERIT"] = True
            hdul.append(fits.ImageHDU(plane.astype(np.float32), hh))
        hdul.writeto(raw / fname, overwrite=True)

    def frame(det, content_trim, bias_level):
        """Assemble a full untrimmed detector plane from trimmed content."""
        p = np.full((ny, nx), bias_level, float)
        p += rng.normal(0, c["rn"][det] / c["gain"][det], (ny, nx))
        p[:, tx0:tx1] += content_trim
        return p

    n = 0
    for i in range(c["n_bias"]):
        planes = [frame(d, np.zeros((ny, tw)), c["bias"][d] + 0.4 * i)
                  for d in range(c["n_det"])]
        write(f"int20240115_{1400000+n:08d}.fits",
              primary("BIAS", "zero", "Bias", 0.0, 16, i), planes)
        n += 1
    for i in range(c["n_flat"]):
        level = 25000.0 + 400.0 * i
        planes = [frame(d, flat_true[d] * level + rng.normal(0, 40.0, (ny, tw)),
                        c["bias"][d] + 0.4 * i) for d in range(c["n_det"])]
        write(f"int20240115_{1400000+n:08d}.fits",
              primary("TARGET", "object", "Sky flat Z", 4.0, 18, i), planes)
        n += 1
    # dithered blank-sky frames carrying the fringe pattern
    for i in range(c["n_fringe"]):
        amp = 1.0 + 0.05 * i
        planes = []
        for d in range(c["n_det"]):
            content = c["sky"] + amp * fringe[d] + rng.normal(0, 4.0, (ny, tw))
            # a couple of stars, dithered so a median rejects them
            content += gaussian_star(ny, tw, 20 + 7 * i, 30 + 5 * i, 9000.0, c["fwhm"])
            planes.append(frame(d, content * flat_true[d], c["bias"][d] + 0.4 * i))
        write(f"int20240115_{1400000+n:08d}.fits",
              primary("TARGET", "object", "Fringe Map", 60.0, 21, i), planes)
        n += 1
    # a non-linearity test block that must be classified as 'ignore'
    for i in range(2):
        planes = [frame(d, np.full((ny, tw), 30000.0), c["bias"][d])
                  for d in range(c["n_det"])]
        write(f"int20240115_{1400000+n:08d}.fits",
              primary("TARGET", "object", "NLtest", 16.0, 19, i), planes)
        n += 1
    for i in range(c["n_science"]):
        amp = 1.4 + 0.25 * i          # fringe amplitude rises through the night
        planes = []
        for d in range(c["n_det"]):
            content = np.full((ny, tw), c["sky"], float) + amp * fringe[d]
            if d == 3:                # target chip
                for (x, y, f) in c["stars"]:
                    content += gaussian_star(ny, tw, x + 0.02 * i, y - 0.01 * i,
                                             f, c["fwhm"])
            content += rng.normal(0, 5.0, (ny, tw))
            planes.append(frame(d, content * flat_true[d], c["bias"][d] + 0.4 * i))
        write(f"int20240115_{1400000+n:08d}.fits",
              primary("TARGET", "object", c["target"], 60.0, 23, i), planes)
        n += 1
    return raw, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topdir")
    ap.add_argument("--flavour", choices=["speculoos", "wfc", "both"],
                    default="both")
    a = ap.parse_args()
    top = Path(a.topdir).expanduser()
    if a.flavour in ("speculoos", "both"):
        raw, n = write_speculoos(top)
        print(f"speculoos: {n} frames in {raw}")
    if a.flavour in ("wfc", "both"):
        raw, n = write_wfc(top)
        print(f"wfc:       {n} frames in {raw}")


if __name__ == "__main__":
    main()
