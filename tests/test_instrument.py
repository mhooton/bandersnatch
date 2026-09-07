"""Tests for the instrument abstraction and the reduction chain."""

import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.io import fits

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instrument import (  # noqa: E402
    Instrument, FrameType, parse_fits_section, resolve_keyword,
    find_dated_entry, load_instrument,
)
import reduction  # noqa: E402


# --------------------------------------------------------------------- fixtures

SPEC_CFG = {
    "instrument_name": "SYNTH",
    "gain": 1.6,
    "phpadu": 1.0,
}

WFC_CFG = {
    "instrument_name": "WFCTEST",
    "raw_dir": "{topdir}/WFCDATA/{date}/download_dir",
    "file_patterns": ["*.fits"],
    "header_hdus": [0, "data"],
    "detectors": {
        "CCD1": {"hdu": "extension1", "trim": "[11:140,1:200]",
                 "overscan": "[146:160,1:200]", "gain": 2.8, "read_noise": 6.4},
        "CCD4": {"hdu": "extension4", "trim": "[11:140,1:200]",
                 "overscan": "[146:160,1:200]", "gain": 2.9, "read_noise": 5.8,
                 "bad_columns": [77]},
    },
    "default_detector": "CCD4",
    "coordinates": "raw",
    "keywords": {
        "exptime": "EXPTIME",
        "filter": {"keyword": "WFFBAND", "strip": True},
        "object": "OBJECT",
        "airmass": {"keywords": ["AIRMASS", "AMSTART"]},
        "altitude": {"expr": "90 - 0.5*(ZDSTART + ZDEND)"},
    },
    "time": {"keyword": "MJD-OBS", "format": "mjd", "scale": "utc",
             "reference": "start"},
    "observatory": {"lat": 28.7619, "lon": -17.8776, "height": 2348},
    "classification": [
        {"type": "bias", "where": {"OBSTYPE": "BIAS"}},
        {"type": "ignore", "where": {"OBJECT": {"regex": "(?i)^NLtest"}}},
        {"type": "flat", "where": {"OBJECT": {"regex": "(?i)flat"}}},
        {"type": "fringe", "where": {"OBJECT": {"regex": "(?i)fringe"}}},
        {"type": "science", "where": {"OBSTYPE": "TARGET"}},
    ],
    "dome_flat": {"OBJECT": {"regex": "(?i)domeflat"}},
    "reduction": {"steps": ["overscan", "bias", "flat", "fringe"]},
}


def make_speculoos_file(path, imagetyp="LIGHT", obj="Targ", filt="zYJ"):
    h = fits.Header()
    h["IMAGETYP"] = imagetyp
    h["OBJECT"] = obj
    h["FILTER"] = filt
    h["EXPTIME"] = 30.0
    h["BJD-OBS"] = 2460324.5
    h["AIRMASS"] = 1.2
    h["ALTITUDE"] = 56.4
    h["DATE-OBS"] = "2024-01-15T22:10:00.000"
    fits.PrimaryHDU(np.full((20, 30), 7.0, np.float32), h).writeto(
        path, overwrite=True)


def make_wfc_file(path, obstype="TARGET", obj="KELT16", ny=200, nx=160):
    p = fits.Header()
    p["OBSTYPE"] = obstype
    p["IMAGETYP"] = "object"
    p["OBJECT"] = obj
    p["WFFBAND"] = "z       "
    p["EXPTIME"] = 60.0
    p["MJD-OBS"] = 60324.9
    p["ZDSTART"] = 20.0
    p["ZDEND"] = 21.0
    p["AIRMASS"] = 1.06
    p["UTSTART"] = "23:15:00.0"
    hdul = fits.HDUList([fits.PrimaryHDU(header=p)])
    for d in range(4):
        hh = fits.Header()
        hh["EXTNAME"] = f"extension{d+1}"
        hh["GAIN"] = [2.8, 3.0, 2.5, 2.9][d]
        hh["INHERIT"] = True
        arr = np.zeros((ny, nx), np.float32)
        arr[:, :] = 100.0 * (d + 1)          # detector-specific level
        arr[:, 145:160] = 50.0 * (d + 1)     # overscan strip
        hdul.append(fits.ImageHDU(arr, hh))
    hdul.writeto(path, overwrite=True)


# ------------------------------------------------------------------ sections

def test_parse_fits_section_is_one_based_inclusive():
    rows, cols = parse_fits_section("[54:2101,1:4096]")
    assert (cols.start, cols.stop) == (53, 2101)
    assert (rows.start, rows.stop) == (0, 4096)
    a = np.arange(4200 * 2154).reshape(4200, 2154)
    assert a[rows, cols].shape == (4096, 2048)


def test_parse_fits_section_ignores_trailing_text():
    # INT/WFC writes '[0:0,0:0], disabled'
    rows, cols = parse_fits_section("[800:1200,1800:2200], disabled")
    assert (cols.start, cols.stop) == (799, 1200)


def test_parse_fits_section_rejects_rubbish():
    with pytest.raises(ValueError):
        parse_fits_section("not a section")
    with pytest.raises(ValueError):
        parse_fits_section("[100:10,1:5]")


# ------------------------------------------------------------------ keywords

def test_resolve_keyword_forms():
    h = fits.Header()
    h["EXPTIME"] = 60.0
    h["WFFBAND"] = "z       "
    h["ZDSTART"] = 20.0
    h["ZDEND"] = 22.0
    h["AMSTART"] = 1.5
    assert resolve_keyword("EXPTIME", h) == 60.0
    assert resolve_keyword({"keyword": "WFFBAND", "strip": True}, h) == "z"
    # first present wins, and a missing first entry falls through
    assert resolve_keyword({"keywords": ["AIRMASS", "AMSTART"]}, h) == 1.5
    assert resolve_keyword({"expr": "90 - 0.5*(ZDSTART + ZDEND)"}, h) == 69.0
    assert resolve_keyword("NOPE", h, default=-1) == -1
    with pytest.raises(KeyError):
        resolve_keyword("NOPE", h, required=True, name="thing")


def test_expression_handles_hyphenated_keywords():
    h = fits.Header()
    h["MJD-OBS"] = 60324.0
    assert resolve_keyword({"expr": "MJD-OBS + 1"}, h) == 60325.0


def test_expression_cannot_reach_builtins():
    h = fits.Header()
    h["X"] = 1
    assert resolve_keyword({"expr": "__import__('os').getcwd()"}, h) is None


# ------------------------------------------------------------ classification

def test_legacy_classification_matches_old_behaviour(tmp_path):
    inst = Instrument(SPEC_CFG)
    for imagetyp, expected in (("LIGHT", FrameType.SCIENCE),
                               ("DARK", FrameType.DARK),
                               ("BIAS", FrameType.BIAS),
                               ("FLAT", FrameType.FLAT),
                               ("dark frame", FrameType.DARK),
                               ("Light Image", FrameType.SCIENCE),
                               ("something else", FrameType.IGNORE)):
        p = tmp_path / "f.fits"
        make_speculoos_file(p, imagetyp=imagetyp)
        assert inst.read_meta(p).frame_type is expected, imagetyp


def test_wfc_classification_rules(tmp_path):
    inst = Instrument(WFC_CFG)
    cases = [("BIAS", "Bias", FrameType.BIAS),
             ("TARGET", "NLtest", FrameType.IGNORE),
             ("TARGET", "Sky flat Z", FrameType.FLAT),
             ("TARGET", "Domeflat Z", FrameType.FLAT),
             ("TARGET", "Fringe Map", FrameType.FRINGE),
             ("TARGET", "KELT16", FrameType.SCIENCE)]
    for obstype, obj, expected in cases:
        p = tmp_path / "f.fits"
        make_wfc_file(p, obstype=obstype, obj=obj)
        assert inst.read_meta(p).frame_type is expected, (obstype, obj)


def test_rule_order_matters_nltest_before_flat(tmp_path):
    """NLtest frames are r-band lamp exposures, not flats: ignore wins."""
    inst = Instrument(WFC_CFG)
    p = tmp_path / "f.fits"
    make_wfc_file(p, obstype="TARGET", obj="NLtest")
    assert inst.read_meta(p).frame_type is FrameType.IGNORE


def test_dome_flats_separated_from_sky_flats(tmp_path):
    inst = Instrument(WFC_CFG)
    p = tmp_path / "f.fits"
    make_wfc_file(p, obj="Domeflat Z")
    assert inst.flat_session(inst.read_meta(p)) == "dome"
    make_wfc_file(p, obj="Sky flat Z")
    assert inst.flat_session(inst.read_meta(p)) in ("dusk", "dawn")


# ----------------------------------------------------------------- detectors

def test_detector_selection_and_header_merge(tmp_path):
    p = tmp_path / "f.fits"
    make_wfc_file(p)
    inst = Instrument(WFC_CFG)                       # default CCD4
    fr = inst.open_frame(p)
    assert fr.data.shape == (200, 160)
    assert np.isclose(np.median(fr.data), 400.0)     # extension4 level
    assert inst.gain() == 2.9
    # primary keywords are visible alongside the extension's own
    assert fr.meta.header["OBSTYPE"].strip() == "TARGET"
    assert fr.meta.header["GAIN"] == 2.9

    inst1 = Instrument(WFC_CFG, detector_name="CCD1")
    assert np.isclose(np.median(inst1.open_frame(p).data), 100.0)
    assert inst1.gain() == 2.8


def test_unknown_detector_is_rejected():
    with pytest.raises(KeyError):
        Instrument(WFC_CFG, detector_name="CCD9")


def test_legacy_single_extension_still_works(tmp_path):
    p = tmp_path / "f.fits"
    make_speculoos_file(p)
    inst = Instrument(SPEC_CFG)
    fr = inst.open_frame(p)
    assert fr.data.shape == (20, 30)
    assert inst.detector.trim is None
    assert inst.trim(fr.data).shape == (20, 30)      # no trim configured
    assert inst.gain() == 1.6


def test_metadata_normalisation(tmp_path):
    p = tmp_path / "f.fits"
    make_wfc_file(p)
    meta = Instrument(WFC_CFG).read_meta(p)
    assert meta.filter == "z"                        # stripped
    assert meta.exptime == 60.0
    assert meta.altitude == pytest.approx(69.5)      # from ZDSTART/ZDEND
    assert meta.airmass == pytest.approx(1.06)


def test_airmass_derived_from_altitude_when_absent(tmp_path):
    cfg = dict(SPEC_CFG)
    cfg["keywords"] = {"airmass": {"keywords": ["NOSUCH"]},
                       "altitude": "ALTITUDE"}
    p = tmp_path / "f.fits"
    make_speculoos_file(p)
    meta = Instrument(cfg).read_meta(p)
    assert meta.airmass == pytest.approx(1.0 / np.sin(np.radians(56.4)))


# ---------------------------------------------------------------------- time

def test_speculoos_time_is_passthrough(tmp_path):
    """The legacy default must reproduce BJD-OBS bit for bit."""
    p = tmp_path / "f.fits"
    make_speculoos_file(p)
    inst = Instrument(SPEC_CFG)
    meta = inst.read_meta(p)
    assert inst.compute_bjd(meta) == 2460324.5


def test_wfc_time_converts_mjd_start_to_bjd_mid(tmp_path):
    """MJD-OBS at exposure start becomes BJD_TDB at mid-exposure."""
    from astropy.coordinates import SkyCoord
    p = tmp_path / "f.fits"
    make_wfc_file(p)
    inst = Instrument(WFC_CFG)
    meta = inst.read_meta(p)
    jd_utc_start = 60324.9 + 2400000.5

    # A target on the ecliptic pole has almost no light-travel correction, so
    # the offset is just half the exposure plus TDB-UTC (about 69 s in 2024).
    pole = SkyCoord(lon=0 * u.deg, lat=90 * u.deg, frame="barycentricmeanecliptic")
    offset_pole = (inst.compute_bjd(meta, pole) - jd_utc_start) * 86400.0
    assert offset_pole == pytest.approx(30.0 + 69.18, abs=2.0)

    # A target in the ecliptic plane picks up light travel of up to 8.3 min,
    # which may be of either sign.
    kelt = SkyCoord("20h57m04.44s", "+31d39m39.6s")
    offset = (inst.compute_bjd(meta, kelt) - jd_utc_start) * 86400.0
    assert abs(offset) < 600.0
    assert offset != pytest.approx(offset_pole, abs=1.0)   # correction applied


def test_missing_time_keyword_is_an_error(tmp_path):
    cfg = dict(SPEC_CFG)
    cfg["time"] = {"keyword": "NOSUCH-TIME", "format": "jd", "scale": "tdb"}
    p = tmp_path / "f.fits"
    make_speculoos_file(p)
    inst = Instrument(cfg)
    with pytest.raises(KeyError, match="no time keyword"):
        inst.compute_bjd(inst.read_meta(p))


# --------------------------------------------------------------- step recipe

def test_default_recipe_is_the_legacy_chain():
    assert Instrument(SPEC_CFG).steps == ["bias", "dark", "flat"]


@pytest.mark.parametrize("steps,match", [
    (["bias", "overscan", "flat"], "must be the first"),
    (["bias", "dark", "flat", "nonsense"], "unknown reduction step"),
    (["dark", "flat"], "requires 'bias'"),
    (["overscan", "bias", "fringe", "flat"], "must follow 'flat'"),
    (["bias", "bias"], "duplicate"),
])
def test_bad_recipes_are_rejected(steps, match):
    cfg = dict(WFC_CFG)
    cfg["reduction"] = {"steps": steps}
    with pytest.raises(ValueError, match=match):
        Instrument(cfg)


def test_overscan_step_needs_an_overscan_section():
    cfg = dict(SPEC_CFG)
    cfg["reduction"] = {"steps": ["overscan", "bias"]}
    with pytest.raises(ValueError, match="no overscan section"):
        Instrument(cfg)


# --------------------------------------------------------------- coordinates

def test_raw_coordinates_are_converted_once():
    inst = Instrument(WFC_CFG)                       # trim starts at x=11 (1-based)
    assert inst.detector.trim_offset == (10, 0)
    out = inst.to_trimmed_coords([[263.0, 1924.6], [100.0, 50.0]])
    assert out[0] == pytest.approx([253.0, 1924.6])
    assert out[1] == pytest.approx([90.0, 50.0])


def test_trimmed_coordinates_are_left_alone():
    cfg = dict(WFC_CFG)
    cfg["coordinates"] = "trimmed"
    inst = Instrument(cfg)
    out = inst.to_trimmed_coords([[263.0, 1924.6]])
    assert out[0] == pytest.approx([263.0, 1924.6])


def test_legacy_instrument_conversion_is_a_no_op():
    inst = Instrument(SPEC_CFG)
    out = inst.to_trimmed_coords([[64.0, 64.0]])
    assert out[0] == pytest.approx([64.0, 64.0])


# ------------------------------------------------------------- dated entries

DATED = {
    "maps": {
        "z": {"CCD4": [
            {"start_date": "20170801", "path": "/a.fits"},
            {"start_date": "20200101", "path": "/b.fits"},
        ]}
    }
}


@pytest.mark.parametrize("date,expected", [
    ("20170805", "/a.fits"),
    ("20170801", "/a.fits"),      # boundary: inclusive
    ("20191231", "/a.fits"),
    ("20200101", "/b.fits"),
    ("20250101", "/b.fits"),
])
def test_find_dated_entry(date, expected):
    e = find_dated_entry(DATED, ["z", "CCD4"], date, root="maps")
    assert e["path"] == expected


def test_find_dated_entry_before_any_range():
    assert find_dated_entry(DATED, ["z", "CCD4"], "20100101", root="maps") is None


def test_find_dated_entry_unknown_keys():
    assert find_dated_entry(DATED, ["i", "CCD4"], "20180101", root="maps") is None
    assert find_dated_entry(DATED, ["z", "CCD1"], "20180101", root="maps") is None
    assert find_dated_entry(None, ["z"], "20180101") is None


# ------------------------------------------------------------------ loading

def test_load_instrument_roundtrip(tmp_path):
    (tmp_path / "WFCTEST.yaml").write_text(yaml.safe_dump(WFC_CFG))
    inst = load_instrument("WFCTEST", tmp_path)
    assert inst.detector.name == "CCD4"
    inst1 = load_instrument("WFCTEST", tmp_path, detector="CCD1")
    assert inst1.detector.name == "CCD1"


def test_load_instrument_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_instrument("NOPE", tmp_path)


def test_raw_dir_templates(tmp_path):
    assert Instrument(SPEC_CFG).raw_dir("/top", "20240115") == \
        Path("/top/Observations/SYNTH/images/20240115")
    assert Instrument(WFC_CFG).raw_dir("/top", "20240115") == \
        Path("/top/WFCDATA/20240115/download_dir")


# ---------------------------------------------------------------- reduction

def test_overscan_scalar_and_row_forms():
    img = np.zeros((10, 20))
    img[:, 15:20] = 5.0
    img[:, :15] = 100.0
    section = (slice(0, 10), slice(15, 20))
    assert reduction.overscan_level(img, section, "median") == 5.0
    row = reduction.overscan_level(img, section, "row_median")
    assert row.shape == (10,) and np.all(row == 5.0)
    out = reduction.apply_overscan(img, 5.0)
    assert np.all(out[:, :15] == 95.0)


def test_reducer_applies_steps_in_order():
    cfg = dict(SPEC_CFG)
    cfg["reduction"] = {"steps": ["bias", "dark", "flat"]}
    inst = Instrument(cfg)
    from instrument import Frame, FrameMeta
    ny, nx = 8, 8
    bias = np.full((ny, nx), 10.0)
    dark = np.full((ny, nx), 0.5)
    flat = np.full((ny, nx), 2.0)
    raw = bias + dark * 20.0 + flat * 300.0
    meta = FrameMeta(path=Path("x.fits"), frame_type=FrameType.SCIENCE,
                     object="t", filter="z", exptime=20.0, bjd_mid=0.0,
                     airmass=1.0, altitude=None, header=fits.Header())
    r = reduction.Reducer(inst, {"bias": bias, "dark": dark, "flat": flat})
    out, rec = r.reduce(Frame(meta=meta, data=raw))
    assert np.allclose(out, 300.0)
    assert rec == {}


def test_reducer_reports_missing_products():
    inst = Instrument(SPEC_CFG)
    with pytest.raises(ValueError, match="master dark"):
        reduction.Reducer(inst, {"bias": np.zeros((4, 4)),
                                 "flat": np.ones((4, 4))})


def test_reducer_trims_and_overscans():
    inst = Instrument(WFC_CFG)
    from instrument import Frame, FrameMeta
    ny, nx = 200, 160
    raw = np.full((ny, nx), 1000.0)
    raw[:, 145:160] = 900.0                 # overscan level
    bias = np.zeros((200, 130))
    flat = np.ones((200, 130))
    template = np.zeros((200, 130))
    meta = FrameMeta(path=Path("x.fits"), frame_type=FrameType.SCIENCE,
                     object="t", filter="z", exptime=60.0, bjd_mid=0.0,
                     airmass=1.0, altitude=None, header=fits.Header())
    r = reduction.Reducer(inst, {"bias": bias, "flat": flat},
                          fringe_template=template)
    out, rec = r.reduce(Frame(meta=meta, data=raw))
    assert out.shape == (200, 130)          # trimmed
    assert rec["overscan_level"] == 900.0
    assert np.allclose(out, 100.0)          # 1000 - 900, no fringe to remove


# ------------------------------------------------------------------- fringe

def _fringe(ny=160, nx=140, amp=8.0):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    p = np.sin(xx / 9.0) + np.cos(yy / 7.0)
    p -= p.mean()
    return amp * p / np.abs(p).std()


def test_fit_fringe_scale_recovers_a_planted_scale():
    rng = np.random.default_rng(0)
    t = _fringe()
    for true_scale in (0.5, 1.0, 2.5):
        img = true_scale * t + 200.0 + rng.normal(0, 1.0, t.shape)
        scale, offset, n = reduction.fit_fringe_scale(img - 200.0, t)
        assert scale == pytest.approx(true_scale, rel=0.01)


def test_fit_fringe_scale_with_a_plane_gradient():
    rng = np.random.default_rng(1)
    t = _fringe()
    ny, nx = t.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    grad = 0.02 * xx - 0.01 * yy
    img = 1.7 * t + grad + rng.normal(0, 1.0, t.shape)
    scale, _, _ = reduction.fit_fringe_scale(img, t, plane=True)
    assert scale == pytest.approx(1.7, rel=0.02)


def test_template_noise_attenuates_the_scale_and_smoothing_fixes_it():
    """The reason prepare_fringe_template smooths: regression attenuation.

    Fitting against a template that carries its own pixel noise biases the
    recovered slope low by var(true)/(var(true)+var(noise)).  On the real
    INT/WFC map that cost 15 to 25 per cent.
    """
    rng = np.random.default_rng(2)
    truth = _fringe()
    noise_sigma = truth.std()            # attenuation should be around 0.5
    noisy_template = truth + rng.normal(0, noise_sigma, truth.shape)
    img = 1.0 * truth + rng.normal(0, 1.0, truth.shape)
    raw_scale, _, _ = reduction.fit_fringe_scale(img, noisy_template)
    assert raw_scale < 0.9                      # biased low, as observed
    smoothed = reduction.prepare_fringe_template(
        noisy_template, highpass_sigma=0, smooth_sigma=2.0)
    fixed, _, _ = reduction.fit_fringe_scale(img, smoothed)
    assert fixed > raw_scale
    assert abs(fixed - 1.0) < abs(raw_scale - 1.0)


def test_fit_fringe_scale_rejects_a_mismatched_template():
    with pytest.raises(ValueError, match="does not match"):
        reduction.fit_fringe_scale(np.zeros((10, 10)), np.zeros((10, 11)))


def test_fit_fringe_scale_needs_enough_sky():
    t = _fringe(20, 20)
    mask = np.zeros((20, 20), bool)
    with pytest.raises(ValueError, match="usable sky pixels"):
        reduction.fit_fringe_scale(t, t, mask=mask)


def test_star_mask_excludes_stars_and_bad_columns():
    img = np.full((100, 100), 10.0)
    img += np.random.default_rng(3).normal(0, 1.0, (100, 100))
    img[50, 50] = 5000.0
    mask = reduction.build_star_mask(img, dilate=5, border=5, bad_columns=[20])
    assert not mask[50, 50]
    assert not mask[:, 20].any()
    assert not mask[0, :].any()
    assert mask.mean() > 0.5


def test_build_fringe_map_rejects_dithered_stars():
    rng = np.random.default_rng(4)
    truth = _fringe()
    frames = []
    for i in range(9):
        f = truth + 300.0 + rng.normal(0, 1.0, truth.shape)
        f[20 + 6 * i, 30 + 5 * i] += 8000.0        # a star, dithered
        frames.append(f)
    m = reduction.build_fringe_map(frames, highpass_sigma=0)
    assert m.max() < 100.0                          # star gone
    assert np.corrcoef(m.ravel(), truth.ravel())[0, 1] > 0.99


# ----------------------------------------------------- clipped combination

def _medabsdevclip_reference(data, clip, nlimit):
    """The per-pixel Python loop that the vectorised version replaced."""
    from utils import mad
    nx, ny, nz = data.shape
    out = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            line = data[i, j, :]
            if nlimit >= 0:
                s1 = np.argsort(line)
                trimmed = line[s1[:nz - nlimit - 1]]
                sig, med = mad(trimmed, sigma=True), np.median(trimmed)
            else:
                sig, med = mad(line, sigma=True), np.median(line)
            if sig == 0:
                sig = 1e-10
            w = np.where(np.abs(line - med) / sig <= clip)[0]
            out[i, j] = np.mean(line[w]) if len(w) else med
    return out


@pytest.mark.parametrize("shape,clip,nlimit", [
    ((30, 25, 9), 5, 5),
    ((20, 20, 21), 3, 5),
    ((15, 15, 5), 5, 0),
    ((12, 12, 4), 5, 5),      # nlimit > nz: the negative-count slice
    ((10, 10, 8), 2, -1),     # nlimit < 0: no trim at all
])
def test_medabsdevclip_matches_the_original_loop(shape, clip, nlimit):
    """Vectorising must not change the numbers, only the speed."""
    from utils import medabsdevclip
    rng = np.random.default_rng(7)
    cube = rng.normal(100, 5, shape)
    cube[3, 3, :2] += 900          # a couple of outliers to clip
    cube[5, 5, 0] -= 400
    expected = _medabsdevclip_reference(cube, clip, nlimit)
    got = medabsdevclip(cube, clip, nlimit)
    # Summation order differs, so agreement is to float64 rounding, not bitwise.
    assert np.allclose(got, expected, rtol=1e-12, atol=0)


def test_medabsdevclip_rejects_an_impossible_trim():
    """nlimit >= nz used to yield an all-NaN master frame, silently."""
    from utils import medabsdevclip
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="leaves no values to combine"):
        medabsdevclip(rng.normal(0, 1, (4, 4, 6)), 5, 5)


def test_medabsdevclip_clips_a_planted_outlier():
    from utils import medabsdevclip
    rng = np.random.default_rng(2)
    cube = rng.normal(50.0, 0.5, (8, 8, 15))
    cube[4, 4, 7] = 5000.0
    out = medabsdevclip(cube, 3, 0)
    assert out[4, 4] == pytest.approx(50.0, abs=1.0)


def test_medabsdevclip_2d_returns_mean_and_std():
    from utils import medabsdevclip
    rng = np.random.default_rng(3)
    out = medabsdevclip(rng.normal(10.0, 2.0, (60, 60)), 5, 0)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(10.0, abs=0.1)
    assert out[1] == pytest.approx(2.0, abs=0.1)


# ------------------------------------------------- fringe map construction

def test_combine_fringe_frames_removes_per_frame_sky():
    """Frames at different sky levels must combine without a level offset."""
    rng = np.random.default_rng(11)
    truth = _fringe(80, 70)
    frames = [truth + sky + rng.normal(0, 0.5, truth.shape)
              for sky in (100.0, 250.0, 400.0, 180.0, 320.0)]
    out = reduction.combine_fringe_frames(frames)
    assert abs(np.median(out)) < 1.0
    assert np.corrcoef(out.ravel(), truth.ravel())[0, 1] > 0.99


def test_highpass_leaves_the_fringe_and_removes_the_gradient():
    truth = _fringe(200, 200)
    ny, nx = truth.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    contaminated = truth + 0.05 * xx + 0.03 * yy
    out = reduction.highpass(contaminated, 40)
    assert np.corrcoef(out.ravel(), truth.ravel())[0, 1] > 0.95
    assert np.ptp(out) < np.ptp(contaminated)


def test_build_fringe_map_equals_combine_then_highpass():
    """make_fringe_map derives the filtered map from the unfiltered one."""
    rng = np.random.default_rng(12)
    frames = [_fringe(60, 50) + 100.0 + rng.normal(0, 0.5, (60, 50))
              for _ in range(7)]
    combined = reduction.combine_fringe_frames(frames)
    assert np.allclose(reduction.build_fringe_map(frames, 20),
                       reduction.highpass(combined, 20))


def test_steps_before():
    inst = Instrument(WFC_CFG)      # overscan, bias, flat, fringe
    assert reduction.steps_before(inst, "flat") == ["overscan", "bias"]
    assert reduction.steps_before(inst, "fringe") == ["overscan", "bias", "flat"]
    assert reduction.steps_before(inst, "bias") == ["overscan"]
    # a step the recipe does not contain leaves the chain unchanged
    assert reduction.steps_before(inst, "dark") == inst.steps


def test_fringe_map_provenance_is_checked(tmp_path):
    """A map from the wrong chip must be refused, not applied."""
    from master_calibrations import write_fringe_map, load_fringe_map
    inst = Instrument(WFC_CFG)                       # CCD4
    template = _fringe(40, 30)
    path = tmp_path / "map.fits"
    write_fringe_map(path, template, template, inst, "z", 12)

    assert load_fringe_map(path, inst, "z").shape == template.shape

    with pytest.raises(ValueError, match="FILTER"):
        load_fringe_map(path, inst, "i")
    other = Instrument(WFC_CFG, detector_name="CCD1")
    with pytest.raises(ValueError, match="DETECTOR"):
        load_fringe_map(path, other, "z")
    with pytest.raises(ValueError, match="shape"):
        load_fringe_map(path, inst, "z", expected_shape=(99, 99))


def test_large_scale_matches_the_direct_filter():
    """The decimated background estimate must track the direct Gaussian."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(21)
    ny, nx = 512, 512
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    background = 50 * np.sin(xx / 300) + 30 * np.cos(yy / 250)
    fringe = 8 * np.sin(xx / 9) + 8 * np.cos(yy / 7)
    img = background + fringe + rng.normal(0, 1.0, (ny, nx))

    direct = img - gaussian_filter(img, 120, mode="nearest")
    fast = reduction.highpass(img, 120)
    assert fast.shape == img.shape
    # Agreement to a few per cent of the fringe amplitude is what matters:
    # the high-pass cut is a definitional choice, not a measurement.
    assert np.std(direct - fast) < 0.1 * np.std(fringe)
    assert np.corrcoef(fast.ravel(), fringe.ravel())[0, 1] > 0.85


def test_large_scale_uses_the_direct_filter_for_small_sigma():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(22)
    img = rng.normal(0, 1, (64, 64))
    assert np.allclose(reduction.large_scale(img, 4.0),
                       gaussian_filter(img, 4.0, mode="nearest"))


def test_large_scale_of_zero_sigma_is_zero():
    img = np.ones((10, 10))
    assert np.all(reduction.large_scale(img, 0) == 0)
    assert np.allclose(reduction.highpass(img, 0), img)


@pytest.mark.parametrize("shape", [(200, 130), (301, 199), (97, 512)])
def test_large_scale_preserves_shape(shape):
    rng = np.random.default_rng(23)
    img = rng.normal(0, 1, shape)
    assert reduction.large_scale(img, 120).shape == shape
