"""Tests for aper.py: the exact (PIXWT) and approximate aperture weightings."""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aper import aper, pixwt  # noqa: E402


# --------------------------------------------------------------------- helpers

def gaussian_star(ny, nx, x, y, flux, fwhm):
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[0:ny, 0:nx]
    return flux / (2 * np.pi * sigma ** 2) * np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def brute_force_pixwt(xc, yc, r, x, y, n=1000):
    """Fraction of the unit pixel centred on (x, y) inside the circle,
    by midpoint supersampling with n x n sub-pixels."""
    s = (np.arange(n) + 0.5) / n - 0.5
    sx, sy = np.meshgrid(s, s)
    inside = (x + sx - xc) ** 2 + (y + sy - yc) ** 2 < r * r
    return inside.mean()


def exact_aperture_sum(image, xc, yc, r):
    """Independent reference: sum of pixel values weighted by the exact
    geometric overlap of each pixel with the circle."""
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    w = pixwt(xc, yc, r, xx.ravel().astype(float), yy.ravel().astype(float))
    return np.sum(image.ravel() * np.clip(w, 0.0, None))


def measure(image, xc, yc, apr, exact, **kw):
    kw.setdefault("skyrad", [12.0, 16.0])
    kw.setdefault("setskyval", 0.0)
    flux, err, sky, skyerr = aper(image, xc, yc, apr=np.asarray(apr, float),
                                  flux=True, silent=True, exact=exact, **kw)
    return flux, err, sky, skyerr


# ---------------------------------------------------------------------- pixwt

def test_pixwt_fully_inside_and_fully_outside():
    assert pixwt(10.0, 10.0, 5.0, 10.0, 10.0) == pytest.approx(1.0)
    assert pixwt(10.0, 10.0, 5.0, 12.0, 11.0) == pytest.approx(1.0)
    assert pixwt(10.0, 10.0, 5.0, 20.0, 10.0) == pytest.approx(0.0, abs=1e-12)


def test_pixwt_inscribed_circle_is_pi_over_four():
    # circle of radius 1/2 centred on the pixel: exactly the inscribed disk
    assert pixwt(3.0, 7.0, 0.5, 3.0, 7.0) == pytest.approx(np.pi / 4, rel=1e-12)


@pytest.mark.parametrize("r", [0.2, 0.5, 0.9, 1.0])
def test_pixwt_quarter_disk_at_a_corner(r):
    # circle centred on the pixel's corner with r <= 1: a quarter disk
    assert pixwt(3.5, 7.5, r, 3.0, 7.0) == pytest.approx(np.pi * r * r / 4, rel=1e-12)


@pytest.mark.parametrize("r", [0.55, 0.6, 0.7])
def test_pixwt_circle_overlapping_all_four_sides(r):
    # centred on the pixel, 0.5 < r < 0.5*sqrt(2): disk minus four segments
    segment = r * r * np.arccos(0.5 / r) - 0.5 * np.sqrt(r * r - 0.25)
    expected = np.pi * r * r - 4 * segment
    assert pixwt(0.0, 0.0, r, 0.0, 0.0) == pytest.approx(expected, rel=1e-12)


def test_pixwt_matches_brute_force_on_boundary_pixels():
    rng = np.random.default_rng(3)
    for _ in range(40):
        r = rng.uniform(1.0, 9.0)
        xc, yc = rng.uniform(-0.5, 0.5, 2)
        theta = rng.uniform(0, 2 * np.pi)
        # a pixel whose centre sits on the circle: genuinely partial
        x = np.round(xc + r * np.cos(theta))
        y = np.round(yc + r * np.sin(theta))
        w = pixwt(xc, yc, r, float(x), float(y))
        assert 0.0 <= w <= 1.0
        assert w == pytest.approx(brute_force_pixwt(xc, yc, r, x, y), abs=3e-3)


def test_pixwt_array_call_matches_scalar_calls():
    # the vectorised ONESIDE branches against the straightforward scalar port
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:30, 0:30]
    xs = xx.ravel().astype(float)
    ys = yy.ravel().astype(float)
    for _ in range(20):
        r = rng.uniform(0.7, 11.0)
        xc, yc = rng.uniform(10, 20, 2)
        vec = pixwt(xc, yc, r, xs, ys)
        scal = np.array([pixwt(xc, yc, r, x, y) for x, y in zip(xs, ys)])
        np.testing.assert_allclose(vec, scal, rtol=0, atol=1e-12)


def test_pixwt_sums_to_the_circle_area():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:40, 0:40]
    xs = xx.ravel().astype(float)
    ys = yy.ravel().astype(float)
    for _ in range(200):
        r = rng.uniform(1.0, 12.0)
        xc, yc = rng.uniform(15, 25, 2)
        total = pixwt(xc, yc, r, xs, ys).sum()
        assert total == pytest.approx(np.pi * r * r, rel=1e-12)


# ----------------------------------------------------------- aper, exact=True

@pytest.mark.parametrize("xc, yc", [(50.3, 30.7), (49.5, 30.5), (50.0, 30.0)])
def test_exact_matches_the_independent_pixwt_sum(xc, yc):
    image = 120.0 + gaussian_star(60, 100, xc, yc, 40000.0, 3.0)
    apr = [2.0, 3.0, 4.5, 6.0, 8.0]
    flux, _, _, _ = measure(image, xc, yc, apr, exact=True)
    expected = [exact_aperture_sum(image, xc, yc, r) for r in apr]
    np.testing.assert_allclose(flux[:, 0], expected, rtol=1e-10)


def test_exact_handles_a_clipped_non_square_subarray():
    # the sky annulus runs off the left and bottom edges, so the subarray
    # is ny x nx with nx != ny; this used to index the mask transposed
    image = 120.0 + gaussian_star(60, 100, 10.2, 14.4, 40000.0, 3.0)
    apr = [2.0, 3.0, 4.0]
    flux, _, _, _ = measure(image, 10.2, 14.4, apr, exact=True,
                            skyrad=[12.0, 16.0])
    expected = [exact_aperture_sum(image, 10.2, 14.4, r) for r in apr]
    np.testing.assert_allclose(flux[:, 0], expected, rtol=1e-10)


def test_exact_several_stars_and_apertures():
    xs = np.array([50.3, 20.1, 80.9])
    ys = np.array([30.7, 40.2, 20.6])
    image = np.full((60, 100), 120.0)
    for x, y, f in zip(xs, ys, (40000.0, 22000.0, 15000.0)):
        image += gaussian_star(60, 100, x, y, f, 3.0)
    apr = [2.0, 3.0, 5.0]
    flux, err, sky, skyerr = measure(image, xs, ys, apr, exact=True)
    assert flux.shape == err.shape == (3, 3)
    assert sky.shape == skyerr.shape == (3,)
    for j, (x, y) in enumerate(zip(xs, ys)):
        expected = [exact_aperture_sum(image, x, y, r) for r in apr]
        np.testing.assert_allclose(flux[:, j], expected, rtol=1e-10)


@pytest.mark.parametrize("exact", [True, False])
def test_flat_field_integrates_to_the_circle_area(exact):
    # both weightings return level * pi r^2 on a flat field: the approximate
    # method renormalises its partial pixels to the exact area, so this
    # check is necessary but does not distinguish the two
    image = np.full((60, 100), 10.0)
    apr = [3.0, 5.0, 7.5]
    for xc, yc in [(50.3, 30.7), (10.2, 30.4)]:
        flux, _, _, _ = measure(image, xc, yc, apr, exact=exact)
        np.testing.assert_allclose(flux[:, 0], 10.0 * np.pi * np.square(apr),
                                   rtol=1e-12)


def test_approximate_weighting_differs_from_exact_on_a_star():
    xc, yc = 50.3, 30.7
    image = 120.0 + gaussian_star(60, 100, xc, yc, 40000.0, 3.0)
    apr = [2.0, 3.0, 4.0, 6.0, 8.0]
    exact, _, _, _ = measure(image, xc, yc, apr, exact=True)
    approx, _, _, _ = measure(image, xc, yc, apr, exact=False)
    rel = np.abs((approx - exact) / exact)[:, 0]
    # where the PSF wings cross the aperture edge the weightings differ,
    # by a small amount; where the boundary pixels are flat sky they agree,
    # because the approximate method renormalises to the exact area
    assert np.all(rel[:3] > 1e-5) and np.all(rel[:3] < 1e-2)
    assert np.all(rel[3:] < 1e-6)


def test_sky_estimate_does_not_depend_on_the_weighting():
    rng = np.random.default_rng(11)
    image = 120.0 + gaussian_star(60, 100, 50.3, 30.7, 40000.0, 3.0)
    image += rng.normal(0, 4.0, image.shape)
    e = aper(image, 50.3, 30.7, apr=np.array([3.0, 5.0]), skyrad=[12.0, 16.0],
             flux=True, silent=True, exact=True)
    a = aper(image, 50.3, 30.7, apr=np.array([3.0, 5.0]), skyrad=[12.0, 16.0],
             flux=True, silent=True, exact=False)
    assert e[2][0] == a[2][0]          # sky
    assert e[3][0] == a[3][0]          # sky error
    assert e[2][0] == pytest.approx(120.0, abs=1.0)
    # and the sky-subtracted fluxes agree to well under a per cent
    np.testing.assert_allclose(e[0], a[0], rtol=5e-3)


@pytest.mark.parametrize("exact", [True, False])
def test_aperture_running_off_the_frame_is_flagged(exact):
    image = 120.0 + gaussian_star(60, 100, 50.0, 5.9, 40000.0, 3.0)
    flux, _, _, _ = measure(image, 50.0, 5.9, [3.0, 5.0, 7.5], exact=exact)
    assert np.isfinite(flux[0, 0]) and np.isfinite(flux[1, 0])
    assert np.isnan(flux[2, 0])


@pytest.mark.parametrize("exact", [True, False])
def test_nan_inside_the_aperture_is_flagged(exact):
    image = 120.0 + gaussian_star(60, 100, 50.3, 30.7, 40000.0, 3.0)
    image[31, 52] = np.nan
    flux, _, _, _ = measure(image, 50.3, 30.7, [1.0, 4.0], exact=exact,
                            nan=True)
    assert np.isfinite(flux[0, 0])      # r = 1 does not reach the bad pixel
    assert np.isnan(flux[1, 0])         # r = 4 does
