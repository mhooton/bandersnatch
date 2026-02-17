#!/usr/bin/env python3
"""
Bad pixel map creation module.
Detects hot pixels, cold pixels, and flat-field bad pixels.
"""

import numpy as np
import logging
from astropy.io import fits
from pathlib import Path

# Set up logger for this module
logger = logging.getLogger(__name__)


def detect_bad_pixels(dark, hot_sigma=5, cold_sigma=5):
    """
    Detect hot and cold pixels as outliers in the dark frame.

    Parameters:
    -----------
    dark : 2D array
        Bias-subtracted master dark (or shortest exposure dark)
    hot_sigma : float
        Number of standard deviations above median to flag hot pixels (default 5)
    cold_sigma : float
        Number of standard deviations below median to flag cold pixels (default 5)

    Returns:
    --------
    hot_pixel_map : 2D boolean array
        True = hot pixel
    cold_pixel_map : 2D boolean array
        True = cold pixel
    """
    logger.debug("Detecting hot and cold pixels with thresholds: hot_sigma=%.1f, cold_sigma=%.1f",
                 hot_sigma, cold_sigma)

    # Calculate threshold
    dark_median = np.median(dark)
    dark_std = np.std(dark)

    hot_threshold = dark_median + hot_sigma * dark_std
    cold_threshold = dark_median - cold_sigma * dark_std

    logger.debug("Dark statistics: median=%.2f, std=%.2f", dark_median, dark_std)
    logger.debug("Thresholds: hot=%.2f, cold=%.2f", hot_threshold, cold_threshold)

    # Flag pixels above/below thresholds
    hot_pixel_map = dark > hot_threshold
    cold_pixel_map = dark < cold_threshold

    n_hot = np.sum(hot_pixel_map)
    n_cold = np.sum(cold_pixel_map)
    logger.debug("Detected %d hot pixels and %d cold pixels", n_hot, n_cold)

    return hot_pixel_map, cold_pixel_map


def detect_flatbad_pixels(master_flat, flat_threshold):
    """
    Detect flatbad pixels with low response in the master flat.

    Parameters:
    -----------
    master_flat : 2D array
        Normalized master flat field
    flat_threshold : float
        Absolute threshold for flat response (e.g., 0.1)

    Returns:
    --------
    flatbad_map : 2D boolean array
        True = flatbad pixel
    """
    logger.debug("Detecting flatbad pixels with threshold: %.3f", flat_threshold)

    # Flag pixels below threshold
    flatbad_map = master_flat < flat_threshold

    n_flatbad = np.sum(flatbad_map)
    logger.debug("Detected %d flatbad pixels", n_flatbad)

    return flatbad_map


def make_bad_pixel_map(outdir, run, master_bias, master_dark, master_flat, config):
    """
    Create bad pixel map from master calibration frames.

    Parameters:
    -----------
    outdir : Path
        Output directory containing calibration files
    run : str
        Run identifier
    master_bias : 2D array
        Master bias frame
    master_dark : 2D array
        Master dark frame
    master_flat : 2D array
        Master flat frame (for the preferred filter)
    config : dict
        Configuration dictionary containing instrument settings

    Returns:
    --------
    bad_pixel_map : 2D boolean array
        Combined bad pixel map
    """
    instrument_name = config['instrument_settings']['inst']
    logger.info("Creating bad pixel map for %s", instrument_name)

    # Extract configuration
    inst_config = config['instrument_config']
    bpm_config = inst_config.get('bad_pixel_correction', None)

    if bpm_config is None:
        logger.info("No bad_pixel_correction configuration found. Skipping BPM creation.")
        return None

    # Get parameters
    hot_sigma = bpm_config.get('hot_sigma_threshold', 5)
    cold_sigma = bpm_config.get('cold_sigma_threshold', 5)
    flat_threshold = bpm_config.get('flat_threshold', 0.1)
    overwrite = bpm_config.get('overwrite_existing', False)

    logger.info("BPM parameters: hot_sigma=%.1f, cold_sigma=%.1f, flat_threshold=%.3f",
                hot_sigma, cold_sigma, flat_threshold)

    # Set up paths
    caldir = outdir / "calib"
    bpm_path = caldir / "bad_pixel_map.fits"

    # Check if BPM already exists
    if bpm_path.exists() and not overwrite:
        logger.info("Bad pixel map already exists and overwrite=False. Loading existing BPM.")
        with fits.open(bpm_path) as hdul:
            bad_pixel_map = hdul[0].data.astype(bool)

        n_bad = np.sum(bad_pixel_map)
        total_pixels = bad_pixel_map.size
        pct_bad = 100 * n_bad / total_pixels
        logger.info("Loaded existing BPM: %d bad pixels (%.4f%%)", n_bad, pct_bad)

        return bad_pixel_map

    # Subtract bias from dark
    dark_bias_subtracted = master_dark - master_bias

    # Detect hot and cold pixels
    logger.info("Detecting hot and cold pixels...")
    hot_pixel_map, cold_pixel_map = detect_bad_pixels(
        dark_bias_subtracted,
        hot_sigma=hot_sigma,
        cold_sigma=cold_sigma
    )

    # Detect flatbad pixels
    logger.info("Detecting flatbad pixels...")
    flatbad_pixel_map = detect_flatbad_pixels(master_flat, flat_threshold)

    # Combine all maps
    bad_pixel_map = hot_pixel_map | cold_pixel_map | flatbad_pixel_map

    # Calculate and log statistics
    total_pixels = bad_pixel_map.size

    logger.info("=" * 60)
    logger.info("BAD PIXEL STATISTICS")
    logger.info("=" * 60)

    maps = {
        'Hot': hot_pixel_map,
        'Cold': cold_pixel_map,
        'Flatbad': flatbad_pixel_map,
        'Total': bad_pixel_map
    }

    for name, pixel_map in maps.items():
        n_bad = np.sum(pixel_map)
        pct_bad = 100 * n_bad / total_pixels
        logger.info("%s: %d pixels (%.4f%%)", name, n_bad, pct_bad)

    logger.info("-" * 60)
    logger.info("UNIQUE BAD PIXELS (not flagged by other maps)")
    logger.info("-" * 60)

    # Calculate unique pixels for each type
    unique_maps = {
        'Hot': hot_pixel_map & ~(cold_pixel_map | flatbad_pixel_map),
        'Cold': cold_pixel_map & ~(hot_pixel_map | flatbad_pixel_map),
        'Flatbad': flatbad_pixel_map & ~(hot_pixel_map | cold_pixel_map)
    }

    for name, unique_map in unique_maps.items():
        n_unique = np.sum(unique_map)
        n_total_for_type = np.sum(maps[name])
        pct_of_total_pixels = 100 * n_unique / total_pixels
        pct_of_category = 100 * n_unique / n_total_for_type if n_total_for_type > 0 else 0
        logger.info("%s: %d pixels (%.4f%% of all, %.2f%% of %s)",
                    name, n_unique, pct_of_total_pixels, pct_of_category, name.lower())

    logger.info("=" * 60)

    # Save bad pixel map
    hdu = fits.PrimaryHDU(bad_pixel_map.astype(np.uint8))
    hdu.header['INSTRUME'] = instrument_name
    hdu.header['BPMTYPE'] = 'COMBINED'
    hdu.header['HOTSIGMA'] = (hot_sigma, 'Hot pixel sigma threshold')
    hdu.header['COLDSIGM'] = (cold_sigma, 'Cold pixel sigma threshold')
    hdu.header['FLATTHRE'] = (flat_threshold, 'Flat field threshold')

    hdu.writeto(bpm_path, overwrite=True)
    logger.info("Bad pixel map saved to %s", bpm_path)

    return bad_pixel_map


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running bad_pixel_map.py as standalone script")