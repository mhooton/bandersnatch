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


def make_bad_pixel_map(outdir, run, inst, config):
    """
    Create bad pixel map from master calibration files.

    Parameters:
    -----------
    outdir : Path
        Output directory containing calibration files
    run : str
        Run identifier
    inst : str
        Instrument name
    config : dict
        Configuration dictionary containing instrument settings

    Returns:
    --------
    bad_pixel_map : 2D boolean array
        Combined bad pixel map
    """
    logger.info("Creating bad pixel map for %s", inst)

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
    preferred_filter = config.get('bad_pixel_correction', {}).get('preferred_flat_filter', None)
    overwrite = bpm_config.get('overwrite_existing', False)

    logger.info("BPM parameters: hot_sigma=%.1f, cold_sigma=%.1f, flat_threshold=%.3f",
                hot_sigma, cold_sigma, flat_threshold)

    if preferred_filter is None:
        logger.error("No preferred_flat_filter specified in date configuration")
        raise ValueError("preferred_flat_filter must be specified in date configuration for BPM creation")

    logger.info("Using preferred flat filter: %s", preferred_filter)

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

    # Load master bias
    bias_path = caldir / f"MasterBias.fits"
    if not bias_path.exists():
        logger.error("Master bias not found at %s", bias_path)
        raise FileNotFoundError(f"Master bias required for BPM creation: {bias_path}")

    logger.debug("Loading master bias from %s", bias_path)
    with fits.open(bias_path) as hdul:
        master_bias = hdul[0].data

    # Load master dark (use shortest exposure or combined)
    dark_files = list(caldir.glob("MasterDark*.fits"))
    if not dark_files:
        logger.error("No master dark files found in %s", caldir)
        raise FileNotFoundError("Master dark required for BPM creation")

    # Try to find shortest exposure dark
    dark_exposures = []
    for dark_file in dark_files:
        if "MasterDark_" in dark_file.name:
            try:
                exp_str = dark_file.name.split("_")[1].replace("s.fits", "")
                exp_time = int(exp_str)
                dark_exposures.append((exp_time, dark_file))
            except:
                pass

    if dark_exposures:
        # Use shortest exposure dark
        dark_exposures.sort(key=lambda x: x[0])
        dark_path = dark_exposures[0][1]
        logger.info("Using dark with exposure time %ds for BPM creation", dark_exposures[0][0])
    else:
        # Use combined dark
        dark_path = caldir / "MasterDark.fits"
        if not dark_path.exists():
            logger.error("No suitable master dark found")
            raise FileNotFoundError("Master dark required for BPM creation")
        logger.info("Using combined master dark for BPM creation")

    logger.debug("Loading master dark from %s", dark_path)
    with fits.open(dark_path) as hdul:
        master_dark = hdul[0].data

    # Subtract bias from dark
    dark_bias_subtracted = master_dark - master_bias

    # Load master flat for preferred filter
    flat_path = caldir / f"MasterFlat_{preferred_filter}.fits"
    if not flat_path.exists():
        logger.error("Master flat for filter '%s' not found at %s", preferred_filter, flat_path)
        raise FileNotFoundError(f"Master flat for preferred filter '{preferred_filter}' not found. "
                                f"Cannot create BPM without this flat.")

    logger.debug("Loading master flat from %s", flat_path)
    with fits.open(flat_path) as hdul:
        master_flat = hdul[0].data

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
    hdu.header['INSTRUME'] = inst
    hdu.header['BPMTYPE'] = 'COMBINED'
    hdu.header['HOTSIGMA'] = (hot_sigma, 'Hot pixel sigma threshold')
    hdu.header['COLDSIGM'] = (cold_sigma, 'Cold pixel sigma threshold')
    hdu.header['FLATTHRE'] = (flat_threshold, 'Flat field threshold')
    hdu.header['FILTER'] = (preferred_filter, 'Filter used for flatbad detection')
    hdu.header['DARKFILE'] = (dark_path.name, 'Dark file used')
    hdu.header['FLATFILE'] = (flat_path.name, 'Flat file used')
    hdu.header['BIASFILE'] = (bias_path.name, 'Bias file used')

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