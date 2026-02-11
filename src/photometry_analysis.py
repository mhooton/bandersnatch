# NEW FILE: photometry_analysis.py
"""
Photometry analysis functions for precision calculation and summary generation.
Shared by both photometry.py and streaming_processor.py
"""

import numpy as np
import logging
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger(__name__)


def rolling_window_outlier_detection(data, window_size, sigma_threshold=3.0):
    """
    Detect outliers using rolling window statistics.

    Parameters:
    -----------
    data : np.ndarray
        1D array of data values
    window_size : int
        Size of rolling window
    sigma_threshold : float
        Number of sigma for outlier threshold

    Returns:
    --------
    np.ndarray : Boolean mask where True indicates outliers
    """
    n_points = len(data)
    outlier_mask = np.zeros(n_points, dtype=bool)

    logger.debug("Running outlier detection: %d points, window=%d, threshold=%.1f",
                 n_points, window_size, sigma_threshold)

    # Handle edge cases
    if n_points < window_size:
        # For short datasets, fall back to global statistics
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val)) * 1.4826
        outlier_mask = np.abs(data - median_val) > sigma_threshold * mad
        n_outliers = np.sum(outlier_mask)
        logger.debug("Short dataset fallback: %d outliers detected", n_outliers)
        return outlier_mask

    half_window = window_size // 2

    for i in range(n_points):
        # Define window boundaries
        start_idx = max(0, i - half_window)
        end_idx = min(n_points, i + half_window + 1)

        # Extract window data
        window_data = data[start_idx:end_idx]

        # Calculate local statistics
        local_median = np.median(window_data)
        local_mad = np.median(np.abs(window_data - local_median)) * 1.4826

        # Avoid division by zero
        if local_mad == 0:
            local_mad = np.std(window_data)
            if local_mad == 0:
                continue  # Skip if no variation in window

        # Check if current point is an outlier
        deviation = np.abs(data[i] - local_median)
        if deviation > sigma_threshold * local_mad:
            outlier_mask[i] = True

    n_outliers = np.sum(outlier_mask)
    logger.debug("Rolling window outlier detection: %d outliers detected", n_outliers)

    return outlier_mask


def median_filter_interpolate(flux_data, outlier_mask, window_size):
    """
    Interpolate outliers using median filter.

    Parameters:
    -----------
    flux_data : np.ndarray
        1D array of flux values
    outlier_mask : np.ndarray
        Boolean mask where True indicates outliers
    window_size : int
        Size of median filter window

    Returns:
    --------
    np.ndarray : Flux data with outliers replaced by median-filtered values
    """
    logger.debug("Interpolating %d outliers with window size %d", np.sum(outlier_mask), window_size)

    from scipy import ndimage

    flux_clean = flux_data.copy()

    # Apply median filter to entire series
    flux_filtered = ndimage.median_filter(flux_data, size=window_size, mode='reflect')

    # Replace outliers with filtered values
    flux_clean[outlier_mask] = flux_filtered[outlier_mask]

    return flux_clean


def process_aperture_photometry(flux_array, flux_err_array, sky_array, sky_err_array,
                                bjd, file_paths, aper_radius, median_filter_window):
    """
    Complete aperture photometry processing pipeline.

    Parameters:
    -----------
    flux_array : np.ndarray
        Flux data (n_images, n_stars)
    flux_err_array : np.ndarray
        Flux error data (n_images, n_stars)
    sky_array : np.ndarray
        Sky data (n_images, n_stars)
    sky_err_array : np.ndarray
        Sky error data (n_images, n_stars)
    bjd : np.ndarray
        Time data
    file_paths : list
        File paths
    aper_radius : float
        Aperture radius
    median_filter_window : int
        Window size for outlier detection

    Returns:
    --------
    dict : Processed photometry data
    """
    n_images, n_stars = flux_array.shape
    aper_name = f"aper{int(aper_radius)}"

    logger.info("Processing aperture %s with %d images, %d stars", aper_name, n_images, n_stars)

    # Step 1: Initial outlier detection
    outlier_flags, frames_to_remove, flux_cleaned, flux_err_cleaned = detect_and_clean_outliers(
        flux_array, flux_err_array, median_filter_window)

    # Step 2: Remove bad frames
    good_frames = ~frames_to_remove
    n_good_frames = np.sum(good_frames)

    logger.info("Aperture %s: keeping %d/%d frames after outlier removal",
                aper_name, n_good_frames, n_images)

    if n_good_frames < 10:
        logger.warning("Aperture %s: Very few frames remaining (%d)", aper_name, n_good_frames)

    # Extract good data
    flux_final = flux_cleaned[good_frames, :]
    flux_err_final = flux_err_cleaned[good_frames, :]
    sky_final = sky_array[good_frames, :]
    sky_err_final = sky_err_array[good_frames, :]
    bjd_final = bjd[good_frames]
    file_paths_final = [file_paths[i] for i in range(n_images) if good_frames[i]]
    outlier_flags_final = outlier_flags[good_frames]

    # Step 3: Calculate normalized flux
    flux_norm, flux_norm_err = calculate_normalized_flux(flux_final, flux_err_final)

    # Step 4: Differential photometry with optimization
    diff_flux = None
    diff_flux_err = None
    good_star_mask = np.ones(n_stars, dtype=bool)  # Default: all stars good

    if n_stars > 1:
        logger.debug("Computing differential photometry for aperture %s", aper_name)

        target_flux = flux_final[:, 0]
        target_flux_err = flux_err_final[:, 0]
        comparison_flux = flux_final[:, 1:]
        comparison_flux_err = flux_err_final[:, 1:]

        # Use shared optimization function
        good_comparisons, diff_flux, diff_flux_err, good_star_mask = optimize_comparison_stars(
            target_flux, target_flux_err, comparison_flux, comparison_flux_err, median_filter_window)

        # Step 5: Secondary differential outlier cleaning
        diff_flux, diff_flux_err, secondary_good_frames = apply_secondary_differential_cleaning(
            diff_flux, diff_flux_err, median_filter_window)

        if np.any(~secondary_good_frames):
            n_secondary_removed = np.sum(~secondary_good_frames)
            logger.info("Aperture %s: removing %d secondary differential outliers", aper_name, n_secondary_removed)

            # Apply secondary cleaning to all arrays
            flux_final = flux_final[secondary_good_frames, :]
            flux_err_final = flux_err_final[secondary_good_frames, :]
            sky_final = sky_final[secondary_good_frames, :]
            sky_err_final = sky_err_final[secondary_good_frames, :]
            bjd_final = bjd_final[secondary_good_frames]
            file_paths_final = [file_paths_final[i] for i in range(len(file_paths_final)) if secondary_good_frames[i]]
            outlier_flags_final = outlier_flags_final[secondary_good_frames]
            flux_norm = flux_norm[secondary_good_frames, :]
            flux_norm_err = flux_norm_err[secondary_good_frames, :]

        final_std = np.std(diff_flux)
        logger.info("Aperture %s differential photometry: σ=%.6f, %d final points",
                    aper_name, final_std, len(diff_flux))

    # Convert good_star_mask to 2D for table storage
    good_star_mask_2d = np.tile(good_star_mask, (len(bjd_final), 1))

    # Return processed data
    result = {
        'BJD': bjd_final,
        'File': file_paths_final,
        'outlier_flag': outlier_flags_final,
        'flux': flux_final,
        'flux_err': flux_err_final,
        'flux_norm': flux_norm,
        'flux_norm_err': flux_norm_err,
        'sky': sky_final,
        'sky_err': sky_err_final,
        'good_star_mask': good_star_mask_2d
    }

    if diff_flux is not None:
        result['diff_flux'] = diff_flux
        result['diff_flux_err'] = diff_flux_err

    return result


def detect_and_clean_outliers(flux_array, flux_err_array, median_filter_window):
    """
    Detect outliers and clean comparison stars.

    Returns:
    --------
    tuple : (outlier_flags, frames_to_remove, flux_cleaned, flux_err_cleaned)
    """
    n_images, n_stars = flux_array.shape

    # Initialize outputs
    outlier_flags = np.zeros(n_images, dtype=int)
    frames_to_remove = np.zeros(n_images, dtype=bool)
    flux_cleaned = flux_array.copy()
    flux_err_cleaned = flux_err_array.copy()

    # Normalize flux for outlier detection
    flux_norm_data = np.zeros_like(flux_array)
    for star_num in range(n_stars):
        star_flux = flux_array[:, star_num]
        median_flux = np.median(star_flux)
        flux_norm_data[:, star_num] = star_flux / median_flux
        logger.debug("Star %d median flux: %.2f", star_num, median_flux)

    # Rolling window outlier detection
    outlier_mask = np.zeros((n_images, n_stars), dtype=bool)
    for star_num in range(n_stars):
        star_flux = flux_norm_data[:, star_num]
        outlier_mask[:, star_num] = rolling_window_outlier_detection(
            star_flux, median_filter_window, sigma_threshold=4.0)

        # Update outlier flags
        outlier_indices = np.where(outlier_mask[:, star_num])[0]
        n_outliers = len(outlier_indices)
        logger.debug("Star %d: detected %d outliers", star_num, n_outliers)

        for idx in outlier_indices:
            outlier_flags[idx] |= (1 << star_num)  # Set bit for this star

    # Determine frames to remove
    frames_removed_target = 0
    frames_removed_comparison = 0

    for frame_idx in range(n_images):
        # Check if target has outlier
        target_outlier = outlier_mask[frame_idx, 0]

        # Check comparison stars (stars 1 to n_stars-1)
        if n_stars > 1:
            comparison_outliers = outlier_mask[frame_idx, 1:]
            all_comparisons_outliers = np.all(comparison_outliers)
        else:
            all_comparisons_outliers = False

        # Remove frame if target outlier OR all comparisons are outliers
        if target_outlier or all_comparisons_outliers:
            frames_to_remove[frame_idx] = True
            if target_outlier:
                logger.debug("Removing frame %d: target star outlier", frame_idx)
                frames_removed_target += 1
            if all_comparisons_outliers:
                logger.debug("Removing frame %d: all comparison stars are outliers", frame_idx)
                frames_removed_comparison += 1

    logger.info("Outlier removal: %d target, %d comparison star frames",
                frames_removed_target, frames_removed_comparison)

    # Apply median filter interpolation to comparison star outliers (before removing frames)
    if n_stars > 1:  # Only if we have comparison stars
        total_interpolated = 0
        for star_num in range(1, n_stars):  # Only comparison stars
            star_outliers = outlier_mask[:, star_num] & ~frames_to_remove  # Only for kept frames
            if np.any(star_outliers):
                n_outliers = np.sum(star_outliers)
                logger.debug("Interpolating %d outliers for comparison star %d", n_outliers, star_num)
                total_interpolated += n_outliers

                flux_cleaned[:, star_num] = median_filter_interpolate(
                    flux_array[:, star_num], star_outliers, median_filter_window)

        if total_interpolated > 0:
            logger.info("Interpolated %d comparison star outliers", total_interpolated)

    return outlier_flags, frames_to_remove, flux_cleaned, flux_err_cleaned


def calculate_normalized_flux(flux_final, flux_err_final):
    """Calculate normalized flux arrays."""
    n_images, n_stars = flux_final.shape
    flux_norm = np.zeros_like(flux_final)
    flux_norm_err = np.zeros_like(flux_err_final)

    for star_num in range(n_stars):
        star_flux = flux_final[:, star_num]
        star_flux_err = flux_err_final[:, star_num]
        median_flux = np.median(star_flux)

        flux_norm[:, star_num] = star_flux / median_flux
        flux_norm_err[:, star_num] = star_flux_err / median_flux

    return flux_norm, flux_norm_err


def apply_secondary_differential_cleaning(diff_flux, diff_flux_err, median_filter_window):
    """Apply secondary outlier cleaning to differential flux."""
    # Secondary rolling window sigma clipping on differential flux
    diff_outlier_mask = rolling_window_outlier_detection(
        diff_flux, median_filter_window, sigma_threshold=4.0)

    good_frames = ~diff_outlier_mask

    # Clean differential flux
    diff_flux_clean = diff_flux[good_frames]
    diff_flux_err_clean = diff_flux_err[good_frames]

    return diff_flux_clean, diff_flux_err_clean, good_frames


def optimize_comparison_stars(target_flux, target_flux_err, comparison_flux, comparison_flux_err,
                              median_filter_window):
    """
    Find the optimal combination of comparison stars that minimizes scatter.

    Parameters:
    -----------
    target_flux : np.ndarray
        Target star flux
    target_flux_err : np.ndarray
        Target star flux errors
    comparison_flux : np.ndarray
        Comparison star fluxes (shape: n_images x n_comparison_stars)
    comparison_flux_err : np.ndarray
        Comparison star flux errors (shape: n_images x n_comparison_stars)
    median_filter_window : int
        Window size for outlier detection

    Returns:
    --------
    tuple : (best_combination, diff_flux, diff_flux_err, good_star_mask)
    """
    from itertools import combinations

    n_comparison_stars = comparison_flux.shape[1]
    comparison_star_indices = list(range(n_comparison_stars))
    all_combinations = []

    # Create all non-empty subsets (limit to reasonable sizes for efficiency)
    max_combo_size = min(5, n_comparison_stars)  # Limit to max 5 stars
    for r in range(1, max_combo_size + 1):
        for combo in combinations(comparison_star_indices, r):
            all_combinations.append(list(combo))

    logger.debug("Testing %d comparison star combinations", len(all_combinations))

    best_combination = None
    best_std = float('inf')
    best_n_points = 0

    # Test each combination
    for combo in all_combinations:
        # Create reference flux from this combination
        ref_flux = np.sum(comparison_flux[:, combo], axis=1)
        ref_flux_err = np.sqrt(np.sum(comparison_flux_err[:, combo] ** 2, axis=1))

        # Calculate differential flux
        diff_flux_raw = target_flux / ref_flux

        # Remove outliers using rolling window
        outlier_mask = rolling_window_outlier_detection(
            diff_flux_raw, median_filter_window, sigma_threshold=4.0)

        # Calculate std of cleaned data
        clean_diff_flux = diff_flux_raw[~outlier_mask]
        n_clean_points = len(clean_diff_flux)

        if n_clean_points > max(10, len(diff_flux_raw) * 0.5):  # Ensure enough data points
            std_dev = np.std(clean_diff_flux)

            if std_dev < best_std:
                best_std = std_dev
                best_combination = combo
                best_n_points = n_clean_points

    # Use the best combination
    if best_combination is not None:
        good_comparisons = best_combination
        logger.info("Best comparison star combination: indices %s (σ=%.6f, n=%d)",
                    good_comparisons, best_std, best_n_points)
    else:
        # Fallback: use all comparison stars
        good_comparisons = comparison_star_indices
        logger.warning("No good combination found, using all comparison stars")

    # Create good_star_mask (target + best comparisons)
    n_total_stars = n_comparison_stars + 1  # +1 for target
    good_star_mask = np.zeros(n_total_stars, dtype=bool)
    good_star_mask[0] = True  # Target always good
    good_star_mask[np.array(good_comparisons) + 1] = True  # Best comparisons (+1 for target offset)

    # Calculate final reference flux using best combination
    ref_flux = np.sum(comparison_flux[:, good_comparisons], axis=1)
    ref_flux_err = np.sqrt(np.sum(comparison_flux_err[:, good_comparisons] ** 2, axis=1))

    # Differential flux = target / reference
    diff_flux_raw = target_flux / ref_flux

    # Calculate initial error
    diff_flux_err_raw = diff_flux_raw * np.sqrt(
        (target_flux_err / target_flux) ** 2 + (ref_flux_err / ref_flux) ** 2)

    # Normalize differential flux to median = 1
    median_diff = np.median(diff_flux_raw)
    diff_flux = diff_flux_raw / median_diff
    diff_flux_err = diff_flux_err_raw / median_diff

    return good_comparisons, diff_flux, diff_flux_err, good_star_mask


def calculate_precision_metrics(aperture_tables, config, outdir, target):
    """
    Calculate theoretical photometric precision for all apertures and stars.

    Parameters:
    -----------
    aperture_tables : dict
        Dictionary of {aper_name: Table} containing photometry results
    config : dict
        Configuration dictionary containing instrument settings
    outdir : Path
        Output directory
    target : str
        Target name

    Returns:
    --------
    dict : Precision data for each aperture
    """
    logger.info("Calculating theoretical photometric precision")

    # Extract instrument parameters
    gain = config['instrument_config']['gain']
    telescope_diameter = config['instrument_config']['telescope_diameter']
    observatory_altitude = config['instrument_config']['observatory_altitude']
    scint_params = config['instrument_config']['scintillation']

    # Read calibration files
    calib_dir = outdir / 'calib'
    readout_noise = 0.0
    dark_current = 0.0

    try:
        ron_file = calib_dir / "readoutnoise.txt"
        with open(ron_file, 'r') as f:
            readout_noise = float(f.read().strip())
        logger.info("Read readout noise: %.3f electrons", readout_noise)
    except Exception as e:
        logger.warning("Could not read readout noise file: %s", e)

    try:
        dark_file = calib_dir / "darkcurrent.txt"
        with open(dark_file, 'r') as f:
            dark_current = float(f.read().strip())
        logger.info("Read dark current: %.3f electrons/sec/pixel", dark_current)
    except Exception as e:
        logger.warning("Could not read dark current file: %s", e)

    precision_data = {}

    for aper_name, table in aperture_tables.items():
        if len(table) == 0:
            continue

        # Extract aperture radius from name (e.g., "aper5" -> 5)
        aper_radius = float(aper_name.replace('aper', ''))

        logger.debug("Processing precision for aperture %s (radius=%.1f)", aper_name, aper_radius)

        # Get data arrays
        flux_final = np.array(table['flux'])  # Shape: (n_images, n_stars)
        sky_final = np.array(table['sky'])
        file_paths_final = list(table['File'])
        n_images, n_stars = flux_final.shape

        # Read headers to get exposure time and altitude
        exp_times = []
        altitudes = []

        for file_path in file_paths_final:
            try:
                # Handle Docker path translation if needed
                from photometry import translate_path_for_docker
                translated_path = translate_path_for_docker(file_path, outdir)
                with fits.open(translated_path) as hdul:
                    exp_times.append(hdul[0].header['EXPTIME'])
                    altitudes.append(hdul[0].header['ALTITUDE'])
            except Exception as e:
                logger.warning("Could not read headers from %s: %s", file_path, e)

        if not exp_times or not altitudes:
            logger.warning("Could not calculate precision for aperture %s: no valid headers", aper_name)
            precision_data[aper_name] = {
                'aperture_radius': aper_radius,
                'sigma_scint': 0.0,
                'star_precisions': [0.0] * n_stars,
                'star_sky_backgrounds': [0.0] * n_stars,
                'n_frames': n_images
            }
            continue

        # Calculate mean values
        mean_exp_time = np.mean(exp_times)
        mean_altitude = np.mean(altitudes)
        mean_zenith_angle = 90.0 - mean_altitude

        # Calculate scintillation using Osborn et al. formula
        C_Y = scint_params['C_Y']
        H = scint_params['H']

        sec_zenith = 1.0 / np.cos(np.radians(mean_zenith_angle))
        altitude_factor = np.exp(-2 * observatory_altitude / H)

        sigma_scint = (1e-5 * C_Y ** 2 * telescope_diameter ** (4 / 3) *
                       (sec_zenith ** 3) * altitude_factor / mean_exp_time)

        logger.debug("Precision calculation parameters:")
        logger.debug("  Mean exposure time: %.2f s", mean_exp_time)
        logger.debug("  Mean zenith angle: %.2f deg", mean_zenith_angle)
        logger.debug("  Scintillation noise: %.6f", sigma_scint)

        # Calculate precision for each star
        n_pixels = np.pi * aper_radius ** 2
        star_precisions = []
        star_sky_backgrounds = []

        for star_num in range(n_stars):
            # Use mean flux and sky for this star
            mu_star = np.mean(flux_final[:, star_num])
            mu_sky = np.mean(sky_final[:, star_num])

            # Calculate theoretical photometric precision
            sigma_scint_variance = sigma_scint * mu_star ** 2
            noise_term = mu_star + sigma_scint_variance + n_pixels * (
                    mu_sky + dark_current * mean_exp_time + readout_noise ** 2)

            if mu_star > 0:
                precision = np.sqrt(noise_term) / mu_star
            else:
                precision = np.inf

            star_precisions.append(precision)
            star_sky_backgrounds.append(mu_sky)

            logger.debug("Star %d: mu_star=%.1f, mu_sky=%.1f, precision=%.6f",
                         star_num, mu_star, mu_sky, precision)

        # Store precision data
        precision_data[aper_name] = {
            'aperture_radius': aper_radius,
            'sigma_scint': sigma_scint,
            'star_precisions': star_precisions,
            'star_sky_backgrounds': star_sky_backgrounds,
            'n_frames': n_images
        }

    return precision_data


def calculate_observed_precision(aperture_tables, median_filter_window=21):
    """
    Calculate observed precision using rolling window statistics.

    Parameters:
    -----------
    aperture_tables : dict
        Dictionary of {aper_name: Table} containing photometry results
    median_filter_window : int
        Window size for precision calculation

    Returns:
    --------
    dict : Updated precision data with observed precisions
    """
    logger.info("Calculating observed precision")

    observed_precisions = {}

    for aper_name, table in aperture_tables.items():
        if len(table) == 0:
            continue

        bjd_final = np.array(table['BJD'])
        flux_norm = np.array(table['flux_norm'])  # Shape: (n_images, n_stars)
        n_images, n_stars = flux_norm.shape

        # Calculate observation duration and window size
        if len(bjd_final) > 0:
            obs_duration_days = np.max(bjd_final) - np.min(bjd_final)
            obs_duration_minutes = obs_duration_days * 24 * 60

            if obs_duration_minutes > 0:
                # Window size = observation duration / 10 minutes, scaled by number of data points
                window_size_points = int(len(bjd_final) / (obs_duration_minutes / 10.0))
                window_size_points = max(5, window_size_points)  # Minimum window size of 5 points

                logger.debug("Observed precision calculation: obs_duration=%.2f min, window_size=%d points",
                             obs_duration_minutes, window_size_points)

                star_obs_precisions = []

                for star_num in range(n_stars):
                    star_flux_norm = flux_norm[:, star_num]

                    if len(star_flux_norm) >= window_size_points:
                        # Calculate rolling standard deviations
                        rolling_stds = []
                        for i in range(len(star_flux_norm) - window_size_points + 1):
                            window_data = star_flux_norm[i:i + window_size_points]
                            rolling_stds.append(np.std(window_data))

                        # Take median of rolling standard deviations
                        if len(rolling_stds) > 0:
                            obs_precision = np.median(rolling_stds)
                        else:
                            obs_precision = np.nan
                    else:
                        # Not enough data points for rolling window
                        obs_precision = np.nan

                    star_obs_precisions.append(obs_precision)
                    logger.debug("Star %d observed precision: %.6f", star_num, obs_precision)

                observed_precisions[aper_name] = star_obs_precisions
            else:
                logger.warning("Zero observation duration for aperture %s", aper_name)
                observed_precisions[aper_name] = [np.nan] * n_stars
        else:
            logger.warning("No data points for aperture %s", aper_name)
            observed_precisions[aper_name] = [np.nan] * n_stars

    return observed_precisions


def create_summary_files(aperture_tables, precision_data, observed_precisions, outdir, target):
    """
    Create differential_summary.txt and precision_metrics.fits files.

    Parameters:
    -----------
    aperture_tables : dict
        Dictionary of {aper_name: Table} containing photometry results
    precision_data : dict
        Precision data from calculate_precision_metrics()
    observed_precisions : dict
        Observed precision data from calculate_observed_precision()
    outdir : Path
        Output directory
    target : str
        Target name
    """
    logger.info("Creating comprehensive photometry summary")

    photometry_dir = outdir / target / "photometry"

    # Create differential summary text file
    summary_lines = ["# Main Summary"]
    summary_lines.append("# Aperture_radius  Diff_flux_std_dev  N_frames_used  sigma_scint_value")

    aperture_stds = []

    for aper_name, table in aperture_tables.items():
        if aper_name in precision_data:
            precision_info = precision_data[aper_name]
            aper_radius = int(precision_info['aperture_radius'])
            sigma_scint = precision_info['sigma_scint']
            n_frames = precision_info['n_frames']

            # Get differential flux std if available
            if 'diff_flux' in table.colnames and len(table) > 0:
                diff_flux_data = np.array(table['diff_flux'])
                std_dev = np.std(diff_flux_data)
                aperture_stds.append((aper_radius, std_dev))
            else:
                std_dev = 0.0

            summary_lines.append(f"{aper_radius:15d}  {std_dev:15.6f}  {n_frames:12d}  {sigma_scint:15.6e}")

    # Add star-specific sections
    if precision_data:
        # Get number of stars from first aperture
        first_aper = list(precision_data.keys())[0]
        n_stars = len(precision_data[first_aper]['star_precisions'])

        for star_num in range(n_stars):
            summary_lines.append(f"\n# Star_{star_num}_data")
            summary_lines.append("# Aperture_radius  sigma_prec  sky_background  sigma_obs_10min")

            for aper_name in precision_data.keys():
                precision_info = precision_data[aper_name]
                aper_radius = int(precision_info['aperture_radius'])

                if star_num < len(precision_info['star_precisions']):
                    sigma_prec = precision_info['star_precisions'][star_num]
                    sky_bg = precision_info['star_sky_backgrounds'][star_num]

                    # Get observed precision if available
                    if (aper_name in observed_precisions and
                            star_num < len(observed_precisions[aper_name])):
                        sigma_obs = observed_precisions[aper_name][star_num]
                        if np.isnan(sigma_obs):
                            sigma_obs_str = "        nan"
                        else:
                            sigma_obs_str = f"{sigma_obs:11.6f}"
                    else:
                        sigma_obs_str = "        nan"

                    summary_lines.append(f"{aper_radius:15d}  {sigma_prec:10.6f}  {sky_bg:13.2f}  {sigma_obs_str}")

    # Write summary file
    try:
        summary_path = photometry_dir / "differential_summary.txt"
        with open(summary_path, 'w') as f:
            for line in summary_lines:
                f.write(line + '\n')

        logger.info("Comprehensive photometry summary saved to %s", summary_path)
    except Exception as e:
        logger.error("Failed to write summary file: %s", e)

    # Create detailed FITS table with precision metrics
    try:
        logger.info("Creating detailed precision metrics FITS table")

        # Check for existing precision metrics file
        precision_table_path = photometry_dir / "precision_metrics.fits"
        existing_data = None

        if precision_table_path.exists():
            try:
                existing_data = Table.read(precision_table_path, format='fits')
                logger.info("Found existing precision metrics with %d rows", len(existing_data))
            except Exception as e:
                logger.warning("Could not read existing precision metrics: %s", e)

        # Get aperture radii being processed in current run
        current_aperture_radii = set()
        for aper_name, precision_info in precision_data.items():
            current_aperture_radii.add(precision_info['aperture_radius'])

        # Prepare data for FITS table
        aperture_radii = []
        star_numbers = []
        sigma_precs = []
        sky_backgrounds = []
        sigma_scints = []
        sigma_obs_10mins = []
        diff_flux_stds = []
        n_frames_used = []

        for aper_name, precision_info in precision_data.items():
            aper_radius = precision_info['aperture_radius']
            sigma_scint = precision_info['sigma_scint']
            n_frames = precision_info['n_frames']

            # Get differential flux std for this aperture
            diff_std = 0.0
            if aper_name in aperture_tables and 'diff_flux' in aperture_tables[aper_name].colnames:
                diff_flux_data = np.array(aperture_tables[aper_name]['diff_flux'])
                if len(diff_flux_data) > 0:
                    diff_std = np.std(diff_flux_data)

            # Add row for each star
            for star_num in range(len(precision_info['star_precisions'])):
                aperture_radii.append(aper_radius)
                star_numbers.append(star_num)
                sigma_precs.append(precision_info['star_precisions'][star_num])
                sky_backgrounds.append(precision_info['star_sky_backgrounds'][star_num])
                sigma_scints.append(sigma_scint)
                diff_flux_stds.append(diff_std)
                n_frames_used.append(n_frames)

                # Add observed precision
                if (aper_name in observed_precisions and
                        star_num < len(observed_precisions[aper_name])):
                    sigma_obs_10mins.append(observed_precisions[aper_name][star_num])
                else:
                    sigma_obs_10mins.append(np.nan)

        # Create FITS table
        precision_table = Table({
            'aperture_radius': aperture_radii,
            'star_number': star_numbers,
            'sigma_precision': sigma_precs,
            'sky_background': sky_backgrounds,
            'sigma_scintillation': sigma_scints,
            'sigma_obs_10min': sigma_obs_10mins,
            'diff_flux_std': diff_flux_stds,
            'n_frames_used': n_frames_used
        })

        # Add units and descriptions
        precision_table['aperture_radius'].unit = 'pixel'
        precision_table['sigma_precision'].unit = 'relative'
        precision_table['sky_background'].unit = 'photon/pixel'
        precision_table['sigma_scintillation'].unit = 'relative'
        precision_table['sigma_obs_10min'].unit = 'relative'
        precision_table['diff_flux_std'].unit = 'relative'

        precision_table['aperture_radius'].description = 'Aperture radius in pixels'
        precision_table['star_number'].description = 'Star index (0=target)'
        precision_table['sigma_precision'].description = 'Theoretical photometric precision'
        precision_table['sky_background'].description = 'Mean sky background per pixel'
        precision_table['sigma_scintillation'].description = 'Atmospheric scintillation noise'
        precision_table['sigma_obs_10min'].description = 'Observed precision (10-min rolling window)'
        precision_table['diff_flux_std'].description = 'Standard deviation of differential flux'
        precision_table['n_frames_used'].description = 'Number of frames after cleaning'

        # Merge with existing data if available
        if existing_data is not None:
            # Remove rows from existing data that match current aperture radii
            mask = ~np.isin(existing_data['aperture_radius'], list(current_aperture_radii))
            filtered_existing = existing_data[mask]

            logger.info("Keeping %d existing rows, adding %d new rows",
                        len(filtered_existing), len(precision_table))

            # Combine tables
            from astropy.table import vstack
            final_table = vstack([filtered_existing, precision_table])

            # Sort by aperture radius and star number for consistency
            final_table.sort(['aperture_radius', 'star_number'])
        else:
            final_table = precision_table

        # Save FITS table
        final_table.write(precision_table_path, format='fits', overwrite=True)

        logger.info("Precision metrics FITS table saved to %s with %d total rows",
                    precision_table_path, len(final_table))

    except Exception as e:
        logger.error("Failed to create precision metrics FITS table: %s", e)

    # Print best aperture
    if aperture_stds:
        best_aper, best_std = min(aperture_stds, key=lambda x: x[1])
        logger.info("Best aperture: %d pixels (σ = %.6f)", best_aper, best_std)


def run_complete_analysis(aperture_tables, config, outdir, target, median_filter_window=21):
    """
    Run the complete photometry analysis pipeline.

    Parameters:
    -----------
    aperture_tables : dict
        Dictionary of {aper_name: Table} containing photometry results
    config : dict
        Configuration dictionary
    outdir : Path
        Output directory
    target : str
        Target name
    median_filter_window : int
        Window size for precision calculations
    """
    logger.info("Running complete photometry analysis for target %s", target)

    # Calculate theoretical precision metrics
    precision_data = calculate_precision_metrics(aperture_tables, config, outdir, target)

    # Calculate observed precision
    observed_precisions = calculate_observed_precision(aperture_tables, median_filter_window)

    # Create summary files
    create_summary_files(aperture_tables, precision_data, observed_precisions, outdir, target)

    logger.info("Complete photometry analysis finished for target %s", target)