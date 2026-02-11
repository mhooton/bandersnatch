import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from aper import aper
from photometry_analysis import (run_complete_analysis, process_aperture_photometry)

# Set up logger for this module
logger = logging.getLogger(__name__)


def translate_path_for_docker(file_path, outdir):
    """Translate host paths to Docker container paths when running in Docker."""
    import os
    from pathlib import Path

    # Check if running in Docker
    is_docker = os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'

    if not is_docker:
        return file_path  # No translation needed

    # Convert to Path object
    path = Path(file_path)

    # Extract the relevant parts after bandersnatch_runs
    # Expected structure: .../bandersnatch_runs/RUN_NAME/TARGET/RUN/filename.fits
    parts = path.parts

    try:
        # Find bandersnatch_runs in the path
        br_idx = None
        for i, part in enumerate(parts):
            if part == 'bandersnatch_runs':
                br_idx = i
                break

        if br_idx is not None and br_idx + 3 < len(parts):
            # Parts after bandersnatch_runs: [RUN_NAME, TARGET, RUN, filename]
            target = parts[br_idx + 2]  # TARGET
            run = parts[br_idx + 3]  # RUN
            filename = parts[-1]  # filename

            # Construct Docker path: /app/output/RUN_NAME/TARGET/RUN/filename
            docker_path = outdir / target / run / filename
            return str(docker_path)
    except:
        pass

    # Fallback: try to extract just target/run/filename from the end
    if len(parts) >= 3:
        target = parts[-3]
        run = parts[-2]
        filename = parts[-1]
        docker_path = outdir / target / run / filename
        return str(docker_path)

    # Final fallback: return original path
    return file_path

def photometry(outdir, run, target, config, aper_min, aper_max, SKYRAD_inner, SKYRAD_outer, sky_suppress,
               median_filter_window=21, time_bin_size=0.005):
    """
    Perform aperture photometry on all processed images for a given target.

    Parameters:
    -----------
    outdir : Path
        Output directory containing the data
    run : str
        Run identifier
    target : str
        Target name
    config : dict
        Configuration dictionary containing instrument settings
    """
    logger.info("Starting aperture photometry for target %s", target)

    # Extract instrument parameters
    gain = config['instrument_config']['gain']
    phpadu = config['instrument_config']['phpadu']
    telescope_diameter = config['instrument_config']['telescope_diameter']
    observatory_altitude = config['instrument_config']['observatory_altitude']
    scint_params = config['instrument_config']['scintillation']

    logger.debug("Parameters: gain=%.3f, apertures=%d-%d, sky_annulus=[%.1f,%.1f], sky_suppress=%s",
                 gain, aper_min, aper_max, SKYRAD_inner, SKYRAD_outer, sky_suppress)

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

    # 1. Read centroids from previous step
    centroiding_dir = outdir / target / "centroiding"
    centroids_file = centroiding_dir / "centroids.fits"

    logger.debug("Reading centroids from %s", centroids_file)

    try:
        centroids_table = Table.read(centroids_file)
        logger.info("Successfully loaded centroids table")
    except Exception as e:
        logger.error("Failed to read centroids file %s: %s", centroids_file, e)
        raise

    # Extract data
    bjd = np.array(centroids_table['BJD'])
    file_paths = list(centroids_table['File'])
    xc_data = np.array(centroids_table['xc'])  # Shape: (n_images, n_stars)
    yc_data = np.array(centroids_table['yc'])  # Shape: (n_images, n_stars)

    n_images, n_stars = xc_data.shape
    logger.info("Processing %d images with %d stars", n_images, n_stars)

    # 2. Set up aper parameters
    apr = np.arange(aper_min, aper_max + 1, dtype=float)  # [5, 6, 7, ..., 12]
    n_apertures = len(apr)
    skyrad = np.array([SKYRAD_inner, SKYRAD_outer])  # [15, 25]
    setskyval = 1e-20 if sky_suppress else None

    logger.info("Aperture photometry setup:")
    logger.info("  Apertures: %s", apr)
    logger.info("  Sky annulus: %s", skyrad)
    logger.info("  Sky suppression: %s", sky_suppress)
    logger.info("  Median filter window: %d", median_filter_window)

    # 3. Create photometry directory
    photometry_dir = outdir / target / "photometry"
    os.makedirs(photometry_dir, exist_ok=True)
    plots_dir = photometry_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    logger.debug("Created photometry directories: %s", photometry_dir)

    # 4. Initialize result lists
    flux_results = []  # Will be (n_images, n_apertures, n_stars)
    flux_err_results = []
    sky_results = []  # Will be (n_images, n_stars)
    sky_err_results = []

    logger.info("Processing %d images for aperture photometry", n_images)

    # 5. Loop through images
    successful_images = 0
    for i in range(n_images):
        logger.debug("Processing image %d/%d: %s", i + 1, n_images, file_paths[i])

        try:
            # Translate path for Docker if needed
            translated_path = translate_path_for_docker(file_paths[i], outdir)
            with fits.open(translated_path) as hdul:
                image = hdul[0].data.astype(np.float64)

            # Extract centroids for this image
            xc_frame = xc_data[i, :]  # Shape: (n_stars,)
            yc_frame = yc_data[i, :]  # Shape: (n_stars,)

            logger.debug("Image %d centroids: x=%s, y=%s", i, xc_frame, yc_frame)

            # Call aper function
            mags, errap, sky, skyerr = aper(
                image=image,
                xc=xc_frame,
                yc=yc_frame,
                phpadu=phpadu,  # Use gain instead of PHPADU
                apr=apr,
                skyrad=skyrad,
                setskyval=setskyval,
                flux=True,  # Return flux instead of magnitudes
                silent=True
            )

            # Convert all outputs from ADU to electrons
            mags_electrons = mags * gain
            errap_electrons = errap * gain
            sky_electrons = sky * gain
            skyerr_electrons = skyerr * gain

            # Append results to lists
            # mags shape: (n_apertures, n_stars)
            # errap shape: (n_apertures, n_stars)
            # sky shape: (n_stars,)
            # skyerr shape: (n_stars,)
            flux_results.append(mags_electrons)
            flux_err_results.append(errap_electrons)
            sky_results.append(sky_electrons)
            sky_err_results.append(skyerr_electrons)

            successful_images += 1

            logger.debug("Image %d photometry completed: flux_shape=%s, sky_shape=%s",
                         i, mags.shape, sky.shape)

        except Exception as e:
            logger.error("Failed to process image %d (%s): %s", i, file_paths[i], e)
            # Continue with remaining images
            continue

    logger.info("Successfully processed %d/%d images for photometry", successful_images, n_images)

    if successful_images == 0:
        logger.error("No images were successfully processed for photometry")
        raise RuntimeError("No images were successfully processed for photometry")

    # 6. Convert to numpy arrays
    flux_array = np.array(flux_results)  # Shape: (n_images, n_apertures, n_stars)
    flux_err_array = np.array(flux_err_results)
    sky_array = np.array(sky_results)  # Shape: (n_images, n_stars)
    sky_err_array = np.array(sky_err_results)

    logger.debug("Created flux arrays: flux_shape=%s, sky_shape=%s",
                 flux_array.shape, sky_array.shape)

    # 7. Process each aperture separately
    logger.info("Processing %d apertures with outlier detection", n_apertures)

    aperture_tables = {}  # Store tables for each aperture
    precision_data = {}  # Store precision calculations for summary

    for aper_idx, aper_radius in enumerate(apr):
        aper_name = f"aper{int(aper_radius)}"
        logger.info("Processing aperture %s (%d/%d)", aper_name, aper_idx + 1, n_apertures)

        # Extract data for this aperture
        flux_this_aper = flux_array[:, aper_idx, :]  # Shape: (n_images, n_stars)
        flux_err_this_aper = flux_err_array[:, aper_idx, :]

        # Use shared processing function
        try:
            aperture_data = process_aperture_photometry(
                flux_this_aper, flux_err_this_aper, sky_array, sky_err_array,
                bjd, file_paths, aper_radius, median_filter_window)
        except Exception as e:
            logger.error("Failed to process aperture %s: %s", aper_name, e)
            continue

        # Create and save table for this aperture
        try:
            aperture_table = Table(aperture_data)
            output_path = photometry_dir / f"photometry_{aper_name}.fits"
            aperture_table.write(output_path, format='fits', overwrite=True)

            logger.info("Aperture %s results saved to %s", aper_name, output_path)

            # Store table for plotting
            aperture_tables[aper_name] = aperture_table

        except Exception as e:
            logger.error("Failed to save aperture %s table: %s", aper_name, e)
            continue

    logger.info("Running complete photometry analysis")
    run_complete_analysis(aperture_tables, config, outdir, target, median_filter_window)

    logger.info("Photometry processing complete: %d aperture tables created", len(aperture_tables))

    # 8. Create diagnostic plots
    try:
        create_photometry_plots(outdir, target, aperture_tables, median_filter_window, time_bin_size)
    except Exception as e:
        logger.error("Failed to create photometry plots: %s", e)
        # Continue execution even if plots fail

    logger.info("Aperture photometry completed successfully for target %s", target)


def create_photometry_plots(outdir, target, aperture_tables, median_filter_window, time_bin_size=0.005):
    """
    Create diagnostic plots for photometry results.

    Parameters:
    -----------
    outdir : Path
        Output directory
    target : str
        Target name
    aperture_tables : dict
        Dictionary of aperture tables {aper_name: Table}
    time_bin_size : float, optional
        Time bin size in days for binning data points
    """
    logger.info("Creating photometry diagnostic plots")

    # Create plots directory
    photometry_dir = outdir / target / "photometry"
    plots_dir = photometry_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Define color palette for up to 20 stars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5']

    plots_created = 0

    # Process each aperture separately
    for aper_name, table in aperture_tables.items():
        logger.debug("Creating plots for aperture %s", aper_name)

        bjd = table['BJD']
        n_stars = len(table['flux'][0]) if len(table) > 0 else 0

        # Add after extracting bjd and n_stars:
        good_star_mask = table['good_star_mask'][0]  # Same for all rows
        good_stars = np.where(good_star_mask)[0]
        good_comparisons = [s for s in good_stars if s > 0]  # Exclude target (star 0)

        if len(table) == 0:
            logger.warning("No data for %s, skipping plots", aper_name)
            continue

        try:
            # Plot normalized flux vs BJD for each star (individual plots)
            flux_norm_data = np.array(table['flux_norm'])  # Shape: (n_images, n_stars)

            for star_num in range(n_stars):
                color = colors[star_num % len(colors)]

                plt.figure(figsize=(10, 6))
                plt.plot(bjd, flux_norm_data[:, star_num], 'o', markersize=3)

                # Add binned data if requested
                if time_bin_size is not None:
                    bjd_binned, flux_binned = bin_data(bjd, flux_norm_data[:, star_num], time_bin_size)
                    if len(bjd_binned) > 0:
                        plt.plot(bjd_binned, flux_binned, 'o', markersize=6,
                                 color=color, markeredgecolor='black')

                plt.xlabel('BJD')
                plt.ylabel('Normalized Flux')
                plt.title(f'Normalized Flux vs BJD - {aper_name} - Star {star_num} - Target {target}')
                plt.grid(True, alpha=0.3)
                if time_bin_size is not None:
                    plt.legend()

                # Calculate rolling standard deviation
                from scipy import ndimage

                flux_star = flux_norm_data[:, star_num]
                if len(flux_star) >= median_filter_window:
                    # Calculate rolling standard deviation using median_filter_window
                    rolling_stds = []
                    for i in range(len(flux_star) - median_filter_window + 1):
                        window_data = flux_star[i:i + median_filter_window]
                        rolling_stds.append(np.std(window_data))
                    std_dev = np.mean(rolling_stds)
                else:
                    std_dev = np.std(flux_star)  # Fallback for short datasets

                plt.text(0.02, 0.98, f'local σ = {std_dev:.4f}', transform=plt.gca().transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Save plot
                plot_path = plots_dir / f"flux_norm_{aper_name}_star_{star_num}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                plots_created += 1

            # Plot raw flux vs BJD (all stars on same plot)
            flux_data = np.array(table['flux'])  # Shape: (n_images, n_stars)

            plt.figure(figsize=(10, 6))
            for star_num in good_stars:  # Only good stars
                color = colors[star_num % len(colors)]
                plt.plot(bjd, flux_data[:, star_num], 'o', markersize=3, alpha=0.5,
                         color=color, label=f'Star {star_num}')

                # Add binned data if requested
                if time_bin_size is not None and time_bin_size > 0:
                    bjd_binned, flux_binned = bin_data(bjd, flux_data[:, star_num], time_bin_size)
                    if len(bjd_binned) > 0:
                        plt.plot(bjd_binned, flux_binned, 'o', markersize=6,
                                 color=color, markeredgecolor='black')

            plt.xlabel('BJD')
            plt.ylabel('Flux (counts)')
            plt.title(f'Flux vs BJD - {aper_name} - Target {target}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = plots_dir / f"flux_{aper_name}_all_stars.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots_created += 1

            # Plot differential flux if available
            if 'diff_flux' in table.colnames:
                diff_flux_data = np.array(table['diff_flux'])

                # Calculate standard deviation
                std_dev = np.std(diff_flux_data)

                plt.figure(figsize=(10, 6))
                plt.plot(bjd, diff_flux_data, 'o', markersize=3)

                # Add binned data if requested
                if time_bin_size is not None:
                    bjd_binned, diff_flux_binned = bin_data(bjd, diff_flux_data, time_bin_size)
                    if len(bjd_binned) > 0:
                        plt.plot(bjd_binned, diff_flux_binned, 'o', markersize=6,
                                 markeredgecolor='black')

                plt.xlabel('BJD')
                plt.ylabel('Differential Flux')
                plt.title(f'Differential Flux vs BJD - {aper_name} - Target {target}')
                plt.grid(True, alpha=0.3)

                # Add standard deviation text
                plt.text(0.02, 0.98, f'σ = {std_dev:.4f}', transform=plt.gca().transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Save plot
                plot_path = plots_dir / f"diff_flux_{aper_name}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                plots_created += 1

        except Exception as e:
            logger.error("Failed to create plots for aperture %s: %s", aper_name, e)
            continue

    # Plot sky vs BJD (all stars on same plot) - use first aperture table
    if aperture_tables:
        try:
            first_table = list(aperture_tables.values())[0]
            bjd = first_table['BJD']
            sky_data = np.array(first_table['sky'])  # Shape: (n_images, n_stars)
            n_stars = sky_data.shape[1] if len(sky_data.shape) > 1 else 1

            good_star_mask = first_table['good_star_mask'][0]  # Same for all rows
            good_stars = np.where(good_star_mask)[0]

            plt.figure(figsize=(10, 6))
            for star_num in good_stars:  # Only good stars
                color = colors[star_num % len(colors)]
                plt.plot(bjd, sky_data[:, star_num], 'o', markersize=3, alpha=0.5,
                         color=color, label=f'Star {star_num}')

                # Add binned data if requested
                if time_bin_size is not None and time_bin_size > 0:
                    bjd_binned, sky_binned = bin_data(bjd, sky_data[:, star_num], time_bin_size)
                    if len(bjd_binned) > 0:
                        plt.plot(bjd_binned, sky_binned, 'o', markersize=6,
                                 color=color, markeredgecolor='black')

            plt.xlabel('BJD')
            plt.ylabel('Sky (counts)')
            plt.title(f'Sky vs BJD - All Stars - Target {target}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = plots_dir / f"sky_all_stars.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots_created += 1

        except Exception as e:
            logger.error("Failed to create sky plot: %s", e)

    logger.info("Created %d photometry plots in %s", plots_created, plots_dir)


def bin_data(time, data, bin_size):
    """
    Bin data by time intervals.

    Parameters:
    -----------
    time : np.ndarray
        Time values (e.g., BJD)
    data : np.ndarray
        Data values to bin
    bin_size : float
        Size of time bins

    Returns:
    --------
    time_binned : np.ndarray
        Center times of bins
    data_binned : np.ndarray
        Mean values in each bin
    """
    if len(time) == 0 or bin_size <= 0:
        return np.array([]), np.array([])

    # Create bins
    time_min = np.min(time)
    time_max = np.max(time)
    bin_edges = np.arange(time_min, time_max + bin_size, bin_size)

    # Find which bin each point belongs to
    bin_indices = np.digitize(time, bin_edges) - 1

    time_binned = []
    data_binned = []

    for i in range(len(bin_edges) - 1):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:  # Only include bins with data
            time_binned.append(np.mean(time[mask]))
            data_binned.append(np.mean(data[mask]))

    return np.array(time_binned), np.array(data_binned)


def calculate_rolling_std(data, window_size):
    """Calculate rolling standard deviation."""
    if len(data) < window_size:
        return np.std(data)
    rolling_stds = []
    for i in range(len(data) - window_size + 1):
        rolling_stds.append(np.std(data[i:i + window_size]))
    return np.mean(rolling_stds)


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running photometry.py as standalone script")
    # You would call photometry() here with appropriate parameters