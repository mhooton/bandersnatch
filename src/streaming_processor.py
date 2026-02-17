# COMPLETE UPDATED FILE: streaming_processor.py

import numpy as np
import os
import logging
from pathlib import Path
from astropy.io import fits
from astropy.table import Table

# Import existing functions
from reduce_science import overscan_corr, apply_calibrations
from centroid import centroid_loop, extract_airmass, write_centroiding_results
from photometry import translate_path_for_docker, count_bad_pixels_in_apertures, write_photometry_results
from aper import aper

logger = logging.getLogger(__name__)


def process_images_streaming(outdir, run, target, config, calib_frames, initial_positions,
                             centroid_params, photometry_params, bad_pixel_map=None, save_processed_images=False):
    """
    Process all images for a target through reduction, centroiding, and photometry in a streaming fashion.

    Parameters:
    -----------
    outdir : Path
        Output directory
    run : str
        Run identifier
    target : str
        Target name
    config : dict
        Full configuration dictionary
    calib_frames : dict
        Dictionary containing master calibration frames
    initial_positions : list
        Initial star positions
    centroid_params : dict
        Centroiding parameters
    photometry_params : dict
        Photometry parameters
    save_processed_images : bool, optional
        Whether to save processed images to disk (default: False)
    """
    logger.info("Starting streaming processing for target %s", target)

    # Extract parameters
    inst_settings = config['instrument_settings']
    gain = config['instrument_config']['gain']
    phpadu = config['instrument_config']['phpadu']

    # Get image list
    list_file = outdir / "calib" / f"{run}_image_{target}.list"
    try:
        with open(list_file) as f:
            filenames = [line.strip() for line in f if line.strip()]
        logger.info("Found %d images for streaming processing", len(filenames))
    except Exception as e:
        logger.error("Failed to read image list %s: %s", list_file, e)
        raise

    if not filenames:
        raise ValueError(f"No images found for target {target}")

    # Initialize result containers
    # Centroiding results
    centroid_results = {
        'BJD': [],
        'File': [],
        'airmass': [],  # ADD THIS LINE
        'xc': [],
        'yc': [],
        'x_bright': [],
        'y_bright': [],
        'val_bright': [],
        'x_width': [],
        'y_width': [],
        'n_pixels_above': [],
        'sig3_pixels': [],
        'sig5_pixels': [],
        'sig10_pixels': []
    }

    # Photometry results - one dict per aperture
    photometry_results = {}

    # Poststamps - shape will be (n_images, 2*boxsize+1, 2*boxsize+1, n_stars)
    all_poststamps = []

    # For saving processed images
    processed_filenames = []

    # Extract star positions
    star_x_input = np.array(initial_positions)[:, 0]
    star_y_input = np.array(initial_positions)[:, 1]
    n_stars = len(star_x_input)

    # Set up photometry apertures
    aper_min = photometry_params['aper_min']
    aper_max = photometry_params['aper_max']
    apr = np.arange(aper_min, aper_max + 1, dtype=float)
    n_apertures = len(apr)
    skyrad = np.array([photometry_params['SKYRAD_inner'], photometry_params['SKYRAD_outer']])
    setskyval = 1e-20 if photometry_params['sky_suppress'] else None

    # Initialize photometry result containers for each aperture
    for aper_idx, aper_radius in enumerate(apr):
        aper_name = f"aper{int(aper_radius)}"
        photometry_results[aper_name] = {
            'BJD': [],
            'File': [],
            'flux': [],
            'flux_err': [],
            'sky': [],
            'sky_err': [],
            'n_bad_pixels_in_aperture': []
        }

    # Create output directories
    target_dir = outdir / target
    run_dir = target_dir / run
    if save_processed_images:
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        logger.info("Created directories for saving processed images")

    # Process each image
    successful_images = 0
    failed_images = []

    for i, filename in enumerate(filenames):
        if (i + 1) % 10 == 0:
            logger.info("Processing image %d/%d", i + 1, len(filenames))

        try:
            # Step 1: Load and reduce image
            logger.debug("Loading and reducing image: %s", filename)

            with fits.open(filename) as hdul:
                image = hdul[inst_settings.get('image_extension', 0)].data.astype(np.float64)
                header = hdul[0].header.copy()
                exp_time = header["EXPTIME"]

                # Extract BJD
                bjd_obs = header.get('BJD-OBS', float(i))

                # Extract AIRMASS - ADD THESE LINES
                airmass_obs = extract_airmass(header)

            # Apply calibrations
            if config.get('to_overscan_correct', False):
                image = overscan_corr(image)

            image = apply_calibrations(image, calib_frames['bias'], calib_frames['dark'],
                                       calib_frames['flat'], exp_time, bad_pixel_map)

            # Save processed image if requested
            if save_processed_images:
                processed_filename = f"proc{Path(filename).name}"
                processed_path = run_dir / processed_filename

                # Create new FITS file with processed data
                new_hdu = fits.PrimaryHDU(data=image, header=header)
                new_hdu.writeto(processed_path, overwrite=True)
                processed_filenames.append(processed_filename)
                logger.debug("Saved processed image: %s", processed_path)

            # Step 2: Perform centroiding
            logger.debug("Performing centroiding for image %d", i)

            centroid_result = centroid_loop(
                star_x_input, star_y_input,
                centroid_params['boxsize'],
                centroid_params['nlimit_centroid'],
                centroid_params['clip_centroid'],
                centroid_params['sky_sigma'],
                centroid_params['tracking_star'],
                centroid_params['flux_above_value'],
                image=image, header=header,
                mask_centroid_pixels=centroid_params['mask_centroid_pixels']
            )

            # Store centroiding results (use original filename for consistency)
            centroid_results['BJD'].append(bjd_obs)
            centroid_results['File'].append(str(filename))  # Store original path
            centroid_results['airmass'].append(airmass_obs)  # ADD THIS LINE
            centroid_results['xc'].append(centroid_result['xc'])
            centroid_results['yc'].append(centroid_result['yc'])
            centroid_results['x_bright'].append(centroid_result['x_bright'])
            centroid_results['y_bright'].append(centroid_result['y_bright'])
            centroid_results['val_bright'].append(centroid_result['val_bright'])
            centroid_results['x_width'].append(centroid_result['x_width'])
            centroid_results['y_width'].append(centroid_result['y_width'])
            centroid_results['n_pixels_above'].append(centroid_result['n_pixels_above'])
            centroid_results['sig3_pixels'].append(centroid_result['sig3_pixels'])
            centroid_results['sig5_pixels'].append(centroid_result['sig5_pixels'])
            centroid_results['sig10_pixels'].append(centroid_result['sig10_pixels'])
            all_poststamps.append(centroid_result['poststamps'])

            # Step 3: Perform photometry using centroid positions
            logger.debug("Performing photometry for image %d", i)

            xc_frame = centroid_result['xc']
            yc_frame = centroid_result['yc']

            # Call aper function
            mags, errap, sky, skyerr = aper(
                image=image,
                xc=xc_frame,
                yc=yc_frame,
                phpadu=phpadu,
                apr=apr,
                skyrad=skyrad,
                setskyval=setskyval,
                flux=True,
                silent=True
            )

            # Count bad pixels in apertures
            n_bad_pix_aperture = count_bad_pixels_in_apertures(
                bad_pixel_map, image.shape, xc_frame, yc_frame, apr, n_stars)

            # Convert to electrons
            mags_electrons = mags * gain
            errap_electrons = errap * gain
            sky_electrons = sky * gain
            skyerr_electrons = skyerr * gain

            # Store photometry results for each aperture (use original filename for consistency)
            for aper_idx, aper_radius in enumerate(apr):
                aper_name = f"aper{int(aper_radius)}"
                photometry_results[aper_name]['BJD'].append(bjd_obs)
                photometry_results[aper_name]['File'].append(str(filename))  # Store original path
                photometry_results[aper_name]['flux'].append(mags_electrons[aper_idx, :])
                photometry_results[aper_name]['flux_err'].append(errap_electrons[aper_idx, :])
                photometry_results[aper_name]['sky'].append(sky_electrons)
                photometry_results[aper_name]['sky_err'].append(skyerr_electrons)
                photometry_results[aper_name]['n_bad_pixels_in_aperture'].append(n_bad_pix_aperture[aper_idx, :])

            successful_images += 1

        except Exception as e:
            logger.error("Failed to process image %s: %s", filename, e)
            failed_images.append(filename)
            continue

    logger.info("Streaming processing complete: %d/%d images successful",
                successful_images, len(filenames))

    if successful_images == 0:
        raise RuntimeError("No images were successfully processed")

    # Write processed image list if images were saved
    if save_processed_images and processed_filenames:
        from create_lists import write_liste
        try:
            write_liste(processed_filenames, f"{run}_proc_{target}.list", outdir)
            logger.info("Created processed frame list: %s_proc_%s.list with %d files",
                        run, target, len(processed_filenames))
        except Exception as e:
            logger.error("Failed to write processed frame list: %s", e)

    # Write all results
    logger.info("Writing accumulated results to disk")

    # Create output directories
    centroiding_dir = outdir / target / "centroiding"
    photometry_dir = outdir / target / "photometry"
    os.makedirs(centroiding_dir, exist_ok=True)
    os.makedirs(photometry_dir, exist_ok=True)

    # Write centroiding results
    logger.info("Writing centroiding results")
    write_centroiding_results(centroiding_dir, centroid_results, all_poststamps, n_stars,
                              centroid_params['boxsize'])

    # Write photometry results
    logger.info("Writing photometry results")
    median_filter_window = config.get('photometry_settings', {}).get('median_filter_window', 21)
    write_photometry_results(photometry_dir, photometry_results, config, target, outdir,
                             median_filter_window)

    logger.info("All results written successfully")

    return successful_images, failed_images

def load_calibration_frames(outdir, run, filter_name, calib_params):
    """Load master calibration frames into memory."""
    logger.info("Loading master calibration frames")

    bias_file = outdir / 'calib' / f"{run}_master_bias.fits"
    dark_file = outdir / 'calib' / f"{run}_master_dark.fits"
    flat_file = outdir / 'calib' / f"{run}_master_flat_{filter_name}.fits"

    try:
        with fits.open(bias_file) as hdul:
            master_bias = hdul[0].data
        logger.debug("Loaded master bias: shape %s", master_bias.shape)

        with fits.open(dark_file) as hdul:
            master_dark = hdul[0].data
        logger.debug("Loaded master dark: shape %s", master_dark.shape)

        with fits.open(flat_file) as hdul:
            master_flat = hdul[0].data
        logger.debug("Loaded master flat: shape %s", master_flat.shape)

    except Exception as e:
        logger.error("Failed to load master calibration frames: %s", e)
        raise

    return {
        'bias': master_bias,
        'dark': master_dark,
        'flat': master_flat
    }