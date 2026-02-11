import numpy as np
import os
import logging
from astropy.io import fits
from master_calibrations import overscan_corr
from create_lists import write_liste

from utils import clean_bad_pixels

# Set up logger for this module
logger = logging.getLogger(__name__)


def reduce_science_frames(outdir, run, target, to_overscan_correct=False, image_extension=0, bad_pixel_map=None):
    logger.info("Starting science frame reduction for target %s", target)
    logger.debug("Parameters: run=%s, overscan_correct=%s, image_ext=%d",
                 run, to_overscan_correct, image_extension)

    filenames = []
    list_file = outdir / "calib" / f"{run}_image_{target}.list"

    try:
        with open(list_file) as f:
            filenames = [line.strip() for line in f if line.strip()]
        logger.info("Found %d science frames for target %s", len(filenames), target)
    except Exception as e:
        logger.error("Failed to read science image list %s: %s", list_file, e)
        raise

    if not filenames:
        logger.error("No science frames found for target %s", target)
        raise ValueError(f"No science frames found for target {target}")

    # Get filter from first image
    try:
        with fits.open(filenames[0]) as hdul:
            filter = hdul[0].header['FILTER']
        logger.info("Target %s uses filter: %s", target, filter)
    except Exception as e:
        logger.error("Failed to read filter from first science frame %s: %s", filenames[0], e)
        raise

    # Load master calibration frames
    bias_file = outdir / 'calib' / f"{run}_master_bias.fits"
    dark_file = outdir / 'calib' / f"{run}_master_dark.fits"
    flat_file = outdir / 'calib' / f"{run}_master_flat_{filter}.fits"

    logger.debug("Loading master calibration frames:")
    logger.debug("  Bias: %s", bias_file)
    logger.debug("  Dark: %s", dark_file)
    logger.debug("  Flat: %s", flat_file)

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

    # Create output directories
    target_dir = outdir / target
    run_dir = target_dir / run

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    logger.debug("Created output directories: %s", run_dir)

    output_filenames = []
    successful_reductions = 0

    logger.info("Processing %d science frames for target %s", len(filenames), target)

    for file_loop, filename in enumerate(filenames):
        logger.debug("Processing science frame %d/%d: %s",
                     file_loop + 1, len(filenames), filename)

        try:
            with fits.open(filename) as hdul:
                image = hdul[image_extension].data
                image = image.astype(np.float64)
                exp_time = hdul[0].header["EXPTIME"]

                # Copy essential headers
                header = hdul[0].header.copy()

                logger.debug("Image shape: %s, exposure time: %.2f", image.shape, exp_time)

            # Apply calibrations
            original_median = np.median(image)

            if to_overscan_correct:
                logger.debug("Applying overscan correction")
                image = overscan_corr(image)

            logger.debug("Applying bias correction")
            image -= master_bias

            logger.debug("Applying dark correction (scaled by exp_time=%.2f)", exp_time)
            image -= master_dark * exp_time

            logger.debug("Applying flat field correction")
            image /= master_flat

            # Apply bad pixel correction if BPM is available
            if bad_pixel_map is not None:
                logger.debug("Applying bad pixel correction to %s", filename)
                image = clean_bad_pixels(image, bad_pixel_map)

            final_median = np.median(image)
            logger.debug("Reduction complete: median %.1f -> %.1f", original_median, final_median)

            # Create output filename and path
            output_filename = f"proc{filename.split('/')[-1]}"
            output_path = run_dir / output_filename

            # Create new FITS file with processed data
            new_hdu = fits.PrimaryHDU(data=image, header=header)
            new_hdu.writeto(output_path, overwrite=True)

            logger.debug("Wrote reduced frame: %s", output_path)
            output_filenames.append(output_filename)
            successful_reductions += 1

        except Exception as e:
            logger.error("Failed to process science frame %s: %s", filename, e)
            # Continue processing other frames rather than failing completely
            logger.warning("Skipping failed frame and continuing with remaining frames")
            continue

    logger.info("Successfully reduced %d/%d science frames for target %s",
                successful_reductions, len(filenames), target)

    if successful_reductions == 0:
        logger.error("No science frames were successfully reduced for target %s", target)
        raise RuntimeError(f"No science frames were successfully reduced for target {target}")

    # Write list of processed files
    try:
        write_liste(output_filenames, f"{run}_proc_{target}.list", outdir)
        logger.info("Created processed frame list: %s_proc_%s.list with %d files",
                    run, target, len(output_filenames))
    except Exception as e:
        logger.error("Failed to write processed frame list: %s", e)
        raise

    logger.info("Science frame reduction completed for target %s", target)


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running reduce_science.py as standalone script")
    # You would call reduce_science_frames() here with appropriate parameters