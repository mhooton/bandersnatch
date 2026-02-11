# Read parameters from files
import numpy as np
import os
import logging
import shutil
from astropy.io import fits
from liris import runset_filter
from utils import medabsdevclip
from pathlib import Path
import yaml
from astropy.table import Table

# Set up logger for this module
logger = logging.getLogger(__name__)


# Function to perform overscan correction
def overscan_corr(image, first_column, last_column  # , first_column_2=None, last_column_2=None
                  ):
    # This is a placeholder for the overscan correction function
    # You would need to implement this based on the original IDL function
    corrected = image.copy()

    # Calculate overscan value from the specified columns
    overscan = np.median(image[:, first_column:last_column + 1])
    logger.debug("Calculated overscan value: %.2f", overscan)

    # Apply correction
    corrected -= overscan

    # # If a second overscan region is specified
    # if first_column_2 is not None and last_column_2 is not None:
    #     overscan_2 = np.median(image[:, first_column_2:last_column_2 + 1])
    #     # Apply additional correction if needed

    return corrected


def calculate_readout_noise(filenames, image_extension, gain, subtract_overscan=False,
                            overscan_first_column=None, overscan_last_column=None):
    """
    Calculate readout noise from the first two bias frames.

    Parameters:
    -----------
    filenames : list
        List of bias frame filenames
    image_extension : int
        FITS extension containing image data
    gain : float
        Gain in electrons per ADU
    subtract_overscan : bool
        Whether to subtract overscan
    overscan_first_column : int
        First column of overscan region
    overscan_last_column : int
        Last column of overscan region

    Returns:
    --------
    float : Readout noise in electrons
    """
    if len(filenames) < 2:
        logger.warning("Need at least 2 bias frames to calculate readout noise")
        return 0.0

    try:
        # Read first two bias frames
        with fits.open(filenames[0]) as hdul:
            bias1 = hdul[image_extension].data.astype(np.float64)

        with fits.open(filenames[1]) as hdul:
            bias2 = hdul[image_extension].data.astype(np.float64)

        # Apply overscan correction if requested
        if subtract_overscan:
            bias1 = overscan_corr(bias1, overscan_first_column, overscan_last_column)
            bias2 = overscan_corr(bias2, overscan_first_column, overscan_last_column)

        # Calculate difference and standard deviation
        diff = bias1 - bias2
        adu_diff = np.std(diff)

        # Convert to electrons using gain and sqrt(2) factor
        # (sqrt(2) factor accounts for difference of two independent measurements)
        ron = adu_diff * gain / np.sqrt(2)

        logger.info("Calculated readout noise: %.2f electrons (%.2f ADU)", ron, adu_diff)
        return ron

    except Exception as e:
        logger.error("Failed to calculate readout noise: %s", e)
        return 0.0


def process_flat_session(filenames, run, outdir, filter, instrument, runset_cut, image_extension,
                         subtract_overscan, overscan_first_column, overscan_last_column,
                         clip, nlimit, session_name):
    """Process a single flat field session (dawn or dusk)"""
    logger.info("Processing %s session for filter %s", session_name, filter)

    # Filter LIRIS data if needed
    if instrument == 'LIRIS':
        original_count = len(filenames)
        filenames = runset_filter(filenames, runset_cut)
        if len(filenames) != original_count:
            logger.info("LIRIS runset filter: %d -> %d files", original_count, len(filenames))

    if not filenames:
        logger.warning("No files found for %s session", session_name)
        return None

    n_files = len(filenames)
    logger.info("Processing %s session with %d files", session_name, n_files)

    # Read in single fits image to find its dimensions
    try:
        with fits.open(filenames[0]) as hdul:
            image = hdul[image_extension].data

        nx = image.shape[1]  # Number of columns
        ny = image.shape[0]  # Number of rows
        logger.debug("Image dimensions: %d x %d pixels", nx, ny)

        # Define data_cube and its dimensions
        data_cube = np.zeros((ny, nx, n_files))
        logger.debug("Created data cube: %d x %d x %d", ny, nx, n_files)

    except Exception as e:
        logger.error("Failed to read first flat file %s: %s", filenames[0], e)
        raise

    # Process all flats in this session
    for file_loop, filename in enumerate(filenames):
        logger.debug("Reading %s flat %d/%d: %s", session_name, file_loop + 1, len(filenames), filename)

        try:
            with fits.open(filename) as hdul:
                image = hdul[image_extension].data
                exp_time = hdul[0].header["EXPTIME"]

            if subtract_overscan:
                logger.debug("Applying overscan correction to %s", filename)
                image = overscan_corr(
                    image,
                    overscan_first_column,
                    overscan_last_column,
                )

            # Apply bias and dark corrections
            image = image.astype(np.float64)

            bias_file = outdir / 'calib' / f"{run}_master_bias.fits"
            dark_file = outdir / 'calib' / f"{run}_master_dark.fits"

            try:
                with fits.open(bias_file) as hdul:
                    master_bias = hdul[0].data
                image -= master_bias
                logger.debug("Applied bias correction")

                with fits.open(dark_file) as hdul:
                    master_dark = hdul[0].data
                image -= master_dark * exp_time
                logger.debug("Applied dark correction (exp_time=%.2f)", exp_time)

            except Exception as e:
                logger.error("Failed to apply calibrations to %s: %s", filename, e)
                raise

            data_cube[:, :, file_loop] = image

        except Exception as e:
            logger.error("Failed to process flat file %s: %s", filename, e)
            raise

    # Create session master flat
    logger.debug("Creating %s session master flat with clip=%.1f, nlimit=%d", session_name, clip, nlimit)
    session_master = medabsdevclip(data_cube, clip, nlimit)
    session_master /= np.median(session_master)

    logger.info("%s session master flat created successfully", session_name.capitalize())
    return session_master


def copy_backup_calibration(backup_path, output_file, calib_type, filter=None):
    """Copy a backup calibration file to the expected output location"""
    if not os.path.exists(backup_path):
        filter_str = f" for filter {filter}" if filter else ""
        logger.error("Backup %s file not found: %s%s", calib_type, backup_path, filter_str)
        raise FileNotFoundError(f"Backup {calib_type} file not found: {backup_path}")

    try:
        # Ensure output directory exists
        os.makedirs(output_file.parent, exist_ok=True)

        # Copy the backup file
        shutil.copy2(backup_path, output_file)

        filter_str = f" for filter {filter}" if filter else ""
        logger.info("Using backup %s%s: copied %s -> %s", calib_type, filter_str, backup_path, output_file)

    except Exception as e:
        logger.error("Failed to copy backup %s file: %s", calib_type, e)
        raise


def load_master_flats_config(config_dir):
    """Load master flats configuration file."""
    master_flats_config_path = config_dir / "master_flats.yaml"
    try:
        with open(master_flats_config_path, 'r') as f:
            master_flats_config = yaml.safe_load(f)
        return master_flats_config
    except FileNotFoundError:
        logger.debug("Master flats config file not found: %s", master_flats_config_path)
        return None
    except yaml.YAMLError as e:
        logger.error("Error parsing master flats YAML config file: %s", e)
        return None


def find_backup_flat_for_date(master_flats_config, filter_name, observation_date):
    """
    Find the appropriate backup flat for a given filter and observation date.

    Parameters:
    -----------
    master_flats_config : dict
        Master flats configuration dictionary
    filter_name : str
        Filter name (e.g., 'zYJ')
    observation_date : str
        Observation date in YYYYMMDD format

    Returns:
    --------
    str or None : Path to backup flat file, or None if not found
    """
    if not master_flats_config or 'filters' not in master_flats_config:
        return None

    if filter_name not in master_flats_config['filters']:
        logger.debug("Filter %s not found in master flats config", filter_name)
        return None

    filter_config = master_flats_config['filters'][filter_name]

    # Find the most recent start_date that is <= observation_date
    best_match = None
    best_start_date = None

    for entry in filter_config:
        start_date = entry['start_date']
        if start_date <= observation_date:
            if best_start_date is None or start_date > best_start_date:
                best_match = entry
                best_start_date = start_date

    if best_match:
        logger.info("Found backup flat for filter %s (date %s): using config from %s",
                    filter_name, observation_date, best_start_date)
        return best_match['path']
    else:
        logger.debug("No suitable backup flat found for filter %s on date %s",
                     filter_name, observation_date)
        return None

def make_master_calibration(type, outdir, run, instrument, config, filter="zYJ", subtract_overscan=False,
                            runset_cut=None,
                            image_extension=0, overscan_first_column=None, overscan_last_column=None,
                            overscan_first_column2=None, overscan_last_column2=None, clip=5, nlimit=5):
    logger.info("Creating master %s calibration", type)
    logger.debug("Parameters: filter=%s, clip=%.1f, nlimit=%d, image_ext=%d",
                 filter, clip, nlimit, image_extension)

    # Ensure output directory exists
    calib_dir = outdir / 'calib'
    os.makedirs(calib_dir, exist_ok=True)
    logger.debug("Created calibration directory: %s", calib_dir)

    # Check for forced backup flats (only for flat type)
    if type == "flat":
        force_backup_flats = config.get('calibration_params', {}).get('force_backup_flats', False)

        if force_backup_flats:
            logger.info("Forced backup flats enabled, checking master flats configuration")

            # Load master flats config
            config_dir = Path(config.get('_config_dir', '.'))  # This will be set by run.py
            master_flats_config = load_master_flats_config(config_dir)

            if master_flats_config:
                observation_date = config['instrument_settings']['date']
                backup_flat_path = find_backup_flat_for_date(master_flats_config, filter, observation_date)

                if backup_flat_path:
                    suffix = f"_{filter}"
                    output_file = calib_dir / f"{run}_master_{type + suffix}.fits"

                    logger.info("Using forced backup flat for filter %s from date-based config", filter)
                    copy_backup_calibration(backup_flat_path, output_file, "flat", filter)
                    return
                else:
                    logger.warning(
                        "No suitable backup flat found in master flats config, falling back to normal creation")
            else:
                logger.warning("Could not load master flats config, falling back to normal creation")

    filenames = []

    if type == "flat":
        suffix = f"_{filter}"
        output_file = calib_dir / f"{run}_master_{type + suffix}.fits"
        logger.info("Processing flat field for filter %s", filter)

        # Handle dawn/dusk flat processing
        dawn_file = outdir / "calib" / f"{run}_flat_{filter}_dawn.list"
        dusk_file = outdir / "calib" / f"{run}_flat_{filter}_dusk.list"
        old_style_file = outdir / "calib" / f"{run}_flat_{filter}.list"

        dawn_filenames = []
        dusk_filenames = []

        # Check for new style dawn/dusk files
        if dawn_file.exists():
            logger.debug("Found dawn flat list: %s", dawn_file)
            with open(dawn_file) as f:
                dawn_filenames = [line.strip() for line in f if line.strip()]
            logger.info("Found %d dawn flat files", len(dawn_filenames))

        if dusk_file.exists():
            logger.debug("Found dusk flat list: %s", dusk_file)
            with open(dusk_file) as f:
                dusk_filenames = [line.strip() for line in f if line.strip()]
            logger.info("Found %d dusk flat files", len(dusk_filenames))

        # Fallback to old style single file
        if not dawn_filenames and not dusk_filenames and old_style_file.exists():
            logger.warning("Using old-style flat list %s", old_style_file)
            logger.warning("Consider splitting into dawn/dusk files for better star rejection")
            with open(old_style_file) as f:
                filenames = [line.strip() for line in f if line.strip()]
            logger.info("Found %d flat files in old-style list", len(filenames))

        # Check if we have any flat files
        has_flat_files = bool(dawn_filenames or dusk_filenames or filenames)

        if not has_flat_files:
            # No flat lists found, try to use master flats config first
            logger.warning("No flat field lists found for filter %s", filter)

            # Try master flats config first
            config_dir = Path(config.get('_config_dir', '.'))
            master_flats_config = load_master_flats_config(config_dir)

            if master_flats_config:
                observation_date = config['instrument_settings']['date']
                backup_flat_path = find_backup_flat_for_date(master_flats_config, filter, observation_date)

                if backup_flat_path:
                    logger.info("Using backup flat for filter %s from master flats config", filter)
                    copy_backup_calibration(backup_flat_path, output_file, "flat", filter)
                    return

            # Fallback to old backup system if master flats config doesn't work
            backup_flats = config.get('calibration_params', {}).get('backup_master_flats', {})
            if filter in backup_flats:
                backup_path = backup_flats[filter]
                logger.info("Using legacy backup master flat for filter %s", filter)
                copy_backup_calibration(backup_path, output_file, "flat", filter)
                return
            else:
                logger.error("No backup master flat found for filter %s", filter)
                raise ValueError(f"No flat field files or backup found for filter {filter}")

        if dawn_filenames or dusk_filenames:
            # Process dawn/dusk sessions separately
            session_masters = []

            if dawn_filenames:
                dawn_master = process_flat_session(
                    dawn_filenames, run, outdir, filter, instrument, runset_cut,
                    image_extension, subtract_overscan, overscan_first_column,
                    overscan_last_column, clip, nlimit, "dawn"
                )
                if dawn_master is not None:
                    session_masters.append(dawn_master)

            if dusk_filenames:
                dusk_master = process_flat_session(
                    dusk_filenames, run, outdir, filter, instrument, runset_cut,
                    image_extension, subtract_overscan, overscan_first_column,
                    overscan_last_column, clip, nlimit, "dusk"
                )
                if dusk_master is not None:
                    session_masters.append(dusk_master)

            if not session_masters:
                logger.error("No valid flat sessions found for filter %s", filter)
                raise ValueError(f"No valid flat sessions found for filter {filter}")

            # Combine session masters
            if len(session_masters) == 1:
                logger.info("Only one session available, using that as master flat")
                master_frame = session_masters[0]
            else:
                logger.info("Combining %d session masters", len(session_masters))
                combined_cube = np.stack(session_masters, axis=2)
                master_frame = np.median(combined_cube, axis=2)
                master_frame /= np.median(master_frame)

            logger.info("Final master flat created from %d session(s)", len(session_masters))

        elif filenames:
            # Process old-style single list (original behavior)
            logger.info("Processing %d flat files using old-style method", len(filenames))

            # Filter LIRIS data if needed
            if instrument == 'LIRIS':
                original_count = len(filenames)
                filenames = runset_filter(filenames, runset_cut)
                if len(filenames) != original_count:
                    logger.info("LIRIS runset filter: %d -> %d files", original_count, len(filenames))

            n_files = len(filenames)

            # Read in single fits image to find its dimensions
            try:
                with fits.open(filenames[0]) as hdul:
                    image = hdul[image_extension].data

                nx = image.shape[1]  # Number of columns
                ny = image.shape[0]  # Number of rows
                logger.debug("Image dimensions: %d x %d pixels", nx, ny)

                # Define data_cube and its dimensions
                data_cube = np.zeros((ny, nx, n_files))
                logger.debug("Created data cube: %d x %d x %d", ny, nx, n_files)

            except Exception as e:
                logger.error("Failed to read first flat file %s: %s", filenames[0], e)
                raise

            # Process all flats
            logger.info("Populating data cube with %d flat files", n_files)
            for file_loop, filename in enumerate(filenames):
                logger.debug("Reading flat %d/%d: %s", file_loop + 1, len(filenames), filename)

                try:
                    with fits.open(filename) as hdul:
                        image = hdul[image_extension].data
                        exp_time = hdul[0].header["EXPTIME"]

                    if subtract_overscan:
                        logger.debug("Applying overscan correction")
                        image = overscan_corr(
                            image,
                            overscan_first_column,
                            overscan_last_column,
                        )

                    image = image.astype(np.float64)

                    bias_file = outdir / 'calib' / f"{run}_master_bias.fits"
                    dark_file = outdir / 'calib' / f"{run}_master_dark.fits"

                    with fits.open(bias_file) as hdul:
                        master_bias = hdul[0].data
                    image -= master_bias

                    with fits.open(dark_file) as hdul:
                        master_dark = hdul[0].data
                    image -= master_dark * exp_time

                    data_cube[:, :, file_loop] = image

                except Exception as e:
                    logger.error("Failed to process flat file %s: %s", filename, e)
                    raise

            # Create master flat
            logger.debug("Creating master flat with clip=%.1f, nlimit=%d", clip, nlimit)
            master_frame = medabsdevclip(data_cube, clip, nlimit)
            master_frame /= np.median(master_frame)

    else:
        # Handle bias and dark calibrations
        suffix = ""
        output_file = calib_dir / f"{run}_master_{type + suffix}.fits"
        list_file = outdir / "calib" / f"{run}_{type + suffix}.list"

        logger.info("Processing %s calibration from %s", type, list_file)

        # Check if list file exists
        if not list_file.exists():
            logger.warning("No %s list file found: %s", type, list_file)

            # Try to use backup
            backup_key = f"backup_master_{type}"
            backup_path = config.get('calibration_params', {}).get(backup_key)

            if backup_path:
                logger.info("Attempting to use backup master %s", type)
                copy_backup_calibration(backup_path, output_file, type)
                return
            else:
                logger.error("No backup master %s specified in config", type)
                raise FileNotFoundError(f"No {type} list file found and no backup specified: {list_file}")

        try:
            with open(list_file) as f:
                filenames = [line.strip() for line in f if line.strip()]
            logger.info("Found %d %s files", len(filenames), type)
        except Exception as e:
            logger.error("Failed to read %s list file %s: %s", type, list_file, e)
            raise

        # Calculate readout noise for bias frames
        if type == "bias" and len(filenames) >= 2:
            logger.info("Calculating readout noise from bias frames...")
            gain = config['instrument_config']['gain']

            ron = calculate_readout_noise(
                filenames, image_extension, gain, subtract_overscan,
                overscan_first_column, overscan_last_column
            )

            # Save readout noise to file
            ron_file = calib_dir / "readoutnoise.txt"
            try:
                with open(ron_file, 'w') as f:
                    f.write(f'{ron:.3f}\n')
                logger.info("Saved readout noise to: %s", ron_file)
            except Exception as e:
                logger.error("Failed to save readout noise file: %s", e)

        # Filter LIRIS data if needed
        if instrument == 'LIRIS':
            original_count = len(filenames)
            filenames = runset_filter(filenames, runset_cut)
            if len(filenames) != original_count:
                logger.info("LIRIS runset filter: %d -> %d files", original_count, len(filenames))

        n_files = len(filenames)

        # Read in single fits image to find its dimensions
        try:
            with fits.open(filenames[0]) as hdul:
                image = hdul[image_extension].data

            nx = image.shape[1]  # Number of columns
            ny = image.shape[0]  # Number of rows
            logger.debug("Image dimensions: %d x %d pixels", nx, ny)

            # Define data_cube and its dimensions
            data_cube = np.zeros((ny, nx, n_files))
            logger.debug("Created data cube: %d x %d x %d", ny, nx, n_files)

        except Exception as e:
            logger.error("Failed to read first %s file %s: %s", type, filenames[0], e)
            raise

        # FEED IN ALL BIASES/DARKS TO DATA CUBE
        logger.info("Populating data cube with %d %s files", n_files, type)
        for file_loop, filename in enumerate(filenames):
            logger.debug("Reading %s %d/%d: %s", type, file_loop + 1, len(filenames), filename)

            try:
                with fits.open(filename) as hdul:
                    image = hdul[image_extension].data
                    exp_time = hdul[0].header["EXPTIME"]

                if subtract_overscan:
                    logger.debug("Applying overscan correction")
                    image = overscan_corr(
                        image,
                        overscan_first_column,
                        overscan_last_column,
                    )

                if type in ["dark"]:
                    image = image.astype(np.float64)
                    bias_file = outdir / 'calib' / f"{run}_master_bias.fits"

                    with fits.open(bias_file) as hdul:
                        master_bias = hdul[0].data
                    image -= master_bias
                    logger.debug("Applied bias correction to dark frame")

                    if type == "dark":
                        image /= exp_time
                        logger.debug("Normalized dark by exposure time: %.2f", exp_time)

                data_cube[:, :, file_loop] = image

            except Exception as e:
                logger.error("Failed to process %s file %s: %s", type, filename, e)
                raise

        # Construct master frame by finding median of each pixel, after sigma clipping and limit
        logger.debug("Creating master %s with clip=%.1f, nlimit=%d", type, clip, nlimit)
        master_frame = medabsdevclip(data_cube, clip, nlimit)

        # Calculate dark current for dark frames
        if type == "dark":
            logger.info("Calculating dark current from master dark frame...")
            gain = config['instrument_config']['gain']

            # Dark current = gain * median of master dark frame
            # (master dark is already bias-corrected and normalized by exposure time)
            dark_current = gain * np.median(master_frame)
            logger.info("Calculated dark current: %.3f electrons/sec/pixel", dark_current)

            # Save dark current to file
            dark_current_file = calib_dir / "darkcurrent.txt"
            try:
                with open(dark_current_file, 'w') as f:
                    f.write(f'{dark_current:.3f}\n')
                logger.info("Saved dark current to: %s", dark_current_file)
            except Exception as e:
                logger.error("Failed to save dark current file: %s", e)

    # Create a new FITS file with the master frame
    logger.debug("Writing master %s to: %s", type, output_file)

    try:
        hdu = fits.PrimaryHDU(master_frame)
        hdu.writeto(output_file, overwrite=True)
        logger.info("Master %s frame created successfully: %s", type, output_file)

        # Log some statistics about the master frame
        logger.debug("Master %s statistics: mean=%.2f, median=%.2f, std=%.2f",
                     type, np.mean(master_frame), np.median(master_frame), np.std(master_frame))

    except Exception as e:
        logger.error("Failed to write master %s file %s: %s", type, output_file, e)
        raise


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running master_calibrations.py as standalone script")
    make_master_calibration()