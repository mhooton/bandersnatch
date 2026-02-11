#!/usr/bin/env python3
"""
Astronomy pipeline runner with YAML configuration support.
"""

import argparse
import os
import sys
import logging
import logging.handlers
from pathlib import Path
import yaml

from create_lists import create_lists
from master_calibrations import make_master_calibration
from reduce_science import reduce_science_frames
from utils import find_targets, find_flat_filters
from astropy.io import fits


def load_config(config_path):
    """Load and validate YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}")
        sys.exit(1)


def load_targets_config(config_dir):
    """Load targets.yaml configuration file."""
    targets_config_path = config_dir / "targets.yaml"
    try:
        with open(targets_config_path, 'r') as f:
            targets_config = yaml.safe_load(f)
        return targets_config
    except FileNotFoundError:
        print(f"Error: Targets config file '{targets_config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing targets YAML config file: {e}")
        sys.exit(1)


def find_target_config_for_date(targets_config, target_name, observation_date):
    """
    Find the appropriate target configuration for a given target and observation date.

    Parameters:
    -----------
    targets_config : dict
        Targets configuration dictionary
    target_name : str
        Target name (e.g., 'Sp0830+0947')
    observation_date : str
        Observation date in YYYYMMDD format

    Returns:
    --------
    dict or None : Target configuration dictionary, or None if not found
    """
    logger = logging.getLogger(__name__)

    if not targets_config or 'targets' not in targets_config:
        return None

    if target_name not in targets_config['targets']:
        logger.debug("Target %s not found in targets config", target_name)
        return None

    target_config = targets_config['targets'][target_name]

    # Check if this is the new date-based format (list) or legacy format (dict)
    if isinstance(target_config, list):
        # New date-based format - find the most recent start_date that is <= observation_date
        best_match = None
        best_start_date = None

        for entry in target_config:
            start_date = entry['start_date']
            if start_date <= observation_date:
                if best_start_date is None or start_date > best_start_date:
                    best_match = entry
                    best_start_date = start_date

        if best_match:
            logger.info("Found target config for %s (date %s): using config from %s",
                        target_name, observation_date, best_start_date)
            return best_match
        else:
            logger.error("No suitable target config found for %s on date %s",
                         target_name, observation_date)
            return None
    else:
        # Legacy format - return the config directly
        logger.warning("Using legacy format for target %s. Consider updating to date-based format.", target_name)
        return target_config


def load_instrument_config(instrument_name, config_dir):
    """Load instrument-specific YAML configuration file."""
    instrument_config_path = config_dir / f"{instrument_name}.yaml"
    try:
        with open(instrument_config_path, 'r') as f:
            instrument_config = yaml.safe_load(f)
        return instrument_config
    except FileNotFoundError:
        print(f"Error: Instrument config file '{instrument_config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing instrument YAML config file: {e}")
        sys.exit(1)


def resolve_target_positions(date_config, targets_config, all_targets):
    """
    Resolve target positions and tracking stars using targets.yaml configuration.
    Now supports both dictionary and list formats for star0_positions, with dictionary
    format determining which targets to process.

    Parameters:
    -----------
    date_config : dict
        Date-specific configuration containing star0_positions
    targets_config : dict
        Targets configuration from targets.yaml
    all_targets : list
        List of all target names found in the file system

    Returns:
    --------
    tuple : (targets_to_process, initial_positions, tracking_stars)
        Lists of targets to process, their initial positions, and tracking stars
    """
    logger = logging.getLogger(__name__)

    star0_positions = date_config['centroid_settings']['star0_positions']
    observation_date = date_config['instrument_settings']['date']

    # Check if star0_positions is a dictionary or list (backward compatibility)
    if isinstance(star0_positions, dict):
        # New dictionary format - extract targets from dictionary keys
        targets_to_process = list(star0_positions.keys())
        logger.info("Using dictionary format for star0_positions")
        logger.info("Targets specified for processing: %s", targets_to_process)

        # Validate that all specified targets exist in the file system
        missing_targets = [target for target in targets_to_process if target not in all_targets]
        if missing_targets:
            error_msg = f"Targets specified in star0_positions not found in file system: {missing_targets}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log any targets being skipped
        skipped_targets = [target for target in all_targets if target not in targets_to_process]
        if skipped_targets:
            logger.info("Skipping targets not specified in star0_positions: %s", skipped_targets)

    else:
        # Legacy list format - use all targets in alphabetical order
        targets_to_process = sorted(all_targets)
        logger.info("Using legacy list format for star0_positions")
        logger.warning("List format for star0_positions is deprecated. Consider using dictionary format.")

        # Validate that we have the right number of star0 positions
        if len(star0_positions) != len(targets_to_process):
            error_msg = (f"Mismatch: found {len(targets_to_process)} targets but {len(star0_positions)} "
                         f"star0_positions in config. Targets: {targets_to_process}")
            logger.error(error_msg)
            raise ValueError(error_msg)

    initial_positions = []
    tracking_stars = []

    logger.info("Resolving positions for %d targets using targets.yaml", len(targets_to_process))

    for i, target in enumerate(targets_to_process):
        logger.debug("Processing target %d: %s", i, target)

        # Find the appropriate target configuration for this date
        target_info = find_target_config_for_date(targets_config, target, observation_date)

        if target_info is None:
            error_msg = f"No suitable target configuration found for '{target}' on date {observation_date}"
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Get date-specific star 0 position based on format
        if isinstance(star0_positions, dict):
            date_star0_pos = star0_positions[target]
        else:
            date_star0_pos = star0_positions[i]
        logger.debug("Date-specific star 0 position for %s: %s", target, date_star0_pos)

        # Get reference positions from targets.yaml
        reference_positions = target_info['all_star_positions']
        reference_star0_pos = reference_positions[0]
        logger.debug("Reference star 0 position for %s: %s", target, reference_star0_pos)

        # Calculate offset
        dx = date_star0_pos[0] - reference_star0_pos[0]
        dy = date_star0_pos[1] - reference_star0_pos[1]
        logger.debug("Calculated offset for %s: dx=%.1f, dy=%.1f", target, dx, dy)

        # Apply offset to all star positions
        adjusted_positions = []
        for ref_pos in reference_positions:
            adjusted_pos = [ref_pos[0] + dx, ref_pos[1] + dy]
            adjusted_positions.append(adjusted_pos)

        logger.debug("Adjusted positions for %s: %s", target, adjusted_positions)

        # Get tracking star index
        tracking_star = target_info['tracking_star']

        # Validate tracking star index
        if tracking_star >= len(reference_positions):
            error_msg = (f"Invalid tracking_star index {tracking_star} for target '{target}'. "
                         f"Target has {len(reference_positions)} stars (indices 0-{len(reference_positions) - 1})")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug("Tracking star for %s: %d", target, tracking_star)

        initial_positions.append(adjusted_positions)
        tracking_stars.append(tracking_star)

    logger.info("Successfully resolved positions for %d targets", len(targets_to_process))
    return targets_to_process, initial_positions, tracking_stars


def copy_config_files(config_path, targets_config, targets_used, outdir, config):
    """
    Copy configuration files to the log directory for record keeping.

    Parameters:
    -----------
    config_path : Path
        Path to the original date configuration file
    targets_config : dict
        Full targets configuration dictionary
    targets_used : list
        List of target names that were processed in this run
    outdir : Path
        Output directory for the run
    config : dict
        Date configuration dictionary
    """
    logger = logging.getLogger(__name__)

    # Get log directory from config
    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', 'logs')
    log_directory = outdir / log_dir

    try:
        # Copy the date config file
        date_config_dest = log_directory / config_path.name
        import shutil
        shutil.copy2(config_path, date_config_dest)
        logger.info(f"Copied date config to: {date_config_dest}")

        # Create filtered targets config with only the targets used in this run
        if targets_config and targets_used:
            observation_date = config['instrument_settings']['date']
            filtered_targets_config = {'targets': {}}

            for target in targets_used:
                if target in targets_config['targets']:
                    target_config = targets_config['targets'][target]

                    # If it's the new date-based format, include only the relevant date entry
                    if isinstance(target_config, list):
                        # Find the config that was actually used for this date
                        from run import find_target_config_for_date  # Import the function we just created
                        used_config = find_target_config_for_date(targets_config, target, observation_date)
                        if used_config:
                            # Store as the new format but only with the used entry
                            filtered_targets_config['targets'][target] = [used_config]
                    else:
                        # Legacy format - copy as-is
                        filtered_targets_config['targets'][target] = target_config

            # Write filtered targets config
            targets_config_dest = log_directory / "targets_used.yaml"
            with open(targets_config_dest, 'w') as f:
                yaml.dump(filtered_targets_config, f, default_flow_style=False, indent=2)

            logger.info(f"Created filtered targets config: {targets_config_dest}")
            logger.info(f"Targets included: {list(filtered_targets_config['targets'].keys())}")
        else:
            logger.debug("No targets configuration to copy (centroiding/photometry not enabled)")

    except Exception as e:
        logger.error(f"Failed to copy configuration files: {e}")
        # Don't raise - this is not critical for pipeline operation


def setup_logging(config, outdir):
    """Set up logging configuration based on config file and output directory."""
    logging_config = config.get('logging', {})
    inst_settings = config.get('instrument_settings', {})

    # Extract logging settings with defaults
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    console_output = logging_config.get('console_output', True)
    file_output = logging_config.get('file_output', True)
    log_dir = logging_config.get('log_dir', 'logs')
    log_filename_pattern = logging_config.get('log_filename', '{inst}_{date}_{run}.log')
    max_file_size_mb = logging_config.get('max_file_size_mb', 50)
    backup_count = logging_config.get('backup_count', 5)
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Set up file handler
    if file_output:
        # Create log directory
        log_directory = outdir / log_dir
        log_directory.mkdir(exist_ok=True)

        # Generate log filename
        log_filename = log_filename_pattern.format(
            inst=inst_settings.get('inst', 'unknown'),
            date=inst_settings.get('date', 'unknown'),
            run=inst_settings.get('run', 'unknown')
        )
        log_filepath = log_directory / log_filename

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_filepath,
            maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_filepath}")

    return logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run astronomy data reduction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('config',
                        help='Path to YAML configuration file (searches in topdir/bandersnatch_runs/configs/ if not found locally)')

    # Override options
    parser.add_argument('--date', help='Override observation date (YYYYMMDD)')
    parser.add_argument('--inst', help='Override instrument name')
    parser.add_argument('--run', help='Override run number')

    # Staging options
    parser.add_argument('--local-staging', action='store_true',
                        help='Use local staging directory for processing')
    parser.add_argument('--staging-dir',
                        help='Local staging directory path')

    # Processing step flags - use separate attributes to avoid conflicts
    parser.add_argument('--make-lists', action='store_true',
                        help='Force enable making file lists')
    parser.add_argument('--no-make-lists', action='store_true',
                        help='Force disable making file lists')

    parser.add_argument('--make-bias', action='store_true',
                        help='Force enable making master bias')
    parser.add_argument('--no-make-bias', action='store_true',
                        help='Force disable making master bias')

    parser.add_argument('--make-dark', action='store_true',
                        help='Force enable making master dark')
    parser.add_argument('--no-make-dark', action='store_true',
                        help='Force disable making master dark')

    parser.add_argument('--make-flat', action='store_true',
                        help='Force enable making master flat')
    parser.add_argument('--no-make-flat', action='store_true',
                        help='Force disable making master flat')

    parser.add_argument('--reduce-science', action='store_true',
                        help='Force enable science frame reduction')
    parser.add_argument('--no-reduce-science', action='store_true',
                        help='Force disable science frame reduction')

    parser.add_argument('--centroid', action='store_true',
                        help='Force enable centroiding')
    parser.add_argument('--no-centroid', action='store_true',
                        help='Force disable centroiding')

    parser.add_argument('--photometry', action='store_true',
                        help='Force enable aperture photometry')
    parser.add_argument('--no-photometry', action='store_true',
                        help='Force disable aperture photometry')

    parser.add_argument('--skip-existing-bias', action='store_true',
                        help='Skip creating master bias if it already exists')
    parser.add_argument('--no-skip-existing-bias', action='store_true',
                        help='Force create master bias even if it exists')

    parser.add_argument('--skip-existing-dark', action='store_true',
                        help='Skip creating master dark if it already exists')
    parser.add_argument('--no-skip-existing-dark', action='store_true',
                        help='Force create master dark even if it exists')

    parser.add_argument('--skip-existing-flat', action='store_true',
                        help='Skip creating master flat if it already exists')
    parser.add_argument('--no-skip-existing-flat', action='store_true',
                        help='Force create master flat even if it exists')

    parser.add_argument('--skip-existing-lists', action='store_true',
                        help='Skip creating file lists if they already exist')
    parser.add_argument('--no-skip-existing-lists', action='store_true',
                        help='Force create file lists even if they exist')

    parser.add_argument('--boxsize', type=int,
                        help='Override centroiding boxsize')

    parser.add_argument('--aper-min', type=int,
                        help='Override minimum aperture radius')
    parser.add_argument('--aper-max', type=int,
                        help='Override maximum aperture radius')

    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose (DEBUG) logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Enable quiet (WARNING) logging')
    parser.add_argument('--log-file', help='Override log file location')
    parser.add_argument('--console-only', action='store_true',
                        help='Log to console only (no file output)')

    return parser.parse_args()


def merge_config_with_args(config, args):
    """Merge command line arguments with config file settings."""
    # Override basic settings if provided
    if args.date:
        config['instrument_settings']['date'] = args.date
    if args.inst:
        config['instrument_settings']['inst'] = args.inst
    if args.run:
        config['instrument_settings']['run'] = args.run

    # Update run_name (always generated from inst and date)
    inst = config['instrument_settings']['inst']
    date = config['instrument_settings']['date']
    config['instrument_settings']['run_name'] = f"{inst}_{date}"

    # Override processing flags if explicitly set
    processing_flags = config['processing_flags']

    # Check for enable/disable flag pairs
    if args.make_lists:
        processing_flags['to_make_lists'] = True
    elif args.no_make_lists:
        processing_flags['to_make_lists'] = False

    if args.make_bias:
        processing_flags['to_make_master_bias'] = True
    elif args.no_make_bias:
        processing_flags['to_make_master_bias'] = False

    if args.make_dark:
        processing_flags['to_make_master_dark'] = True
    elif args.no_make_dark:
        processing_flags['to_make_master_dark'] = False

    if args.make_flat:
        processing_flags['to_make_master_flat'] = True
    elif args.no_make_flat:
        processing_flags['to_make_master_flat'] = False

    if args.reduce_science:
        processing_flags['to_reduce_science_images'] = True
    elif args.no_reduce_science:
        processing_flags['to_reduce_science_images'] = False

    if args.centroid:
        processing_flags['to_centroid'] = True
    elif args.no_centroid:
        processing_flags['to_centroid'] = False

    if args.photometry:
        processing_flags['to_aperture_photometry'] = True
    elif args.no_photometry:
        processing_flags['to_aperture_photometry'] = False

        # Override skip existing calibration flags if explicitly set
    calib_params = config['calibration_params']

    if args.skip_existing_bias:
        calib_params['skip_existing_bias'] = True
    elif args.no_skip_existing_bias:
        calib_params['skip_existing_bias'] = False

    if args.skip_existing_dark:
        calib_params['skip_existing_dark'] = True
    elif args.no_skip_existing_dark:
        calib_params['skip_existing_dark'] = False

    if args.skip_existing_flat:
        calib_params['skip_existing_flat'] = True
    elif args.no_skip_existing_flat:
        calib_params['skip_existing_flat'] = False

    if args.skip_existing_lists:
        calib_params['skip_existing_lists'] = True
    elif args.no_skip_existing_lists:
        calib_params['skip_existing_lists'] = False

        # Override centroiding settings if specified
    if args.boxsize:
        config['centroid_settings']['boxsize'] = args.boxsize

    # Override photometry settings if specified
    if args.aper_min:
        config['photometry_settings']['aper_min'] = args.aper_min
    if args.aper_max:
        config['photometry_settings']['aper_max'] = args.aper_max

    # Override logging settings if specified
    if args.verbose:
        config.setdefault('logging', {})['level'] = 'DEBUG'
    elif args.quiet:
        config.setdefault('logging', {})['level'] = 'WARNING'

    if args.console_only:
        config.setdefault('logging', {})['file_output'] = False

    if args.log_file:
        config.setdefault('logging', {})['log_filename'] = Path(args.log_file).name
        config.setdefault('logging', {})['log_dir'] = str(Path(args.log_file).parent)

    return config


def setup_paths(config, is_docker, use_local_staging=False, local_staging_dir=None):
    """Set up paths based on environment and staging configuration."""
    inst_settings = config['instrument_settings']
    path_settings = config['paths']

    if is_docker:
        # Docker paths remain the same
        topdir = Path("/app")
        rawdir = topdir / "data" / inst_settings['inst'] / inst_settings['date']
        outdir = topdir / "output" / inst_settings['run_name']
    elif use_local_staging and local_staging_dir:
        # Local staging paths
        staging_base = Path(local_staging_dir)
        topdir = staging_base
        rawdir = staging_base / "data" / inst_settings['inst'] / inst_settings['date']
        outdir = staging_base / "bandersnatch_runs" / inst_settings['run_name']
    else:
        # Original local conda environment paths
        topdir = Path(path_settings['topdir']).expanduser()
        rawdir = topdir / "data" / inst_settings['inst'] / inst_settings['date']
        outdir = topdir / "bandersnatch_runs" / inst_settings['run_name']

    return topdir, rawdir, outdir


def main():
    """Main pipeline execution function."""
    args = parse_arguments()

    # Detect if running in Docker container
    is_docker = os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'

    # Handle config file path resolution
    config_path = Path(args.config)

    # If config path is not absolute, look in configs directory
    if not config_path.is_absolute() and not config_path.exists():
        # First, we need to get topdir from a default location or environment
        # For now, we'll use a fallback approach
        default_topdir = Path.home()  # Use user's home directory as fallback

        # Try to find topdir from common locations or use default
        potential_topdirs = [
            Path("/Volumes/ARC_reborn"),
            Path("/Users/matthewhooton"),  # Your current setup
            Path.home(),
            Path.cwd()
        ]

        found_config = False
        for topdir_candidate in potential_topdirs:
            configs_dir = topdir_candidate / "bandersnatch_runs" / "configs"
            potential_config_path = configs_dir / config_path.name

            if potential_config_path.exists():
                config_path = potential_config_path
                found_config = True
                break

        if not found_config:
            # If not found in any configs directory, check current directory
            if not Path(args.config).exists():
                print(f"Error: Config file '{args.config}' not found in:")
                print(f"  - Current directory")
                for topdir_candidate in potential_topdirs:
                    configs_dir = topdir_candidate / "bandersnatch_runs" / "configs"
                    print(f"  - {configs_dir}")
                sys.exit(1)

    config = load_config(config_path)
    config = merge_config_with_args(config, args)

    # Load instrument configuration
    config_dir = Path(config_path).parent
    config['_config_dir'] = str(config_dir)
    instrument_config = load_instrument_config(config['instrument_settings']['inst'], config_dir)
    config['instrument_config'] = instrument_config

    # Extract configuration sections
    inst_settings = config['instrument_settings']
    proc_flags = config['processing_flags']
    calib_params = config['calibration_params']
    centroid_settings = config['centroid_settings']
    photometry_settings = config['photometry_settings']
    # path_settings = config['paths']

    # Set up paths with staging support
    topdir, rawdir, outdir = setup_paths(config, is_docker, args.local_staging, args.staging_dir)

    caldir = outdir / "calib"

    os.makedirs(outdir, exist_ok=True)

    # Set up logging (must be done after creating outdir)
    logger = setup_logging(config, outdir)

    # Log which config file was used
    logger.info("Using config file: %s", config_path)
    logger.info("Using instrument config: %s", instrument_config['instrument_name'])

    logger.info("=" * 60)
    logger.info(f"Starting astronomy pipeline for {inst_settings['inst']}")
    logger.info(f"Observation date: {inst_settings['date']}")
    logger.info(f"Run: {inst_settings['run']}")
    logger.info(f"Raw data directory: {rawdir}")
    logger.info(f"Output directory: {outdir}")
    logger.info("=" * 60)

    # Log processing flags
    enabled_steps = [step for step, enabled in proc_flags.items() if enabled]
    logger.info(f"Enabled processing steps: {', '.join(enabled_steps)}")

    # Create file lists
    if proc_flags['to_make_lists']:
        skip_existing = calib_params.get('skip_existing_lists', False)

        # Check if key list files already exist
        calib_dir = outdir / "calib"
        key_list_files = [
            calib_dir / f"{inst_settings['run']}_bias.list",
            calib_dir / f"{inst_settings['run']}_dark.list"
        ]
        lists_exist = all(list_file.exists() for list_file in key_list_files)

        if skip_existing and lists_exist:
            logger.info("Skipping file list creation - key list files already exist")
        else:
            logger.info("Creating file lists...")
            try:
                _ = create_lists(rawdir, outdir, inst_settings['run'],
                                 ext='fits',
                                 discard_n_first_science=inst_settings['discard_n_first_science'])
                logger.info("File lists created successfully")
            except Exception as e:
                logger.error(f"Failed to create file lists: {e}")
                raise

    # Make master bias
    if proc_flags['to_make_master_bias']:
        bias_file = outdir / 'calib' / f"{inst_settings['run']}_master_bias.fits"
        skip_existing = calib_params.get('skip_existing_bias', False)

        if skip_existing and bias_file.exists():
            logger.info("Skipping master bias creation - file already exists: %s", bias_file)
        else:
            logger.info("Creating master bias...")
            try:
                _ = make_master_calibration("bias", outdir, inst_settings['run'], inst_settings['inst'], config,
                                            image_extension=calib_params['image_extension'],
                                            clip=calib_params['bias_MAD_clip'],
                                            nlimit=calib_params['bias_MAD_nlimit'])
                logger.info("Master bias created successfully")
            except Exception as e:
                logger.error(f"Failed to create master bias: {e}")
                raise

    # Make master dark
    if proc_flags['to_make_master_dark']:
        dark_file = outdir / 'calib' / f"{inst_settings['run']}_master_dark.fits"
        skip_existing = calib_params.get('skip_existing_dark', False)

        if skip_existing and dark_file.exists():
            logger.info("Skipping master dark creation - file already exists: %s", dark_file)
        else:
            logger.info("Creating master dark...")
            try:
                _ = make_master_calibration("dark", outdir, inst_settings['run'], inst_settings['inst'], config,
                                            image_extension=calib_params['image_extension'],
                                            clip=calib_params['dark_MAD_clip'],
                                            nlimit=calib_params['dark_MAD_nlimit'])
                logger.info("Master dark created successfully")
            except Exception as e:
                logger.error(f"Failed to create master dark: {e}")
                raise

    # Make master flats - modify the loop to pass config
    if proc_flags['to_make_master_flat']:
        logger.info("Creating master flats...")
        try:
            filters = find_flat_filters(caldir, inst_settings['run'])
            logger.info(f"Found filters for flat creation: {filters}")

            # If no filters found, check master flats config for available filters
            if not filters:
                logger.warning("No flat field lists found for any filter")

                # Try to get filters from master flats config
                config_dir = Path(config_path).parent
                try:
                    with open(config_dir / "master_flats.yaml", 'r') as f:
                        master_flats_config = yaml.safe_load(f)
                    available_filters = list(master_flats_config.get('filters', {}).keys())

                    if available_filters:
                        logger.info("Found filters in master flats config: %s", available_filters)
                        skip_existing = calib_params.get('skip_existing_flat', False)

                        for filter in available_filters:
                            flat_file = outdir / 'calib' / f"{inst_settings['run']}_master_flat_{filter}.fits"

                            if skip_existing and flat_file.exists():
                                logger.info(
                                    f"Skipping master flat creation for filter {filter} - file already exists: %s",
                                    flat_file)
                            else:
                                logger.info(f"Processing filter {filter} from master flats config")
                                _ = make_master_calibration("flat", outdir, inst_settings['run'], inst_settings['inst'],
                                                            config,
                                                            filter=filter,
                                                            image_extension=calib_params['image_extension'],
                                                            clip=calib_params['flat_MAD_clip'],
                                                            nlimit=calib_params['flat_MAD_nlimit'])
                                logger.info(f"Master flat for filter {filter} processed successfully")
                    else:
                        raise ValueError("No filters found in flat lists or master flats config")
                except Exception as e:
                    logger.error("No flat field lists found and could not load master flats config: %s", e)
                    raise ValueError("No flat field files or master flats config found")
            else:
                # Process found filters normally
                for filter in filters:
                    # Process found filters normally
                    skip_existing = calib_params.get('skip_existing_flat', False)

                    for filter in filters:
                        flat_file = outdir / 'calib' / f"{inst_settings['run']}_master_flat_{filter}.fits"

                        if skip_existing and flat_file.exists():
                            logger.info(f"Skipping master flat creation for filter {filter} - file already exists: %s",
                                        flat_file)
                        else:
                            logger.info(f"Creating master flat for filter {filter}")
                            _ = make_master_calibration("flat", outdir, inst_settings['run'], inst_settings['inst'],
                                                        config,
                                                        filter=filter,
                                                        image_extension=calib_params['image_extension'],
                                                        clip=calib_params['flat_MAD_clip'],
                                                        nlimit=calib_params['flat_MAD_nlimit'])
                            logger.info(f"Master flat for filter {filter} created successfully")

            logger.info("All master flats created successfully")
        except Exception as e:
            logger.error(f"Failed to create master flats: {e}")
            raise

    # Process science images, centroiding, and photometry
    if (proc_flags['to_reduce_science_images'] or
            proc_flags['to_centroid'] or
            proc_flags['to_aperture_photometry']):

        try:
            all_targets = find_targets(caldir, inst_settings['run'])
            logger.info(f"Found targets in file system: {all_targets}")

            # Load targets configuration and resolve positions
            targets_config = None
            if proc_flags['to_centroid'] or proc_flags['to_aperture_photometry']:
                logger.info("Loading targets configuration...")
                config_dir = Path(config_path).parent
                targets_config = load_targets_config(config_dir)
                logger.info("Targets configuration loaded successfully")

                targets_to_process, initial_positions, tracking_stars = resolve_target_positions(config, targets_config,
                                                                                                 all_targets)
                logger.info("Target positions resolved successfully")
            else:
                # If not doing centroiding or photometry, process all targets
                targets_to_process = sorted(all_targets)
                initial_positions = None
                tracking_stars = None

            # Copy configuration files to log directory for record keeping
            copy_config_files(config_path, targets_config, targets_to_process, outdir, config)

            for i, target in enumerate(targets_to_process):
                logger.info(f"Processing target {i + 1}/{len(targets_to_process)}: {target}")

                # Check if we should use streaming processing
                use_streaming = (proc_flags['to_reduce_science_images'] and
                                 proc_flags['to_centroid'] and
                                 proc_flags['to_aperture_photometry'])

                if use_streaming:
                    logger.info(f"Using streaming processing for target {target}")

                    # Get filter from first image to load correct flat
                    list_file = outdir / "calib" / f"{inst_settings['run']}_image_{target}.list"
                    with open(list_file) as f:
                        first_filename = f.readline().strip()

                    with fits.open(first_filename) as hdul:
                        filter_name = hdul[0].header['FILTER']

                    # Load calibration frames
                    from streaming_processor import load_calibration_frames, process_images_streaming
                    calib_frames = load_calibration_frames(outdir, inst_settings['run'], filter_name, calib_params)

                    # Prepare parameters
                    initial_position = initial_positions[i]
                    tracking_star = tracking_stars[i]

                    centroid_params = {
                        'boxsize': centroid_settings['boxsize'],
                        'nlimit_centroid': centroid_settings['nlimit_centroid'],
                        'clip_centroid': centroid_settings['clip_centroid'],
                        'sky_sigma': centroid_settings['sky_sigma'],
                        'tracking_star': tracking_star,
                        'flux_above_value': centroid_settings['flux_above_value'],
                        'mask_centroid_pixels': centroid_settings['mask_centroid_pixels']
                    }

                    photometry_params = {
                        'aper_min': photometry_settings['aper_min'],
                        'aper_max': photometry_settings['aper_max'],
                        'SKYRAD_inner': photometry_settings['SKYRAD_inner'],
                        'SKYRAD_outer': photometry_settings['SKYRAD_outer'],
                        'sky_suppress': photometry_settings['sky_suppress']
                    }

                    try:
                        save_processed_images = config.get('save_processed_images_streaming', False)

                        successful_images, failed_images = process_images_streaming(
                            outdir, inst_settings['run'], target, config, calib_frames,
                            initial_position, centroid_params, photometry_params,
                            save_processed_images=save_processed_images
                        )
                        logger.info(f"Streaming processing completed for {target}: {successful_images} successful")
                        if failed_images:
                            logger.warning(f"Failed images for {target}: {failed_images}")
                    except Exception as e:
                        logger.error(f"Failed streaming processing for {target}: {e}")
                        raise

                else:
                    # Fall back to original processing for partial workflows
                    logger.info(f"Using traditional processing for target {target}")

                    # Reduce science frames
                    if proc_flags['to_reduce_science_images']:
                        logger.info(f"Performing reduction for target {target}")
                        try:
                            _ = reduce_science_frames(outdir, inst_settings['run'], target)
                            logger.info(f"Science frame reduction completed for {target}")
                        except Exception as e:
                            logger.error(f"Failed to reduce science frames for {target}: {e}")
                            raise

                    # Centroiding
                    if proc_flags['to_centroid']:
                        logger.info(f"Performing centroiding for target {target}")
                        initial_position = initial_positions[i]
                        tracking_star = tracking_stars[i]
                        logger.debug(f"Centroiding parameters: boxsize={centroid_settings['boxsize']}, "
                                     f"tracking_star={tracking_star}, flux_threshold={centroid_settings['flux_above_value']}")
                        try:
                            from centroid import centroid
                            centroid(outdir, inst_settings['run'], target, initial_position,
                                     centroid_settings['boxsize'],
                                     centroid_settings['nlimit_centroid'],
                                     centroid_settings['clip_centroid'],
                                     centroid_settings['sky_sigma'],
                                     tracking_star,
                                     centroid_settings['flux_above_value'],
                                     mask_centroid_pixels=centroid_settings['mask_centroid_pixels'])
                            logger.info(f"Centroiding completed for {target}")
                        except Exception as e:
                            logger.error(f"Failed to perform centroiding for {target}: {e}")
                            raise

                    # Aperture photometry
                    if proc_flags['to_aperture_photometry']:
                        logger.info(f"Performing aperture photometry for target {target}")
                        logger.debug(
                            f"Photometry parameters: apertures={photometry_settings['aper_min']}-{photometry_settings['aper_max']}, "
                            f"sky_annulus={photometry_settings['SKYRAD_inner']}-{photometry_settings['SKYRAD_outer']}")
                        try:
                            from photometry import photometry
                            photometry(outdir, inst_settings['run'], target,
                                       config,
                                       photometry_settings['aper_min'],
                                       photometry_settings['aper_max'],
                                       photometry_settings['SKYRAD_inner'],
                                       photometry_settings['SKYRAD_outer'],
                                       photometry_settings['sky_suppress'])
                            logger.info(f"Aperture photometry completed for {target}")
                        except Exception as e:
                            logger.error(f"Failed to perform aperture photometry for {target}: {e}")
                            raise

                logger.info(f"Completed processing for target {target}")

        except Exception as e:
            logger.error(f"Failed during target processing: {e}")
            raise
    else:
        # Even if not processing targets, copy the date config for record keeping
        try:
            copy_config_files(config_path, None, [], outdir, config)
        except Exception as e:
            logger.warning(f"Failed to copy date config file: {e}")

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()