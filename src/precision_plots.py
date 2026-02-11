#!/usr/bin/env python3
"""
Precision vs Brightness Plot Generator

This script automatically discovers all pipeline runs and generates photometric precision
vs stellar brightness plots for every target on every night. It searches through the
bandersnatch_runs directory structure and processes all targets with completed photometry.

Usage:
    python precision_plots.py [--topdir /path/to/topdir] [--config-dir /path/to/configs]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_topdir():
    """Find the topdir by searching common locations."""
    potential_topdirs = [
        Path("/Volumes/ARC_reborn"),
        Path("/Users/matthewhooton"),
        Path.home(),
        Path.cwd()
    ]

    for topdir_candidate in potential_topdirs:
        bandersnatch_dir = topdir_candidate / "bandersnatch_runs"
        if bandersnatch_dir.exists():
            logger.info(f"Found bandersnatch_runs directory: {bandersnatch_dir}")
            return topdir_candidate

    return None


def discover_pipeline_runs(topdir):
    """Discover all pipeline run directories with completed photometry."""
    bandersnatch_dir = topdir / "bandersnatch_runs"

    if not bandersnatch_dir.exists():
        logger.error(f"bandersnatch_runs directory not found in {topdir}")
        return []

    # Find all run directories (format: INST_YYYYMMDD)
    run_dirs = []
    for item in bandersnatch_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'configs':
            # Check if this looks like a run directory
            if '_' in item.name and len(item.name.split('_')[-1]) == 8:  # YYYYMMDD format
                run_dirs.append(item)

    logger.info(f"Found {len(run_dirs)} potential pipeline run directories")
    return sorted(run_dirs)


def discover_targets_in_run(run_dir):
    """Discover all targets with completed photometry in a run directory."""
    targets_found = []

    for item in run_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this is a target directory with photometry
            photometry_dir = item / "photometry"
            precision_metrics = photometry_dir / "precision_metrics.fits"

            if precision_metrics.exists():
                targets_found.append({
                    'target_name': item.name,
                    'target_dir': item,
                    'photometry_dir': photometry_dir,
                    'run_name': run_dir.name
                })

    return targets_found


def load_stellar_catalog(config_dir):
    """Load the ml_40pc.txt stellar catalog."""
    catalog_path = config_dir / "ml_40pc.txt"

    try:
        # Read the first line to get column names (comma-separated)
        with open(catalog_path, 'r') as f:
            header_line = f.readline().strip()

        # Parse column names from comma-separated header
        column_names = [col.strip() for col in header_line.split(',')]
        logger.debug(f"Parsed column names: {column_names}")

        # Read the data using whitespace separation, skipping the header line
        catalog = pd.read_csv(catalog_path, sep=r'\s+', skiprows=1, names=column_names)

        logger.info(f"Loaded stellar catalog with {len(catalog)} entries")
        logger.debug("Catalog columns: %s", list(catalog.columns))
        logger.debug("First row data types: %s", catalog.dtypes)

        # Check if required columns exist
        if 'T_eff' not in catalog.columns:
            logger.error("T_eff column not found. Available columns: %s", list(catalog.columns))
            raise ValueError("Required column T_eff not found in catalog")

        if 'Dis' not in catalog.columns:
            logger.error("Dis column not found. Available columns: %s", list(catalog.columns))
            raise ValueError("Required column Dis not found in catalog")

        logger.debug("Sample T_eff values: %s", catalog['T_eff'].head().tolist())
        logger.debug("Sample Dis values: %s", catalog['Dis'].head().tolist())

        return catalog
    except Exception as e:
        logger.error(f"Failed to load stellar catalog from {catalog_path}: {e}")
        raise


def find_target_parameters(target_name, catalog):
    """Find stellar parameters for a target in the catalog."""
    # First, try exact match
    mask = catalog['Sp_ID'].str.strip() == target_name.strip()
    matches = catalog[mask]

    if len(matches) == 0:
        # Try case-insensitive match
        mask = catalog['Sp_ID'].str.strip().str.upper() == target_name.strip().upper()
        matches = catalog[mask]

    if len(matches) == 0:
        # Try partial matching - look for targets that start with the same pattern
        # Convert Sp1255-5928 to Sp1255-59 pattern for matching
        target_base = target_name[:9] if len(target_name) >= 9 else target_name  # Sp1255-59
        mask = catalog['Sp_ID'].str.strip().str.startswith(target_base)
        matches = catalog[mask]

        if len(matches) > 0:
            logger.info(f"Found partial match for {target_name}: {matches.iloc[0]['Sp_ID']}")

    if len(matches) == 0:
        # Debug: show some example catalog entries for troubleshooting
        logger.warning(f"Target {target_name} not found in stellar catalog")
        logger.debug("First 5 catalog entries:")
        for i, sp_id in enumerate(catalog['Sp_ID'].head()):
            logger.debug(f"  {i + 1}: '{sp_id}'")
        return None, None

    if len(matches) > 1:
        logger.warning(f"Multiple matches for target {target_name}, using first: {matches.iloc[0]['Sp_ID']}")

    row = matches.iloc[0]

    # Debug the actual values being read
    logger.debug(f"Raw row data for {target_name}:")
    logger.debug(f"  T_eff column value: {row['T_eff']} (type: {type(row['T_eff'])})")
    logger.debug(f"  Dis column value: {row['Dis']} (type: {type(row['Dis'])})")

    try:
        teff = float(row['T_eff'])
        distance = float(row['Dis'])

        # Check for nan values
        if pd.isna(teff) or pd.isna(distance):
            logger.warning(f"NaN values found for {target_name}: Teff={teff}, Dis={distance}")
            return None, None

        logger.debug(f"Target {target_name}: Teff={teff}K, distance={distance}pc")
        return teff, distance

    except (ValueError, TypeError) as e:
        logger.error(f"Could not convert values to float for {target_name}: {e}")
        logger.error(f"  T_eff value: '{row['T_eff']}'")
        logger.error(f"  Dis value: '{row['Dis']}'")
        return None, None


def load_system_response(config_dir, filter_name, instrument_name):
    """Load and generate system response using mphot."""
    try:
        import mphot
    except ImportError:
        logger.error("mphot library not found. Please install with: pip install mphot")
        raise

    # Construct file paths
    filter_path = config_dir / f"{filter_name}.csv"
    instrument_path = config_dir / f"{instrument_name}.csv"

    if not filter_path.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_path}")
    if not instrument_path.exists():
        raise FileNotFoundError(f"Instrument file not found: {instrument_path}")

    logger.debug(f"Loading system response: filter={filter_name}, instrument={instrument_name}")

    # Generate system response
    name, system_response = mphot.generate_system_response(
        str(instrument_path), str(filter_path)
    )

    logger.debug(f"Generated system response: {name}")
    return name, system_response


def find_best_aperture(precision_metrics_path):
    """Find the aperture with the best (lowest) sigma_obs_10min for the target star."""
    try:
        precision_table = Table.read(precision_metrics_path, format='fits')

        # Filter for target star (star_number = 0)
        target_mask = precision_table['star_number'] == 0
        target_data = precision_table[target_mask]

        if len(target_data) == 0:
            logger.warning("No target star data found in precision metrics")
            return None, None

        # Find aperture with minimum sigma_obs_10min (excluding NaN values)
        valid_mask = ~np.isnan(target_data['sigma_obs_10min'])
        if np.sum(valid_mask) == 0:
            logger.warning("No valid sigma_obs_10min values found")
            return None, None

        valid_data = target_data[valid_mask]
        best_idx = np.argmin(valid_data['sigma_obs_10min'])
        best_row = valid_data[best_idx]

        best_aperture = best_row['aperture_radius']
        best_precision = best_row['sigma_obs_10min']

        logger.debug(f"Best aperture: {best_aperture} pixels (σ_obs = {best_precision:.6f})")
        return best_aperture, best_precision

    except Exception as e:
        logger.error(f"Failed to find best aperture: {e}")
        return None, None


def get_target_flux(photometry_dir, best_aperture):
    """Get the median flux for the target star from the best aperture."""
    try:
        aper_name = f"aper{int(best_aperture)}"
        photometry_path = photometry_dir / f"photometry_{aper_name}.fits"

        if not photometry_path.exists():
            logger.error(f"Photometry file not found: {photometry_path}")
            return None

        photometry_table = Table.read(photometry_path, format='fits')

        # Extract flux data for target star (star 0)
        flux_data = np.array(photometry_table['flux'])  # Shape: (n_images, n_stars)
        target_flux = flux_data[:, 0]  # Target star flux

        # Calculate median flux
        median_flux = np.median(target_flux)

        # Debug: check if this looks like electrons or ADU
        logger.debug(f"Target median flux: {median_flux:.1f}")
        logger.debug(f"Flux range: {np.min(target_flux):.1f} to {np.max(target_flux):.1f}")

        # Typical values for comparison:
        # - ADU: usually 1000-65000 for normal exposures
        # - Electrons: usually 10000-1000000+ for normal exposures (gain ~1-10)
        if median_flux < 100000:
            logger.warning(f"Flux value ({median_flux:.1f}) seems low for electrons - check if this is actually in ADU")

        return median_flux

    except Exception as e:
        logger.error(f"Failed to get target flux: {e}")
        return None


def calculate_exposure_time(photometry_dir):
    """Estimate exposure time from the first science image."""
    try:
        # Find any photometry table to get a file path
        photometry_files = list(photometry_dir.glob("photometry_aper*.fits"))
        if not photometry_files:
            logger.warning("No photometry files found for exposure time estimation")
            return 60.0  # Default fallback

        table = Table.read(photometry_files[0], format='fits')
        if len(table) == 0:
            return 60.0

        # Get first file path
        first_file = table['File'][0]

        # Try to read the header
        try:
            with fits.open(first_file) as hdul:
                exp_time = hdul[0].header.get('EXPTIME', 60.0)
                logger.debug(f"Detected exposure time: {exp_time}s")
                return float(exp_time)
        except:
            logger.warning(f"Could not read exposure time from {first_file}")
            return 60.0

    except Exception as e:
        logger.warning(f"Failed to calculate exposure time: {e}")
        return 60.0


def generate_precision_plot(target_info, teff, distance, target_flux, exp_time,
                            system_name, config_dir, observed_precision=None):
    """Generate precision vs brightness plot for a target."""
    try:
        import mphot
    except ImportError:
        logger.error("mphot library not found. Cannot generate precision plot")
        return False

    target_name = target_info['target_name']
    run_name = target_info['run_name']

    logger.debug(f"Generating precision plot for {target_name} ({run_name})")

    # Define atmospheric conditions (reasonable estimates)
    props_sky = {
        "pwv": 2.5,  # PWV [mm] - typical mid-latitude
        "airmass": 1.2,  # airmass - reasonable estimate
        "seeing": 1.5  # seeing [arcsec] - typical ground-based
    }

    # Define instrument properties (based on your config)
    props_instrument = {
        "name": system_name,
        "plate_scale": 0.35,  # From your configs
        "N_dc": 0.2,  # Dark current [e/pix/s] - estimate
        "N_rn": 6.0,  # Read noise [e_rms/pix] - estimate
        "well_depth": 64000,  # Well depth [e/pix] - typical
        "well_fill": 0.7,  # Fractional fill
        "read_time": 10.0,  # Read time [s] - estimate
        "r0": 0.5,  # Primary mirror radius [m] - estimate
        "r1": 0.14,  # Secondary mirror radius [m] - estimate
    }

    try:
        # Calculate precision for the observed target
        logger.debug(f"Calculating precision: Teff={teff}K, distance={distance}pc, exp_time={exp_time}s")

        # Validate inputs
        if not (100 < teff < 10000):
            logger.error(f"Invalid Teff value: {teff}K (should be 100-10000K)")
            return False
        if not (0.1 < distance < 1000):
            logger.error(f"Invalid distance value: {distance}pc (should be 0.1-1000pc)")
            return False
        if not (0.1 < target_flux < 1e9):
            logger.error(f"Invalid target flux: {target_flux} e/s")
            return False
        if not (0.1 < exp_time < 10000):
            logger.error(f"Invalid exposure time: {exp_time}s")
            return False

        image_precision, binned_precision, components = mphot.get_precision(
            props_instrument, props_sky, teff, distance,
            exp_time=exp_time, N_star=target_flux
        )

        # Debug what mphot returned
        logger.debug(
            f"mphot returned types: image_precision={type(image_precision)}, binned_precision={type(binned_precision)}, components={type(components)}")
        logger.debug(f"image_precision value: {image_precision}")
        logger.debug(f"binned_precision value: {binned_precision}")

        # Handle case where binned_precision might be a dict
        if isinstance(binned_precision, dict):
            logger.debug(f"binned_precision is a dict with keys: {list(binned_precision.keys())}")
            # We'll use the full dict for plotting individual components
            binned_prec_value = binned_precision['All']  # Just for validation
            precision_components = binned_precision
        else:
            binned_prec_value = binned_precision
            precision_components = {'All': binned_precision}

        logger.debug(f"mphot calculation successful: total_precision={binned_prec_value:.6f}")

        # Generate flux variation analysis
        logger.debug("Generating flux range for precision analysis")
        # Extend flux range to cover full x-axis range (10^2 to 10^8)
        # Use target flux as reference but ensure we cover the full range
        flux_min = 1e2  # 10^2 e/s
        flux_max = 1e8  # 10^8 e/s

        # Create logarithmic range covering the full axis
        flux_array_full = np.logspace(np.log10(flux_min), np.log10(flux_max), 100)

        # Store results for each component
        precision_results = {}
        flux_array = []

        for i, N_star in enumerate(flux_array_full):
            try:
                image_prec, binned_prec, comp = mphot.get_precision(
                    props_instrument, props_sky, teff, distance,
                    exp_time=exp_time, N_star=N_star
                )

                # Handle case where binned_prec is a dict with components
                if isinstance(binned_prec, dict):
                    # First iteration - initialize the component arrays
                    if len(precision_results) == 0:
                        for component_name in binned_prec.keys():
                            precision_results[component_name] = []

                    # Store each component
                    for component_name, component_value in binned_prec.items():
                        precision_results[component_name].append(component_value)
                else:
                    # Fallback for simple values
                    if 'All' not in precision_results:
                        precision_results['All'] = []
                    precision_results['All'].append(binned_prec)

                flux_array.append(comp['N_star [e/s]'])

            except Exception as e:
                logger.warning(f"Failed to calculate precision for flux {N_star}: {e}")
                continue

        if len(precision_results) == 0 or len(flux_array) == 0:
            logger.error("No valid precision calculations in flux sweep")
            return False

        logger.debug(
            f"Completed flux sweep: {len(flux_array)} valid points, components: {list(precision_results.keys())}")

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colors and line styles for different components
        component_styles = {
            'All': {'color': 'blue', 'linestyle': '-', 'linewidth': 2},
            'Star': {'color': 'orange', 'linestyle': '-', 'linewidth': 1.5},
            'Scintillation': {'color': 'green', 'linestyle': '-', 'linewidth': 1.5},
            'Sky': {'color': 'red', 'linestyle': '-', 'linewidth': 1.5},
            'Dark current': {'color': 'purple', 'linestyle': '-', 'linewidth': 1.5},
            'Read noise': {'color': 'brown', 'linestyle': '-', 'linewidth': 1.5}
        }

        # Plot each precision component
        for component_name, precision_values in precision_results.items():
            # Convert precision to ppt (parts per thousand)
            precision_ppt = np.array(precision_values) * 1000

            style = component_styles.get(component_name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1})

            ax.loglog(flux_array, precision_ppt,
                      label=component_name,
                      color=style['color'],
                      linestyle=style['linestyle'],
                      linewidth=style['linewidth'])

        # Add vertical line for target star
        ax.axvline(target_flux, linestyle='--', color='black', alpha=0.7,
                   linewidth=2, label=f'Target: {target_flux:.0f} e/s')

        # Add horizontal line for observed precision if available
        if observed_precision is not None and not np.isnan(observed_precision):
            obs_prec_ppt = observed_precision * 1000  # Convert to ppt
            ax.axhline(obs_prec_ppt, linestyle=':', color='black', alpha=0.7,
                       linewidth=2, label=f'Observed: {obs_prec_ppt:.2f} ppt')

        # Set plot limits and formatting
        ax.set_xlim(1e2, 1e8)  # 10^2 to 10^8 e/s
        ax.set_ylim(1e-2, 1e3)  # 10^-2 to 10^3 ppt
        ax.minorticks_on()
        ax.grid(True, alpha=0.3)

        # Labels and title - handle components dict safely
        if isinstance(components, dict) and 'binning [mins]' in components:
            binning_time = components['binning [mins]']
        else:
            binning_time = 10  # Default fallback

        ax.set_xlabel('Stellar Flux [e/s]')
        ax.set_ylabel(f'{binning_time:.0f} min binned precision [ppt]')
        ax.set_title(f'Precision vs Brightness - {target_name} ({run_name})\n'
                     f'Teff={teff:.0f}K, d={distance:.1f}pc, exp={exp_time:.0f}s')

        # Update legend
        ax.legend(loc='best')

        # Create plots directory and save
        plots_dir = target_info['photometry_dir'] / "plots"
        plots_dir.mkdir(exist_ok=True)

        output_path = plots_dir / f"precision_vs_brightness_{target_name}_{run_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.debug(f"Precision plot saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate precision plot: {e}")
        # Add more detailed error information
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False


def process_target(target_info, stellar_catalog, system_name, config_dir):
    """Process a single target."""
    target_name = target_info['target_name']
    run_name = target_info['run_name']
    photometry_dir = target_info['photometry_dir']

    logger.debug(f"Processing target: {target_name} ({run_name})")

    # Find stellar parameters
    teff, distance = find_target_parameters(target_name, stellar_catalog)
    if teff is None or distance is None:
        logger.warning(f"Skipping {target_name} ({run_name}): stellar parameters not found")
        return False

    # Check for required files
    precision_metrics_path = photometry_dir / "precision_metrics.fits"

    # Find best aperture
    best_aperture, observed_precision = find_best_aperture(precision_metrics_path)
    if best_aperture is None:
        logger.warning(f"Skipping {target_name} ({run_name}): could not determine best aperture")
        return False

    # Get target flux
    target_flux = get_target_flux(photometry_dir, best_aperture)
    if target_flux is None:
        logger.warning(f"Skipping {target_name} ({run_name}): could not get target flux")
        return False

    # Calculate exposure time
    exp_time = calculate_exposure_time(photometry_dir)

    # Generate precision plot
    success = generate_precision_plot(
        target_info, teff, distance, target_flux, exp_time,
        system_name, config_dir, observed_precision
    )

    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate precision vs brightness plots for all pipeline targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--topdir',
                        help='Path to top directory containing bandersnatch_runs (default: auto-detect)')
    parser.add_argument('--config-dir',
                        help='Path to configuration directory (default: topdir/bandersnatch_runs/configs)')
    parser.add_argument('--filter-name', default='zYJ',
                        help='Filter name for system response')
    parser.add_argument('--instrument-name', default='SPIRIT2',
                        help='Instrument name for system response')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without generating plots')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find topdir
    if args.topdir:
        topdir = Path(args.topdir)
        if not topdir.exists():
            logger.error(f"Specified topdir does not exist: {topdir}")
            return 1
    else:
        topdir = find_topdir()
        if topdir is None:
            logger.error("Could not auto-detect topdir. Please specify with --topdir")
            return 1

    logger.info(f"Using topdir: {topdir}")

    # Find config directory
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = topdir / "bandersnatch_runs" / "configs"

    if not config_dir.exists():
        logger.error(f"Config directory does not exist: {config_dir}")
        return 1

    logger.info(f"Using config directory: {config_dir}")

    # Discover all pipeline runs
    logger.info("Discovering pipeline runs...")
    run_dirs = discover_pipeline_runs(topdir)

    if not run_dirs:
        logger.error(f"No pipeline run directories found in {topdir}/bandersnatch_runs")
        return 1

    # Discover all targets across all runs
    logger.info("Discovering targets with completed photometry...")
    all_targets = []

    for run_dir in run_dirs:
        targets_in_run = discover_targets_in_run(run_dir)
        all_targets.extend(targets_in_run)
        if targets_in_run:
            logger.info(f"Found {len(targets_in_run)} targets in {run_dir.name}")

    if not all_targets:
        logger.error("No targets with completed photometry found")
        return 1

    logger.info(f"Total targets found: {len(all_targets)}")

    # Show summary of what will be processed
    target_summary = {}
    for target_info in all_targets:
        target_name = target_info['target_name']
        run_name = target_info['run_name']

        if target_name not in target_summary:
            target_summary[target_name] = []
        target_summary[target_name].append(run_name)

    logger.info(f"Targets by name:")
    for target_name, runs in sorted(target_summary.items()):
        logger.info(f"  {target_name}: {len(runs)} runs ({', '.join(sorted(runs))})")

    if args.dry_run:
        logger.info("Dry run complete - no plots generated")
        return 0

    # Load required data for processing
    logger.info("Loading stellar catalog...")
    try:
        stellar_catalog = load_stellar_catalog(config_dir)
    except Exception as e:
        logger.error(f"Failed to load stellar catalog: {e}")
        return 1

    logger.info("Loading system response...")
    try:
        system_name, system_response = load_system_response(
            config_dir, args.filter_name, args.instrument_name
        )
    except Exception as e:
        logger.error(f"Failed to load system response: {e}")
        return 1

    # Process all targets
    logger.info("Processing targets...")
    successful = 0
    failed = 0

    for i, target_info in enumerate(all_targets, 1):
        target_name = target_info['target_name']
        run_name = target_info['run_name']

        logger.info(f"Processing {i}/{len(all_targets)}: {target_name} ({run_name})")

        try:
            if process_target(target_info, stellar_catalog, system_name, config_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {target_name} ({run_name}): {e}")
            failed += 1

    logger.info(f"Processing complete: {successful} successful, {failed} failed")

    if successful == 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())