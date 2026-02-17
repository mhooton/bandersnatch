#!/usr/bin/env python3
"""
Standalone differential light curve plotter with extensive user control.

Allows customization of:
- Aperture radius
- Time binning
- X/Y axis limits
- Plot appearance (colors, markers, labels)
- Data filtering
- Multiple plot types (flux, residuals, O-C)

Usage:
    python plot_differential_lc.py /path/to/topdir RUN_NAME TARGET_NAME APERTURE [options]

Example:
    python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6 \
        --time-bin-size 0.002 \
        --xlim 0 0.2 \
        --ylim 0.98 1.02 \
        --output my_transit.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def load_photometry_data(topdir, run_name, target_name, aperture):
    """
    Load differential photometry data from FITS file.

    Parameters
    ----------
    topdir : str
        Base directory path
    run_name : str
        Run name (e.g., 'SPIRIT_20251215')
    target_name : str
        Target name (e.g., 'TOI-696c')
    aperture : int
        Aperture radius in pixels

    Returns
    -------
    astropy.table.Table
        Photometry table with BJD, diff_flux, diff_flux_err, etc.
    """
    topdir = Path(topdir).expanduser()
    photometry_file = topdir / "bandersnatch_runs" / run_name / target_name / "photometry" / f"photometry_aper{aperture}.fits"

    if not photometry_file.exists():
        raise FileNotFoundError(f"Photometry file not found: {photometry_file}")

    with fits.open(photometry_file) as hdul:
        data = Table(hdul[1].data)

    return data


def apply_filters(data, bjd_min=None, bjd_max=None, max_diff_flux_err=None, exclude_outliers=True):
    """
    Apply filters to photometry data.

    Parameters
    ----------
    data : astropy.table.Table
        Photometry data
    bjd_min : float, optional
        Minimum BJD to include
    bjd_max : float, optional
        Maximum BJD to include
    max_diff_flux_err : float, optional
        Maximum differential flux error to include
    exclude_outliers : bool
        Exclude points flagged as outliers (outlier_flag != 0)

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask of data to include
    """
    mask = np.ones(len(data), dtype=bool)

    # BJD range filter
    if bjd_min is not None:
        mask &= data['BJD'] >= bjd_min
    if bjd_max is not None:
        mask &= data['BJD'] <= bjd_max

    # Error threshold filter
    if max_diff_flux_err is not None:
        mask &= data['diff_flux_err'] <= max_diff_flux_err

    # Outlier filter
    if exclude_outliers:
        mask &= data['outlier_flag'] == 0

    return mask


def bin_data(x, y, yerr, bin_size):
    """
    Bin data by x-coordinate.

    Parameters
    ----------
    x : array-like
        X coordinates (e.g., phase or time)
    y : array-like
        Y values (e.g., flux)
    yerr : array-like
        Y errors
    bin_size : float
        Bin size in x units

    Returns
    -------
    x_binned : numpy.ndarray
        Bin centers
    y_binned : numpy.ndarray
        Binned y values (weighted mean)
    yerr_binned : numpy.ndarray
        Binned y errors (propagated)
    n_binned : numpy.ndarray
        Number of points in each bin
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_sorted = np.array(y)[sort_idx]
    yerr_sorted = np.array(yerr)[sort_idx]

    # Define bin edges
    x_min = x_sorted.min()
    x_max = x_sorted.max()
    bin_edges = np.arange(x_min, x_max + bin_size, bin_size)

    x_binned = []
    y_binned = []
    yerr_binned = []
    n_binned = []

    for i in range(len(bin_edges) - 1):
        # Find points in this bin
        in_bin = (x_sorted >= bin_edges[i]) & (x_sorted < bin_edges[i + 1])

        if np.sum(in_bin) == 0:
            continue

        x_bin = x_sorted[in_bin]
        y_bin = y_sorted[in_bin]
        yerr_bin = yerr_sorted[in_bin]

        # Weighted mean
        weights = 1.0 / yerr_bin ** 2
        y_mean = np.sum(y_bin * weights) / np.sum(weights)
        yerr_mean = 1.0 / np.sqrt(np.sum(weights))

        x_binned.append(np.mean(x_bin))
        y_binned.append(y_mean)
        yerr_binned.append(yerr_mean)
        n_binned.append(len(x_bin))

    return (np.array(x_binned), np.array(y_binned),
            np.array(yerr_binned), np.array(n_binned))


def plot_light_curve(ax, x, y, yerr,
                     xlabel='Phase', ylabel='Differential Flux',
                     xlim=None, ylim=None,
                     color='black', marker='o', markersize=3, alpha=0.7,
                     show_errors=True, show_grid=True):
    """
    Plot light curve on given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : array-like
        X coordinates
    y : array-like
        Y values
    yerr : array-like
        Y errors
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    xlim : tuple, optional
        (xmin, xmax) axis limits
    ylim : tuple, optional
        (ymin, ymax) axis limits
    color : str
        Point color
    marker : str
        Marker style
    markersize : float
        Marker size
    alpha : float
        Point transparency
    show_errors : bool
        Show error bars
    show_grid : bool
        Show grid lines
    """
    if show_errors:
        ax.errorbar(x, y, yerr=yerr,
                    fmt=marker, color=color, markersize=markersize,
                    alpha=alpha, elinewidth=0.5, capsize=0)
    else:
        ax.plot(x, y, marker=marker, color=color, markersize=markersize,
                alpha=alpha, linestyle='none')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show_grid:
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    ax.tick_params(labelsize=10)


def main():
    parser = argparse.ArgumentParser(
        description='Plot differential light curves with extensive customization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plot with aperture 6
  python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6

  # With time binning
  python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6 \\
      --time-bin-size 0.002 \\
      --xlim 0 0.2

  # Custom appearance
  python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6 \\
      --color blue --markersize 5 --alpha 0.5 \\
      --no-errors --no-grid

  # Filter data
  python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6 \\
      --bjd-min 2459123.4 --bjd-max 2459123.6 \\
      --max-diff-flux-err 0.005

  # Save high-res figure
  python plot_differential_lc.py ~/ SPIRIT_20251215 TOI-696c 6 \\
      --output transit.png --dpi 300 --figsize 12 8
        """
    )

    # Required arguments
    parser.add_argument('topdir', help='Base directory path (e.g., ~/)')
    parser.add_argument('run_name', help='Run name (e.g., SPIRIT_20251215)')
    parser.add_argument('target_name', help='Target name (e.g., TOI-696c)')
    parser.add_argument('aperture', type=int, help='Aperture radius in pixels')

    # Time binning option
    parser.add_argument('--time-bin-size', type=float,
                        help='Time bin size in days (e.g., 0.002 = ~3 minutes)')

    # Time axis options
    time_group = parser.add_argument_group('time axis')
    time_group.add_argument('--time-format', choices=['relative', 'bjd'], default='relative',
                            help='Time axis format: relative (days from start) or bjd (full BJD)')
    time_group.add_argument('--t0', type=float,
                            help='Reference time for relative time axis (BJD). If not provided, uses first observation')

    # Axis limits
    axis_group = parser.add_argument_group('axis limits')
    axis_group.add_argument('--xlim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                            help='X-axis limits')
    axis_group.add_argument('--ylim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                            help='Y-axis limits')

    # Data filtering
    filter_group = parser.add_argument_group('data filtering')
    filter_group.add_argument('--bjd-min', type=float, help='Minimum BJD to include')
    filter_group.add_argument('--bjd-max', type=float, help='Maximum BJD to include')
    filter_group.add_argument('--max-diff-flux-err', type=float,
                              help='Maximum differential flux error to include')
    filter_group.add_argument('--include-outliers', action='store_true',
                              help='Include points flagged as outliers')

    # Appearance options
    appearance_group = parser.add_argument_group('appearance')
    appearance_group.add_argument('--color', default='tab:blue', help='Point color')
    appearance_group.add_argument('--bin-color', default='black', help='Binned point edge color')
    appearance_group.add_argument('--marker', default='o', help='Marker style')
    appearance_group.add_argument('--markersize', type=float, default=3, help='Marker size')
    appearance_group.add_argument('--bin-markersize', type=float, default=8,
                                  help='Binned point marker size')
    appearance_group.add_argument('--alpha', type=float, default=0.2,
                                  help='Point transparency (0-1)')
    appearance_group.add_argument('--bin-alpha', type=float, default=1.0,
                                  help='Binned point transparency (0-1)')
    appearance_group.add_argument('--no-errors', action='store_true',
                                  help='Hide error bars')
    appearance_group.add_argument('--no-grid', action='store_true',
                                  help='Hide grid lines')

    # Labels
    label_group = parser.add_argument_group('labels')
    label_group.add_argument('--xlabel', help='Custom X-axis label')
    label_group.add_argument('--ylabel', default='Differential Flux',
                             help='Custom Y-axis label')
    label_group.add_argument('--title', help='Plot title')

    # Output options
    output_group = parser.add_argument_group('output')
    output_group.add_argument('--output', '-o', help='Output filename (default: show plot)')
    output_group.add_argument('--dpi', type=int, default=150, help='Output DPI')
    output_group.add_argument('--figsize', nargs=2, type=float, default=[10, 6],
                              metavar=('WIDTH', 'HEIGHT'), help='Figure size in inches')

    # Additional plot types
    plot_group = parser.add_argument_group('additional plots')
    plot_group.add_argument('--show-residuals', action='store_true',
                            help='Show residuals panel (requires --model-file)')
    plot_group.add_argument('--model-file', help='Path to model light curve file (2 columns: time, flux)')

    args = parser.parse_args()

    # Validate arguments
    if args.show_residuals and args.model_file is None:
        parser.error("--show-residuals requires --model-file")

    # Load data
    print(f"Loading photometry data for {args.target_name}, aperture {args.aperture}...")
    try:
        data = load_photometry_data(args.topdir, args.run_name, args.target_name, args.aperture)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} data points")

    # Apply filters
    mask = apply_filters(data,
                         bjd_min=args.bjd_min,
                         bjd_max=args.bjd_max,
                         max_diff_flux_err=args.max_diff_flux_err,
                         exclude_outliers=not args.include_outliers)

    data_filtered = data[mask]
    print(f"After filtering: {len(data_filtered)} data points")

    if len(data_filtered) == 0:
        print("Error: No data points remain after filtering!", file=sys.stderr)
        sys.exit(1)

    # Extract data
    bjd = np.array(data_filtered['BJD'])
    diff_flux = np.array(data_filtered['diff_flux'])
    diff_flux_err = np.array(data_filtered['diff_flux_err'])

    # Compute x-axis values (time)
    if args.time_format == 'bjd':
        x = bjd
        default_xlabel = 'BJD'
        print(f"Plotting vs BJD")
    else:
        # Relative time
        t_ref = args.t0 if args.t0 is not None else bjd.min()
        x = bjd - t_ref
        default_xlabel = f'Time since BJD {t_ref:.3f} (days)'
        print(f"Plotting vs time relative to BJD {t_ref:.3f}")

    xlabel = args.xlabel if args.xlabel is not None else default_xlabel

    # Create figure
    if args.show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=args.figsize,
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=args.figsize)

    # Plot unbinned data
    plot_light_curve(ax1, x, diff_flux, diff_flux_err,
                     xlabel=xlabel if not args.show_residuals else '',
                     ylabel=args.ylabel,
                     xlim=args.xlim, ylim=args.ylim,
                     color=args.color, marker=args.marker,
                     markersize=args.markersize, alpha=args.alpha,
                     show_errors=not args.no_errors,
                     show_grid=not args.no_grid)

    # Plot binned data if requested
    if args.time_bin_size is not None:
        bin_size = args.time_bin_size

        x_binned, y_binned, yerr_binned, n_binned = bin_data(x, diff_flux, diff_flux_err, bin_size)

        print(f"Binned into {len(x_binned)} bins (bin size = {bin_size} days)")
        print(f"Mean points per bin: {n_binned.mean():.1f}")

        ax1.errorbar(x_binned, y_binned, yerr=yerr_binned,
                     fmt='o', markerfacecolor='white', markeredgecolor=args.bin_color,
                     markeredgewidth=1.5, markersize=args.bin_markersize,
                     alpha=args.bin_alpha, elinewidth=3, capsize=0,
                     ecolor=args.bin_color,
                     label=f'Binned ({bin_size} d)', zorder=10)

        ax1.legend(loc='best', fontsize=10)

    # Load and plot model if provided
    if args.model_file is not None:
        try:
            model_data = np.loadtxt(args.model_file)
            model_time = model_data[:, 0]
            model_flux = model_data[:, 1]

            ax1.plot(model_time, model_flux, 'r-', linewidth=2,
                     label='Model', zorder=5, alpha=0.8)
            ax1.legend(loc='best', fontsize=10)
            print(f"Loaded model from {args.model_file}")
        except Exception as e:
            print(f"Warning: Could not load model file: {e}", file=sys.stderr)

    # Add title
    if args.title:
        ax1.set_title(args.title, fontsize=14)
    else:
        default_title = f"{args.target_name} - Aperture {args.aperture} pixels"
        ax1.set_title(default_title, fontsize=14)

    # Plot residuals if requested
    if args.show_residuals:
        if args.model_file is not None:
            # Interpolate model to data points
            model_interp = np.interp(x, model_time, model_flux)
            residuals = diff_flux - model_interp

            plot_light_curve(ax2, x, residuals, diff_flux_err,
                             xlabel=xlabel, ylabel='O-C',
                             xlim=args.xlim, ylim=None,
                             color=args.color, marker=args.marker,
                             markersize=args.markersize, alpha=args.alpha,
                             show_errors=not args.no_errors,
                             show_grid=not args.no_grid)

            # Add zero line
            ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

            # Share x-axis
            ax1.set_xticklabels([])

            # Calculate RMS
            rms = np.std(residuals)
            ax2.text(0.02, 0.98, f'RMS = {rms:.6f}',
                     transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            print("Warning: --show-residuals requires --model-file", file=sys.stderr)

    plt.tight_layout()

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()