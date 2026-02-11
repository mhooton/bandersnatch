#!/usr/bin/env python3
"""
Convert photometry pipeline outputs to juliet-compatible lc.dat format.
"""

import numpy as np
from astropy.io import fits
import os
import sys


def normalize_to_range(data, min_val=-1, max_val=1):
    """Normalize data to specified range [min_val, max_val]"""
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max == data_min:
        # Handle case where all values are the same
        return np.full_like(data, 0.0)

    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val


def load_photometry_data(photometry_file):
    """Load photometry data from FITS file"""
    with fits.open(photometry_file) as hdul:
        data = hdul[1].data  # Data is in extension 1 (BinTableHDU)

    bjd = data['BJD']
    diff_flux = data['diff_flux']
    diff_flux_err = data['diff_flux_err']
    sky = data['SKY'][:, 0]  # Extract star 0

    return bjd, diff_flux, diff_flux_err, sky


def load_centroids_data(centroids_file):
    """Load centroids data from FITS file"""
    with fits.open(centroids_file) as hdul:
        data = hdul[1].data  # Data is in extension 1 (BinTableHDU)

    bjd = data['BJD']
    xc = data['xc'][:, 0]  # Extract star 0
    yc = data['yc'][:, 0]  # Extract star 0
    x_width = data['x_width'][:, 0]  # Extract star 0
    y_width = data['y_width'][:, 0]  # Extract star 0
    airmass = data['airmass']  # airmass is in centroids file

    return bjd, xc, yc, x_width, y_width, airmass


def match_bjd_values(phot_bjd, phot_data, cent_bjd, cent_data):
    """Match photometry and centroids data based on BJD values"""
    # Find common BJD values
    common_mask = np.isin(phot_bjd, cent_bjd)

    if not np.any(common_mask):
        raise ValueError("No matching BJD values found between photometry and centroids files")

    # Filter photometry data
    matched_phot_bjd = phot_bjd[common_mask]
    matched_phot_data = [data[common_mask] for data in phot_data]

    # Find indices in centroids data that match
    cent_indices = []
    for bjd_val in matched_phot_bjd:
        idx = np.where(cent_bjd == bjd_val)[0]
        if len(idx) > 0:
            cent_indices.append(idx[0])
        else:
            raise ValueError(f"BJD value {bjd_val} not found in centroids data")

    cent_indices = np.array(cent_indices)
    matched_cent_data = [data[cent_indices] for data in cent_data]

    return matched_phot_bjd, matched_phot_data, matched_cent_data


def create_juliet_file(top_dir, run_name, target_name, aperture_number, instrument_name, output_file="lc.dat"):
    """Create juliet-compatible lc.dat file from pipeline outputs"""

    # Construct file paths
    photometry_file = os.path.join(top_dir, "bandersnatch_runs", run_name, target_name,
                                   "photometry", f"photometry_aper{aperture_number}.fits")
    centroids_file = os.path.join(top_dir, "bandersnatch_runs", run_name, target_name,
                                  "centroiding", "centroids.fits")

    # Check if files exist
    if not os.path.exists(photometry_file):
        raise FileNotFoundError(f"Photometry file not found: {photometry_file}")
    if not os.path.exists(centroids_file):
        raise FileNotFoundError(f"Centroids file not found: {centroids_file}")

    print(f"Loading photometry data from: {photometry_file}")
    # Load photometry data
    phot_bjd, diff_flux, diff_flux_err, sky = load_photometry_data(photometry_file)

    print(f"Loading centroids data from: {centroids_file}")
    # Load centroids data
    cent_bjd, xc, yc, x_width, y_width, airmass = load_centroids_data(centroids_file)

    print("Matching BJD values between datasets...")
    # Match data based on BJD
    matched_bjd, matched_phot_data, matched_cent_data = match_bjd_values(
        phot_bjd, [diff_flux, diff_flux_err, sky],
        cent_bjd, [xc, yc, x_width, y_width, airmass]
    )

    matched_diff_flux, matched_diff_flux_err, matched_sky = matched_phot_data
    matched_xc, matched_yc, matched_x_width, matched_y_width, matched_airmass = matched_cent_data

    print(f"Found {len(matched_bjd)} matching data points")

    # Normalize theta columns to [-1, 1]
    print("Normalizing theta columns...")
    norm_sky = normalize_to_range(matched_sky)
    norm_xc = normalize_to_range(matched_xc)
    norm_yc = normalize_to_range(matched_yc)
    norm_x_width = normalize_to_range(matched_x_width)
    norm_y_width = normalize_to_range(matched_y_width)
    norm_airmass = normalize_to_range(matched_airmass)

    # Create instrument name column
    instrument_col = np.full(len(matched_bjd), instrument_name, dtype='U20')

    # Ensure BJD is float (in case it was read as string)
    matched_bjd_float = np.array(matched_bjd, dtype=float)

    # Save to file - handle mixed data types properly
    print(f"Saving to {output_file}...")

    # Create format string for each row
    with open(output_file, 'w') as f:
        for i in range(len(matched_bjd_float)):
            f.write(f"{matched_bjd_float[i]:.8f} "
                    f"{matched_diff_flux[i]:.8e} "
                    f"{matched_diff_flux_err[i]:.8e} "
                    f"{instrument_col[i]} "
                    f"{norm_sky[i]:.6f} "
                    f"{norm_xc[i]:.6f} "
                    f"{norm_yc[i]:.6f} "
                    f"{norm_x_width[i]:.6f} "
                    f"{norm_y_width[i]:.6f} "
                    f"{norm_airmass[i]:.6f}\n")

    print(f"Successfully created {output_file} with {len(matched_bjd)} data points")
    return output_file


def main():
    """Main function with command-line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert pipeline outputs to juliet-compatible lc.dat format")
    parser.add_argument("top_dir", help="Top directory path")
    parser.add_argument("run_name", help="Run name")
    parser.add_argument("target_name", help="Target name")
    parser.add_argument("aperture_number", help="Aperture number")
    parser.add_argument("instrument_name", help="Instrument name")
    parser.add_argument("-o", "--output", default="lc.dat", help="Output filename (default: lc.dat)")

    args = parser.parse_args()

    print("Pipeline to Juliet Converter")
    print("=" * 30)
    print(f"Top directory: {args.top_dir}")
    print(f"Run name: {args.run_name}")
    print(f"Target name: {args.target_name}")
    print(f"Aperture number: {args.aperture_number}")
    print(f"Instrument name: {args.instrument_name}")
    print(f"Output file: {args.output}")
    print()

    try:
        output_file = create_juliet_file(args.top_dir, args.run_name, args.target_name,
                                         args.aperture_number, args.instrument_name, args.output)
        print(f"\nConversion completed successfully!")
        print(f"Output file: {output_file}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()