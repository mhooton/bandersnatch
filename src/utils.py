import numpy as np
import os
import re
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def mad(array, sigma=False, med=None):
    """
    Calculate the Median Absolute Deviation of an array.

    Parameters:
    -----------
    array : array-like
        Input array
    sigma : bool, optional
        If True, scale MAD to approximate standard deviation
    med : float, optional
        If provided, use this as the median value instead of calculating it

    Returns:
    --------
    float
        Median Absolute Deviation value
    """
    if med is None:
        med = np.median(array)

    mad_value = np.median(np.abs(array - med))

    if sigma:
        # Scale factor to approximate standard deviation (assuming normal distribution)
        # The exact value is 1/0.6744897501960817 ≈ 1.4826
        mad_value = mad_value * 1.5  # Rounded value as in original code

    logger.debug("MAD calculation: median=%.3f, mad=%.3f, sigma_scaled=%s",
                 med, mad_value, sigma)

    return mad_value


def medabsdevclip(data, clip, nlimit):
    """
    Calculate clipped statistics based on median absolute deviation.

    Parameters:
    -----------
    data : array-like
        Input data (2D array or 3D cube)
    clip : float
        Number of MADs to clip at
    nlimit : int
        Limit for number of values

    Returns:
    --------
    For 2D input: array with [mean, stddev] of clipped data
    For 3D input: 2D array of clipped means across the third dimension
    """
    data_shape = np.shape(data)
    data_ndim = len(data_shape)

    logger.debug("medabsdevclip: input shape=%s, clip=%.1f, nlimit=%d",
                 data_shape, clip, nlimit)

    if data_ndim == 2:
        # 2D array
        if not isinstance(clip, (int, float)):
            clip = 5
            logger.warning("Invalid clip parameter, using default: %.1f", clip)
        if not isinstance(nlimit, (int, float)):
            nlimit = 50000
            logger.warning("Invalid nlimit parameter, using default: %d", nlimit)

        nx, ny = data_shape

        # Remove NaN values first
        flat_data = data.flatten()
        valid_mask = ~np.isnan(flat_data)
        valid_data = flat_data[valid_mask]

        logger.debug("2D processing: total_pixels=%d, valid_pixels=%d, nlimit=%d",
                     len(flat_data), len(valid_data), nlimit)

        if len(valid_data) == 0:
            logger.warning("No valid data found in 2D array")
            return np.array([np.nan, np.nan])

        if nlimit >= 0 and len(valid_data) >= 10 * nlimit:
            # Sort the valid data
            s1 = np.argsort(valid_data)
            trimmed_data = valid_data[s1[:len(valid_data) - nlimit - 1]]
            sig, med = mad(trimmed_data, sigma=True), np.median(trimmed_data)
            logger.debug("Using trimmed data: %d points (removed %d)",
                         len(trimmed_data), nlimit)
        else:
            sig, med = mad(valid_data, sigma=True), np.median(valid_data)
            logger.debug("Using all valid data: %d points", len(valid_data))

        logger.debug("Initial statistics: median=%.3f, sigma=%.3f", med, sig)

        # Handle division by zero
        if sig == 0 or np.isnan(sig):
            logger.warning("Sigma is zero or NaN, using fallback value")
            sig = 1e-10

        # Find values within clip limit
        w_g = np.abs(valid_data - med) / sig <= clip
        clipped_data = valid_data[w_g]

        logger.debug("Clipping results: %d points remain (from %d, %.1f%% kept)",
                     len(clipped_data), len(valid_data),
                     100 * len(clipped_data) / len(valid_data) if len(valid_data) > 0 else 0)

        output = np.zeros(2)

        if len(clipped_data) > 0:
            # Calculate mean and standard deviation of good values
            output[0] = np.mean(clipped_data)
            output[1] = np.std(clipped_data)
        else:
            logger.warning("No points survived clipping, using median")
            output[0] = med
            output[1] = 0.0

        logger.debug("Final 2D result: mean=%.3f, std=%.3f", output[0], output[1])

        return output

    elif data_ndim == 3:
        # 3D cube
        if not isinstance(clip, (int, float)):
            clip = 5
            logger.warning("Invalid clip parameter, using default: %.1f", clip)
        if not isinstance(nlimit, (int, float)):
            nlimit = 0
            logger.warning("Invalid nlimit parameter, using default: %d", nlimit)

        nx, ny, nz = data_shape
        logger.info("Processing 3D cube: %dx%dx%d, clip=%.1f, nlimit=%d",
                    nx, ny, nz, clip, nlimit)

        # Define output frame
        master_frame = np.zeros((nx, ny), dtype=float)
        n_pixels = np.zeros(nx * ny, dtype=int)

        # Track processing progress for large cubes
        total_pixels = nx * ny
        log_interval = max(1, total_pixels // 10)  # Log every 10% of progress

        # Process each pixel across the z-dimension
        processed_count = 0
        for i in range(nx):
            for j in range(ny):
                # Extract 1D line along z axis
                line = data[i, j, :]

                if nlimit >= 0:
                    # Sort the line
                    s1 = np.argsort(line)
                    trimmed_line = line[s1[:nz - nlimit - 1]]
                    sig, med = mad(trimmed_line, sigma=True), np.median(trimmed_line)
                    n_pixels_used = len(trimmed_line)
                else:
                    sig, med = mad(line, sigma=True), np.median(line)
                    n_pixels_used = len(line)

                # Handle division by zero
                if sig == 0:
                    sig = 1e-10

                # Store the number of pixels used - FIXED INDEXING
                n_pixels[i * ny + j] = n_pixels_used

                # Find values within clip limit
                w_g = np.where(np.abs(line - med) / sig <= clip)[0]

                # Calculate and store mean of good values
                if len(w_g) > 0:
                    master_frame[i, j] = np.mean(line[w_g])
                else:
                    master_frame[i, j] = med  # Fallback if no good values

                processed_count += 1

                # Log progress for large cubes
                if processed_count % log_interval == 0:
                    progress = 100 * processed_count / total_pixels
                    logger.debug("3D processing progress: %.1f%% (%d/%d pixels)",
                                 progress, processed_count, total_pixels)

        logger.info("3D cube processing completed: %dx%d output frame", nx, ny)

        # Log some statistics about the final frame
        final_median = np.median(master_frame)
        final_std = np.std(master_frame)
        logger.debug("Final 3D frame statistics: median=%.3f, std=%.3f",
                     final_median, final_std)

        return master_frame

    else:
        error_msg = f"Array must be 2D or 3D, got {data_ndim}D with shape {data_shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def find_targets(directory, run):
    """
    Find target names from science image list files.

    Parameters:
    -----------
    directory : str or Path
        Directory to search for list files
    run : str
        Run identifier

    Returns:
    --------
    list
        Sorted list of target names
    """
    logger.debug("Finding targets in directory: %s, run: %s", directory, run)

    # Define the regex pattern to match {run}_image_{target}.list
    # The pattern captures the target portion as a group
    pattern = re.compile(f"^{re.escape(run)}_image_(.+)\.list$")

    # List to store extracted target names
    targets = []

    try:
        # Scan the directory
        for filename in os.listdir(directory):
            # Check if the file matches our pattern
            match = pattern.match(filename)
            if match:
                # Extract the target name (captured group)
                target = match.group(1)
                targets.append(target)
                logger.debug("Found target: %s (from file: %s)", target, filename)

        targets_sorted = sorted(targets)
        logger.info("Found %d targets: %s", len(targets_sorted), targets_sorted)

        return targets_sorted

    except Exception as e:
        logger.error("Failed to scan directory %s: %s", directory, e)
        raise


def find_flat_filters(directory, run):
    """
    Find filter names from flat field list files.

    Parameters:
    -----------
    directory : str or Path
        Directory to search for list files
    run : str
        Run identifier

    Returns:
    --------
    list
        Sorted list of filter names
    """
    logger.debug("Finding flat filters in directory: %s, run: %s", directory, run)

    # Define the regex pattern to match both old and new style flat lists
    # Old style: {run}_flat_{filter}.list
    # New style: {run}_flat_{filter}_dawn.list and {run}_flat_{filter}_dusk.list
    old_pattern = re.compile(f"^{re.escape(run)}_flat_(.+)\.list$")
    new_pattern = re.compile(f"^{re.escape(run)}_flat_(.+)_(dawn|dusk)\.list$")

    # Set to store unique filter names
    filters = set()
    old_style_files = []
    new_style_files = []

    try:
        # Scan the directory
        for filename in os.listdir(directory):
            # Check for new style first (dawn/dusk)
            new_match = new_pattern.match(filename)
            if new_match:
                # Extract the filter name (first captured group)
                filter_name = new_match.group(1)
                time_period = new_match.group(2)
                filters.add(filter_name)
                new_style_files.append(filename)
                logger.debug("Found new-style flat: filter=%s, period=%s (file: %s)",
                             filter_name, time_period, filename)
            else:
                # Check for old style
                old_match = old_pattern.match(filename)
                if old_match:
                    # Extract the filter name (captured group)
                    filter_name = old_match.group(1)
                    filters.add(filter_name)
                    old_style_files.append(filename)
                    logger.debug("Found old-style flat: filter=%s (file: %s)",
                                 filter_name, filename)

        # Convert set back to list and sort for consistent ordering
        filters_sorted = sorted(list(filters))

        logger.info("Found %d filters: %s", len(filters_sorted), filters_sorted)
        if old_style_files:
            logger.info("Old-style flat files: %d", len(old_style_files))
        if new_style_files:
            logger.info("New-style flat files: %d", len(new_style_files))

        return filters_sorted

    except Exception as e:
        logger.error("Failed to scan directory %s: %s", directory, e)
        raise


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running utils.py as standalone script")

    # Example usage
    logger.info("Utils module functions available:")
    logger.info("  - mad(): Calculate Median Absolute Deviation")
    logger.info("  - medabsdevclip(): Clipped statistics using MAD")
    logger.info("  - find_targets(): Find target names from image lists")
    logger.info("  - find_flat_filters(): Find filter names from flat lists")