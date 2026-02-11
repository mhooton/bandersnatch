import os
import numpy as np
import logging
from astropy.io import fits
from astropy.table import Table
from utils import medabsdevclip
from scipy import interpolate, ndimage

# Set up logger for this module
logger = logging.getLogger(__name__)


def create_poststamp(image, x_position, y_position, boxsize, sky_mean=None, sky_sd=None, sky_sigma=None,
                     mask_centroid_pixels=False):
    """
    Extract poststamps around star positions.

    Parameters:
    -----------
    image : np.ndarray
        2D image array
    x_position : np.ndarray
        X positions of stars
    y_position : np.ndarray
        Y positions of stars
    boxsize : int
        Half-width of the box around each star

    Returns:
    --------
    np.ndarray : 3D array of shape (2*boxsize+1, 2*boxsize+1, n_stars)
    """
    n_stars = len(x_position)
    poststamp_array = np.zeros((2 * boxsize + 1, 2 * boxsize + 1, n_stars))
    ny, nx = image.shape  # Note: numpy arrays are (y, x) order

    logger.debug("Creating poststamps for %d stars with boxsize %d", n_stars, boxsize)
    logger.debug("Image shape: %d x %d", nx, ny)

    for i in range(n_stars):
        x_pos = int(x_position[i])
        y_pos = int(y_position[i])

        if (x_pos - boxsize >= 0 and x_pos + boxsize <= nx - 1 and
                y_pos - boxsize >= 0 and y_pos + boxsize <= ny - 1):
            poststamp_array[:, :, i] = image[y_pos - boxsize:y_pos + boxsize + 1,
                                       x_pos - boxsize:x_pos + boxsize + 1]
            logger.debug("Created poststamp for star %d at (%d, %d)", i, x_pos, y_pos)
        else:
            logger.warning("Star %d at (%d, %d) too close to image edge for boxsize %d",
                           i, x_pos, y_pos, boxsize)

    if mask_centroid_pixels and sky_mean is not None and sky_sd is not None and sky_sigma is not None:
        logger.debug("Applying centroid pixel masking with %.1f sigma threshold", sky_sigma)
        # Create a copy of the image for masking
        masked_image = image.copy()

        for i in range(n_stars):
            x_pos = int(x_position[i])
            y_pos = int(y_position[i])

            if (x_pos - boxsize >= 0 and x_pos + boxsize <= nx - 1 and
                    y_pos - boxsize >= 0 and y_pos + boxsize <= ny - 1):
                # Extract region around star
                region = masked_image[y_pos - boxsize:y_pos + boxsize + 1, x_pos - boxsize:x_pos + boxsize + 1]

                # Mask pixels above threshold
                threshold = sky_mean[i] + sky_sd[i] * sky_sigma
                mask_pixels = region >= threshold
                masked_count = np.sum(mask_pixels)
                logger.debug("Masked %d pixels above threshold %.2f for star %d",
                             masked_count, threshold, i)
                region[mask_pixels] = np.nan  # Set to bright value

                poststamp_array[:, :, i] = region

    return poststamp_array


def sky_background_square(image, x_position, y_position, nlimit, clip, boxsize):
    """
    Calculate sky background statistics around star positions.

    Parameters:
    -----------
    image : np.ndarray
        2D image array
    x_position : np.ndarray
        X positions of stars
    y_position : np.ndarray
        Y positions of stars
    nlimit : int
        Limit parameter for clipping
    clip : int
        Clipping parameter
    boxsize : int
        Half-width of the box around each star

    Returns:
    --------
    np.ndarray : 2D array of shape (2, n_stars) containing [median, mad] for each star
    """
    # Copy image to avoid modifying original
    im = image.copy().astype(float)

    n_stars = len(x_position)
    sky_med = np.zeros((2, n_stars))
    ny, nx = image.shape

    logger.debug("Calculating sky background for %d stars", n_stars)

    for i in range(n_stars):
        x_pos = int(x_position[i])
        y_pos = int(y_position[i])

        logger.debug("Processing sky for star %d at (%d, %d)", i, x_pos, y_pos)

        if (x_pos - boxsize >= 0 and x_pos + boxsize <= nx - 1 and
                y_pos - boxsize >= 0 and y_pos + boxsize <= ny - 1):

            # Mask the star region with NaNs
            im[y_pos - boxsize:y_pos + boxsize + 1, x_pos - boxsize:x_pos + boxsize + 1] = np.nan

            # Determine outer box size - check if we have space for 2*boxsize
            if (x_pos - 2 * boxsize >= 0 and x_pos + 2 * boxsize <= nx - 1 and
                    y_pos - 2 * boxsize >= 0 and y_pos + 2 * boxsize <= ny - 1):
                outer_box = 2.0
            else:
                outer_box = 1.5

            logger.debug("Using outer_box = %.1f for star %d", outer_box, i)

            # Extract sky region with proper bounds checking
            x_start = max(0, int(x_pos - outer_box * boxsize))
            x_end = min(nx, int(x_pos + outer_box * boxsize + 1))
            y_start = max(0, int(y_pos - outer_box * boxsize))
            y_end = min(ny, int(y_pos + outer_box * boxsize + 1))

            logger.debug("Sky region for star %d: x[%d:%d], y[%d:%d]", i, x_start, x_end, y_start, y_end)

            sky_region = im[y_start:y_end, x_start:x_end]
            valid_pixels = np.sum(~np.isnan(sky_region))
            logger.debug("Sky region shape: %s, valid pixels: %d", sky_region.shape, valid_pixels)

            if valid_pixels > 10:  # Need minimum pixels for statistics
                sky_med[:, i] = medabsdevclip(sky_region, clip, nlimit)
                logger.debug("Sky stats for star %d: median=%.2f, mad=%.2f",
                             i, sky_med[0, i], sky_med[1, i])
            else:
                logger.warning("Insufficient sky pixels (%d) for star %d", valid_pixels, i)
                sky_med[:, i] = [0.0, 1.0]  # Default values
        else:
            logger.warning("Star %d too close to edge for sky calculation", i)
            sky_med[:, i] = [0.0, 1.0]  # Default values

    return sky_med


def get_brightest_pixels(poststamp, x_position, y_position, boxsize):
    """
    Find the brightest pixel in each star's poststamp.

    Parameters:
    -----------
    poststamp : np.ndarray
        3D array of poststamps
    x_position : np.ndarray
        X positions of stars
    y_position : np.ndarray
        Y positions of stars
    boxsize : int
        Half-width of the box around each star

    Returns:
    --------
    np.ndarray : 3D array of shape (3, n_stars) containing [x, y, value] of brightest pixel
    """
    n_stars = poststamp.shape[2]
    brightest = np.zeros((3, n_stars))

    logger.debug("Finding brightest pixels for %d stars", n_stars)

    for i in range(n_stars):
        # Find location of maximum value
        stamp = poststamp[:, :, i]
        bright_location = np.unravel_index(np.argmax(stamp), stamp.shape)
        bright_value = stamp[bright_location]

        # Convert to original image coordinates
        brightest[0, i] = bright_location[1] + x_position[i] - boxsize  # x coordinate
        brightest[1, i] = bright_location[0] + y_position[i] - boxsize  # y coordinate
        brightest[2, i] = bright_value

        logger.debug("Star %d brightest pixel: (%.1f, %.1f) value=%.2f",
                     i, brightest[0, i], brightest[1, i], bright_value)

    return brightest


def get_star_widths(poststamp, x_position, y_position):
    """
    Calculate star widths in x and y directions.

    Parameters:
    -----------
    poststamp : np.ndarray
        3D array of poststamps
    x_position : np.ndarray
        X positions of stars (unused but kept for consistency)
    y_position : np.ndarray
        Y positions of stars (unused but kept for consistency)

    Returns:
    --------
    np.ndarray : 2D array of shape (2, n_stars) containing [x_width, y_width]
    """
    n_stars = poststamp.shape[2]
    widths = np.zeros((2, n_stars))

    logger.debug("Calculating star widths for %d stars", n_stars)

    for i in range(n_stars):
        stamp = poststamp[:, :, i]

        # Sum along axes to get profiles
        total_x = np.sum(stamp, axis=0)  # Sum along y-axis
        total_y = np.sum(stamp, axis=1)  # Sum along x-axis

        # Find peaks
        max_x = np.max(total_x)
        max_y = np.max(total_y)
        index_x = np.argmax(total_x)
        index_y = np.argmax(total_y)

        # Interpolate to find half-maximum points
        if index_x > 0 and index_x < len(total_x) - 1:
            # Left side
            x_indices_1 = np.arange(index_x + 1)
            interpol_x_1 = np.interp(0.5 * max_x, total_x[:index_x + 1], x_indices_1)

            # Right side
            x_indices_2 = np.arange(index_x, len(total_x))
            interpol_x_2 = np.interp(0.5 * max_x, total_x[index_x:][::-1], x_indices_2[::-1])

            widths[0, i] = interpol_x_2 - interpol_x_1

        if index_y > 0 and index_y < len(total_y) - 1:
            # Left side
            y_indices_1 = np.arange(index_y + 1)
            interpol_y_1 = np.interp(0.5 * max_y, total_y[:index_y + 1], y_indices_1)

            # Right side
            y_indices_2 = np.arange(index_y, len(total_y))
            interpol_y_2 = np.interp(0.5 * max_y, total_y[index_y:][::-1], y_indices_2[::-1])

            widths[1, i] = interpol_y_2 - interpol_y_1

        logger.debug("Star %d widths: x=%.2f, y=%.2f", i, widths[0, i], widths[1, i])

    return widths


def pixels_above_value(poststamp, flux_above_value):
    """
    Count pixels above a specified flux value for each star.

    Parameters:
    -----------
    poststamp : np.ndarray
        3D array of poststamps
    flux_above_value : float
        Flux threshold

    Returns:
    --------
    np.ndarray : 1D array with pixel counts for each star
    """
    n_stars = poststamp.shape[2]
    pixels_above = np.zeros(n_stars)

    logger.debug("Counting pixels above flux value %.2f for %d stars", flux_above_value, n_stars)

    for star_loop in range(n_stars):
        stamp = poststamp[:, :, star_loop]
        pixels_above[star_loop] = np.sum(stamp >= flux_above_value)
        logger.debug("Star %d: %d pixels above flux threshold", star_loop, pixels_above[star_loop])

    return pixels_above


def pixels_above_x_sigma(poststamp, sky_mean, sky_sd, x):
    """
    Count pixels above x sigma from sky background for each star.

    Parameters:
    -----------
    poststamp : np.ndarray
        3D array of poststamps
    sky_mean : np.ndarray
        Sky background means for each star
    sky_sd : np.ndarray
        Sky background standard deviations for each star
    x : float
        Sigma threshold

    Returns:
    --------
    np.ndarray : 1D array with pixel counts for each star
    """
    n_stars = len(sky_mean)
    pixels = np.zeros(n_stars)

    logger.debug("Counting pixels above %.1f sigma for %d stars", x, n_stars)

    for i in range(n_stars):
        if sky_sd[i] > 0:
            stamp = poststamp[:, :, i]
            sigma_values = np.abs(stamp - sky_mean[i]) / sky_sd[i]
            pixels[i] = np.sum(sigma_values >= x)
            logger.debug("Star %d: %d pixels above %.1f sigma", i, pixels[i], x)
        else:
            logger.warning("Star %d has zero sky standard deviation", i)

    return pixels


def extract_airmass(header):
    """
    Extract AIRMASS value from FITS header.

    Parameters:
    -----------
    header : astropy.io.fits.Header
        FITS header object

    Returns:
    --------
    float : AIRMASS value
    """
    # First try direct AIRMASS header
    airmass = header.get('AIRMASS', None)
    if airmass is not None:
        return float(airmass)

    # If not found, try to derive from ALTITUDE
    altitude = header.get('ALTITUDE', None)
    if altitude is not None:
        # Convert altitude to airmass using sec(zenith_angle)
        # zenith_angle = 90 - altitude (in degrees)
        import numpy as np
        altitude_rad = np.radians(float(altitude))
        airmass = 1.0 / np.sin(altitude_rad)  # sec(90-alt) = 1/sin(alt)
        return airmass

    # If neither found, return -1
    return -1.0


def centroids_onestar(poststamp, x_init, y_init, mean_background, sd_background, sd_cutoff, tracking_star, boxsize,
                      mask_centroid_pixels=False):
    """
    Calculate centroids for stars based on tracking star using 4-connected components.

    Parameters:
    -----------
    poststamp : np.ndarray
        3D array of poststamps
    x_init : np.ndarray
        Initial x positions
    y_init : np.ndarray
        Initial y positions
    mean_background : np.ndarray
        Sky background means
    sd_background : np.ndarray
        Sky background standard deviations
    sd_cutoff : float
        Sigma threshold for pixel selection
    tracking_star : int
        Index of tracking star
    boxsize : int
        Half-width of the box around each star
    mask_centroid_pixels : bool
        Whether to mask pixels used for centroiding

    Returns:
    --------
    tuple : (x_final, y_final) - final centroid positions
    """
    n_stars = len(x_init)
    mean_x = np.zeros(n_stars)
    mean_y = np.zeros(n_stars)

    logger.debug("Calculating centroids using tracking star %d with %.1f sigma cutoff",
                 tracking_star, sd_cutoff)

    # Use tracking star to determine offset
    if sd_background[tracking_star] > 0:
        tracking_stamp = poststamp[:, :, tracking_star]
        sigma_values = np.abs(tracking_stamp - mean_background[tracking_star]) / sd_background[tracking_star]

        # Create binary mask of pixels above threshold
        above_threshold = sigma_values >= sd_cutoff

        # Define 4-connected structure (only orthogonal neighbors)
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)

        # Find connected components
        labeled_array, num_features = ndimage.label(above_threshold, structure=structure)

        logger.debug("Found %d connected components above %.1f sigma threshold", num_features, sd_cutoff)

        if num_features == 0:
            # No pixels above threshold - use previous positions
            logger.warning("No pixels above sigma threshold for tracking star %d - using initial positions",
                           tracking_star)
            mean_x[:] = 0.0
            mean_y[:] = 0.0

        elif num_features == 1:
            # Only one component - use it
            component_coords = np.where(labeled_array == 1)
            mean_x_offset = np.mean(component_coords[1]) - boxsize  # x coordinates
            mean_y_offset = np.mean(component_coords[0]) - boxsize  # y coordinates

            logger.debug("Single component offset: x=%.2f, y=%.2f (from %d pixels)",
                         mean_x_offset, mean_y_offset, len(component_coords[0]))

            mean_x[:] = mean_x_offset
            mean_y[:] = mean_y_offset

        else:
            # Multiple components - find largest by pixel count
            component_sizes = []
            component_centers = []

            for component_id in range(1, num_features + 1):
                component_coords = np.where(labeled_array == component_id)
                size = len(component_coords[0])
                center_x = np.mean(component_coords[1])
                center_y = np.mean(component_coords[0])

                component_sizes.append(size)
                component_centers.append((center_x, center_y))

            # Find largest component(s)
            max_size = max(component_sizes)
            largest_indices = [i for i, size in enumerate(component_sizes) if size == max_size]

            if len(largest_indices) == 1:
                # Single largest component
                chosen_component = largest_indices[0] + 1  # +1 because component IDs start at 1
                logger.debug("Largest component has %d pixels", max_size)

            else:
                # Tie-breaking: choose closest to center
                center_x, center_y = boxsize, boxsize  # Center of poststamp
                min_distance = float('inf')
                closest_idx = 0

                for idx in largest_indices:
                    comp_center_x, comp_center_y = component_centers[idx]
                    distance = np.sqrt((comp_center_x - center_x) ** 2 + (comp_center_y - center_y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx

                chosen_component = closest_idx + 1
                logger.debug("Tie-breaking: chose component %d (closest to center, distance=%.2f)",
                             chosen_component, min_distance)

            # Calculate centroid from chosen component
            component_coords = np.where(labeled_array == chosen_component)

            if len(component_coords[0]) == 0:
                # Shouldn't happen, but safety check
                logger.warning("Chosen component is empty - using initial positions")
                mean_x[:] = 0.0
                mean_y[:] = 0.0
            else:
                mean_x_offset = np.mean(component_coords[1]) - boxsize  # x coordinates
                mean_y_offset = np.mean(component_coords[0]) - boxsize  # y coordinates

                # Check if component is very small
                if len(component_coords[0]) <= 3:
                    logger.warning("Largest component only has %d pixels - centroid may be unreliable",
                                   len(component_coords[0]))

                logger.debug("Chosen component offset: x=%.2f, y=%.2f (from %d pixels)",
                             mean_x_offset, mean_y_offset, len(component_coords[0]))

                mean_x[:] = mean_x_offset
                mean_y[:] = mean_y_offset

        # Handle isolated pixels case
        if num_features > 0:
            all_sizes = []
            for component_id in range(1, num_features + 1):
                component_coords = np.where(labeled_array == component_id)
                all_sizes.append(len(component_coords[0]))

            if all(size == 1 for size in all_sizes):
                # All components are isolated pixels
                logger.warning("All %d components are isolated pixels - choosing closest to center", num_features)

                center_x, center_y = boxsize, boxsize
                min_distance = float('inf')
                closest_component = 1

                for component_id in range(1, num_features + 1):
                    component_coords = np.where(labeled_array == component_id)
                    comp_x = component_coords[1][0]  # Single pixel
                    comp_y = component_coords[0][0]
                    distance = np.sqrt((comp_x - center_x) ** 2 + (comp_y - center_y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_component = component_id

                # Use the closest isolated pixel
                component_coords = np.where(labeled_array == closest_component)
                mean_x_offset = component_coords[1][0] - boxsize
                mean_y_offset = component_coords[0][0] - boxsize

                logger.debug("Using closest isolated pixel: offset x=%.2f, y=%.2f",
                             mean_x_offset, mean_y_offset)

                mean_x[:] = mean_x_offset
                mean_y[:] = mean_y_offset

    else:
        logger.warning("Tracking star %d has zero sky standard deviation", tracking_star)
        mean_x[:] = 0.0
        mean_y[:] = 0.0

    # Calculate final positions
    x_final = x_init + mean_x
    y_final = y_init + mean_y

    logger.debug("Applied centroid corrections to all %d stars", n_stars)

    if mask_centroid_pixels and sd_background[tracking_star] > 0:
        # Mask pixels used for centroiding in the poststamp
        tracking_stamp = poststamp[:, :, tracking_star]
        sigma_values = np.abs(tracking_stamp - mean_background[tracking_star]) / sd_background[tracking_star]

        # Only mask the pixels from the chosen component
        above_threshold = sigma_values >= sd_cutoff
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        labeled_array, num_features = ndimage.label(above_threshold, structure=structure)

        if num_features > 0:
            # Find the same component we used for centroiding
            # (This repeats some logic but ensures consistency)
            if num_features == 1:
                mask_pixels = labeled_array == 1
            else:
                component_sizes = []
                for component_id in range(1, num_features + 1):
                    component_coords = np.where(labeled_array == component_id)
                    component_sizes.append(len(component_coords[0]))

                max_size = max(component_sizes)
                largest_indices = [i for i, size in enumerate(component_sizes) if size == max_size]

                if len(largest_indices) == 1:
                    chosen_component = largest_indices[0] + 1
                else:
                    # Same tie-breaking logic
                    center_x, center_y = boxsize, boxsize
                    min_distance = float('inf')
                    closest_idx = 0

                    for idx in largest_indices:
                        component_coords = np.where(labeled_array == (idx + 1))
                        comp_center_x = np.mean(component_coords[1])
                        comp_center_y = np.mean(component_coords[0])
                        distance = np.sqrt((comp_center_x - center_x) ** 2 + (comp_center_y - center_y) ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_idx = idx

                    chosen_component = closest_idx + 1

                mask_pixels = labeled_array == chosen_component

            masked_count = np.sum(mask_pixels)
            logger.debug("Masked %d centroiding pixels in tracking star poststamp", masked_count)
            poststamp[:, :, tracking_star][mask_pixels] = np.max(tracking_stamp) * 1.5

    return x_final, y_final


def create_centroiding_plots(centroiding_dir):
    """
    Create diagnostic plots for centroiding results.

    Parameters:
    -----------
    centroiding_dir : Path
        Centroiding output directory
    """
    logger.info("Creating centroiding diagnostic plots")

    try:
        import matplotlib.pyplot as plt

        # Read the centroids table
        centroids_file = centroiding_dir / "centroids.fits"

        from astropy.table import Table
        table = Table.read(centroids_file)

        bjd = table['BJD']
        n_stars = len(table['xc'][0])  # Number of stars

        logger.debug("Creating plots for %d stars from %d observations", n_stars, len(bjd))

        # Create plots directory
        plots_dir = centroiding_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Variables to plot
        variables = ['xc', 'yc', 'x_bright', 'y_bright', 'val_bright',
                     'x_width', 'y_width', 'n_pixels_above', 'sig3_pixels',
                     'sig5_pixels', 'sig10_pixels']

        for var in variables:
            data = np.array(table[var])  # Shape: (n_images, n_stars)
            logger.debug("Creating plots for variable %s", var)

            for star_num in range(n_stars):
                plt.figure(figsize=(10, 6))
                plt.plot(bjd, data[:, star_num], 'o-', markersize=3)
                plt.xlabel('BJD')
                plt.ylabel(var)
                plt.title(f'{var} vs BJD - Star {star_num}')
                plt.grid(True, alpha=0.3)

                # Save plot
                plot_path = plots_dir / f"{var}_star_{star_num}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

        logger.info("Centroiding plots saved to %s", plots_dir)

    except ImportError:
        logger.warning("Matplotlib not available, skipping centroiding plots")
    except Exception as e:
        logger.error("Failed to create centroiding plots: %s", e)


def centroid_loop(star_x_input, star_y_input, boxsize, nlimit_centroid, clip_centroid, sky_sigma,
                  tracking_star, flux_above_value, image=None, fk5=False, header=None,
                  route_string=None, reduce_string=None, filename=None, suffix=None,
                  mask_centroid_pixels=False):
    """
    Perform centroiding calculations for a single image.

    Returns:
    --------
    dict : Dictionary containing all calculated values for this image
    """
    logger.debug("Processing centroiding for single image")

    if image is None:
        from astropy.io import fits
        image_path = f"{route_string}{reduce_string}/{filename}_{reduce_string}{suffix}.fits"
        logger.debug("Loading image from %s", image_path)
        image = fits.getdata(image_path, scale_back=True)

    if fk5 and header is not None:
        logger.debug("Using WCS coordinates")
        from astropy.wcs import WCS
        wcs = WCS(header)
        star_x_init, star_y_init = wcs.all_world2pix(star_x_input, star_y_input, 0)
        star_x_final = star_x_init.copy()
        star_y_final = star_y_init.copy()
    else:
        star_x_init = star_x_input.copy()
        star_y_init = star_y_input.copy()
        star_x_final = None
        star_y_final = None

    # Get poststamp of the star
    poststamp_array = create_poststamp(image, star_x_init, star_y_init, boxsize)

    # Get mean and sd of the larger square outside the poststamp (the sky background)
    sky_background = sky_background_square(image, star_x_init, star_y_init, nlimit_centroid, clip_centroid, boxsize)
    sky_mean = sky_background[0, :]
    sky_sd = sky_background[1, :]

    logger.debug("Sky statistics: mean=%s", sky_mean)
    logger.debug("Sky statistics: sd=%s", sky_sd)
    logger.debug("Sky thresholds (mean + %.1f*sd): %s", sky_sigma, sky_mean + sky_sigma * sky_sd)

    # Get the positions of the brightest pixels in the PSF, and store in the structure
    brightest_pixels = get_brightest_pixels(poststamp_array, star_x_init, star_y_init, boxsize)
    x_bright_frame = brightest_pixels[0, :]
    y_bright_frame = brightest_pixels[1, :]
    val_bright_frame = brightest_pixels[2, :]

    # Get the x and y widths of the star, and store in the structure
    star_widths = get_star_widths(poststamp_array, star_x_init, star_y_init)
    x_width_frame = star_widths[0, :]
    y_width_frame = star_widths[1, :]

    # Store the number of pixels with a value above a specified value
    n_pixels_above_frame = pixels_above_value(poststamp_array, flux_above_value)

    # Store the number of pixels in the PSF 3, 5 and 10 sigma above the sky background level
    sig3_pixels_frame = pixels_above_x_sigma(poststamp_array, sky_mean, sky_sd, 3)
    sig5_pixels_frame = pixels_above_x_sigma(poststamp_array, sky_mean, sky_sd, 5)
    sig10_pixels_frame = pixels_above_x_sigma(poststamp_array, sky_mean, sky_sd, 10)

    if not fk5:
        # Get the x and y centroids of the star, and store in the structure
        star_x_final, star_y_final = centroids_onestar(poststamp_array, star_x_init, star_y_init, sky_mean,
                                                       sky_sd, sky_sigma, tracking_star, boxsize,
                                                       mask_centroid_pixels=mask_centroid_pixels)

    # Save poststamps for each star using final positions
    if mask_centroid_pixels:
        final_poststamps = create_poststamp(image, star_x_final, star_y_final, boxsize,
                                            sky_mean, sky_sd, sky_sigma, mask_centroid_pixels=True)
    else:
        final_poststamps = create_poststamp(image, star_x_final, star_y_final, boxsize)

    # Return all calculated values as a dictionary
    return {
        'xc': star_x_final,
        'yc': star_y_final,
        'x_bright': x_bright_frame,
        'y_bright': y_bright_frame,
        'val_bright': val_bright_frame,
        'x_width': x_width_frame,
        'y_width': y_width_frame,
        'n_pixels_above': n_pixels_above_frame,
        'sig3_pixels': sig3_pixels_frame,
        'sig5_pixels': sig5_pixels_frame,
        'sig10_pixels': sig10_pixels_frame,
        'poststamps': final_poststamps
    }


def centroid(outdir, run, target, initial_positions, boxsize, nlimit_centroid, clip_centroid, sky_sigma, tracking_star,
             flux_above_value, mask_centroid_pixels=False):
    """
    Perform centroiding on all processed images for a given target.

    Parameters:
    -----------
    outdir : Path
        Output directory containing the data
    run : str
        Run identifier
    target : str
        Target name
    initial_positions : list
        List of [x, y] positions for each star
    boxsize : int
        Size of box around star for centroiding
    nlimit_centroid : int
        Limit for centroiding
    clip_centroid : int
        Clipping parameter for centroiding
    sky_sigma : int
        Sky sigma parameter
    tracking_star : int
        Index of tracking star
    flux_above_value : float
        Flux threshold value
    """
    logger.info("Starting centroiding for target %s", target)
    logger.debug("Parameters: boxsize=%d, tracking_star=%d, sky_sigma=%d, flux_threshold=%.1f",
                 boxsize, tracking_star, sky_sigma, flux_above_value)

    # 1. Initialize result lists
    xc = []
    yc = []
    airmass = []
    x_bright = []
    y_bright = []
    val_bright = []
    x_width = []
    y_width = []
    n_pixels_above = []
    sig3_pixels = []
    sig5_pixels = []
    sig10_pixels = []
    poststamps = []
    file_paths = []
    bjd = []

    centroiding_dir = outdir / target / "centroiding"
    os.makedirs(centroiding_dir, exist_ok=True)
    logger.debug("Created centroiding directory: %s", centroiding_dir)

    # Create plots directory inside centroiding
    plots_dir = centroiding_dir / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Update all file paths to use centroiding_dir:
    structure_path = centroiding_dir / "centroids.fits"

    # 2. Read processed image list
    list_file = outdir / "calib" / f"{run}_proc_{target}.list"
    logger.debug("Reading processed image list from %s", list_file)

    try:
        filenames = []
        with open(list_file, 'r') as f:
            for line in f:
                filename = line.strip()
                if filename:
                    filenames.append(filename)
        logger.info("Found %d processed images for target %s", len(filenames), target)
    except Exception as e:
        logger.error("Failed to read processed image list %s: %s", list_file, e)
        raise

    # 3. Extract star positions
    star_x_input = np.array(initial_positions)[:, 0]
    star_y_input = np.array(initial_positions)[:, 1]

    logger.info("Processing %d stars at initial positions: %s",
                len(star_x_input), list(zip(star_x_input, star_y_input)))

    # 4. Loop through images
    successful_images = 0
    for i, filename in enumerate(filenames):
        logger.debug("Processing image %d/%d: %s", i + 1, len(filenames), filename)

        try:
            image_path = outdir / target / run / filename
            with fits.open(image_path) as hdul:
                image = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                # Extract BJD from header
                bjd_obs = header.get('BJD-OBS', 0.0)
                if bjd_obs == 0.0:
                    logger.warning("No BJD-OBS found in header for %s, using image index", filename)
                    bjd_obs = float(i)

                # Extract AIRMASS - ADD THESE LINES
                airmass_obs = extract_airmass(header)

            # Call centroid_loop for this image
            results = centroid_loop(star_x_input, star_y_input, boxsize, nlimit_centroid,
                                    clip_centroid, sky_sigma, tracking_star, flux_above_value,
                                    image=image, header=header, mask_centroid_pixels=mask_centroid_pixels)

            # Append results to lists
            xc.append(results['xc'])
            yc.append(results['yc'])
            airmass.append(airmass_obs)
            x_bright.append(results['x_bright'])
            y_bright.append(results['y_bright'])
            val_bright.append(results['val_bright'])
            x_width.append(results['x_width'])
            y_width.append(results['y_width'])
            n_pixels_above.append(results['n_pixels_above'])
            sig3_pixels.append(results['sig3_pixels'])
            sig5_pixels.append(results['sig5_pixels'])
            sig10_pixels.append(results['sig10_pixels'])
            poststamps.append(results['poststamps'])
            file_paths.append(str(image_path))
            bjd.append(bjd_obs)

            successful_images += 1

        except Exception as e:
            logger.error("Failed to process image %s: %s", filename, e)
            # Continue with remaining images
            continue

    logger.info("Successfully processed %d/%d images", successful_images, len(filenames))

    if successful_images == 0:
        logger.error("No images were successfully processed for centroiding")
        raise RuntimeError("No images were successfully processed for centroiding")

    # 5. Create FITS table with one row per image
    logger.info("Creating centroids table with %d observations", len(xc))

    # Convert lists to numpy arrays
    table_data = {
        'BJD': np.array(bjd),
        'File': file_paths,
        'airmass': np.array(airmass),  # ADD THIS LINE
        'xc': np.array(xc),
        'yc': np.array(yc),
        'x_bright': np.array(x_bright),
        'y_bright': np.array(y_bright),
        'val_bright': np.array(val_bright),
        'x_width': np.array(x_width),
        'y_width': np.array(y_width),
        'n_pixels_above': np.array(n_pixels_above),
        'sig3_pixels': np.array(sig3_pixels),
        'sig5_pixels': np.array(sig5_pixels),
        'sig10_pixels': np.array(sig10_pixels)
    }

    # Create astropy Table
    centroid_table = Table(table_data)
    logger.debug("Created table with columns: %s", list(centroid_table.colnames))

    # Create and save poststamps data cubes for each star
    logger.info("Saving poststamps for %d stars", len(star_x_input))
    poststamps_array = np.array(poststamps)  # Shape: (n_images, 2*boxsize+1, 2*boxsize+1, n_stars)

    n_stars = len(star_x_input)

    for star_num in range(n_stars):
        try:
            # Extract poststamps for this star: (n_images, 2*boxsize+1, 2*boxsize+1)
            star_poststamps = poststamps_array[:, :, :, star_num]

            # Create and write FITS image file for this star's poststamps
            poststamp_path = centroiding_dir / f"poststamps_{star_num}.fits"
            fits.writeto(poststamp_path, star_poststamps, overwrite=True)

            logger.debug("Poststamps for star %d saved to %s (shape: %s)",
                         star_num, poststamp_path, star_poststamps.shape)

        except Exception as e:
            logger.error("Failed to save poststamps for star %d: %s", star_num, e)

    # 6. Write centroids table
    try:
        centroid_table.write(structure_path, format='fits', overwrite=True)
        logger.info("Centroids table saved to %s", structure_path)
    except Exception as e:
        logger.error("Failed to save centroids table: %s", e)
        raise

    # Create diagnostic plots
    try:
        create_centroiding_plots(centroiding_dir)
    except Exception as e:
        logger.error("Failed to create diagnostic plots: %s", e)
        # Continue execution even if plots fail

    logger.info("Centroiding completed successfully for target %s", target)
    logger.info("Results saved to %s", centroiding_dir)


if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Running centroid.py as standalone script")
    # You would call centroid() here with appropriate parameters