import numpy as np
import warnings
from typing import Tuple, Optional, Union
from scipy import stats

def mmm(sky_vector: np.ndarray,
        highbad: Optional[float] = None,
        debug: bool = False,
        readnoise: Optional[float] = None,
        integer: bool = False,
        maxiter: int = 50,
        silent: bool = False,
        minsky: int = 20) -> Tuple[float, float, float, int]:
    """
    Estimate the sky background in a stellar contaminated field.

    MMM assumes that contaminated sky pixel values overwhelmingly display
    POSITIVE departures from the true value. Adapted from DAOPHOT.

    Parameters:
    -----------
    sky_vector : np.ndarray
        Array containing sky values
    highbad : float, optional
        Scalar value of the (lowest) "bad" pixel level
    debug : bool, optional
        If True, display additional information
    readnoise : float, optional
        Read noise value for better median computation
    integer : bool, optional
        Set if input contains only discrete integer values
    maxiter : int, optional
        Maximum number of iterations (default: 50)
    silent : bool, optional
        Suppress error messages
    minsky : int, optional
        Minimum number of sky values required (default: 20)

    Returns:
    --------
    skymod : float
        Estimated mode of sky values
    sigma : float
        Standard deviation of sky peak (-1.0 if failed)
    skew : float
        Skewness of sky peak
    nsky : int
        Number of pixels used for computation
    """

    nsky = len(sky_vector)

    if nsky < minsky:
        if not silent:
            warnings.warn(f"Input vector must contain at least {minsky} elements")
        return 0.0, -1.0, 0.0, 0

    nlast = nsky - 1

    if debug:
        print(f"Processing {nsky} element array")

    # Sort sky values
    sky = np.sort(sky_vector)

    # Compute median
    skymid = 0.5 * sky[(nsky - 1) // 2] + 0.5 * sky[nsky // 2]

    # Initial cuts
    cut1 = min(skymid - sky[0], sky[nsky - 1] - skymid)
    if highbad is not None:
        cut1 = min(cut1, highbad - skymid)

    cut2 = skymid + cut1
    cut1 = skymid - cut1

    # Select pixels between cuts
    good = (sky <= cut2) & (sky >= cut1)
    ngood = np.sum(good)

    if ngood == 0:
        if not silent:
            warnings.warn(f"No sky values fall within {cut1} and {cut2}")
        return 0.0, -1.0, 0.0, 0

    # Get indices of good pixels
    good_indices = np.where(good)[0]
    minimm = good_indices[0] - 1
    maximm = good_indices[-1]

    # Subtract median for better arithmetic accuracy
    delta = sky[good] - skymid
    sum_delta = np.sum(delta, dtype=np.float64)
    sumsq = np.sum(delta ** 2, dtype=np.float64)

    # Compute initial mean and sigma
    skymed = 0.5 * sky[(minimm + maximm + 1) // 2] + 0.5 * sky[(minimm + maximm) // 2 + 1]
    skymn = sum_delta / (maximm - minimm)
    sigma = np.sqrt(sumsq / (maximm - minimm) - skymn ** 2)
    skymn = skymn + skymid

    # Initial mode estimate
    skymod = 3.0 * skymed - 2.0 * skymn if skymed < skymn else skymn

    # Rejection and recomputation loop
    niter = 0
    clamp = 1.0
    old = 0.0

    while True:
        niter += 1

        if niter > maxiter:
            if not silent:
                warnings.warn(f"Too many ({maxiter}) iterations, unable to compute sky")
            return 0.0, -1.0, 0.0, 0

        if maximm - minimm < minsky:
            if not silent:
                warnings.warn(f"Too few ({maximm - minimm}) valid sky elements")
            return 0.0, -1.0, 0.0, 0

        # Compute Chauvenet rejection criterion
        r = np.log10(float(maximm - minimm))
        r = max(2.0, (-0.1042 * r + 1.1695) * r + 0.8895)

        # Compute rejection limits
        cut = r * sigma + 0.5 * abs(skymn - skymod)
        if integer:
            cut = max(cut, 1.5)

        cut1 = skymod - cut
        cut2 = skymod + cut

        # Adjust minimum
        redo = False
        newmin = minimm
        tst_min = sky[newmin + 1] >= cut1 if newmin + 1 < nsky else True
        done = (newmin == -1) and tst_min

        if not done:
            done = (sky[max(newmin, 0)] < cut1) and tst_min

        if not done:
            istep = 1 if not tst_min else -1
            while not done:
                newmin += istep
                done = (newmin == -1) or (newmin == nlast)
                if not done:
                    done = (sky[newmin] <= cut1) and (sky[newmin + 1] >= cut1)

            if tst_min:
                delta = sky[newmin + 1:minimm + 1] - skymid
            else:
                delta = sky[minimm + 1:newmin + 1] - skymid

            sum_delta -= istep * np.sum(delta, dtype=np.float64)
            sumsq -= istep * np.sum(delta ** 2, dtype=np.float64)
            redo = True
            minimm = newmin

        # Adjust maximum
        newmax = maximm
        tst_max = sky[maximm] <= cut2
        done = (maximm == nlast) and tst_max

        if not done:
            done = tst_max and (sky[min(maximm + 1, nlast)] > cut2)

        if not done:
            istep = 1 if tst_max else -1
            while not done:
                newmax += istep
                done = (newmax == nlast) or (newmax == -1)
                if not done:
                    done = (sky[newmax] <= cut2) and (sky[newmax + 1] >= cut2)

            if tst_max:
                delta = sky[maximm + 1:newmax + 1] - skymid
            else:
                delta = sky[newmax + 1:maximm + 1] - skymid

            sum_delta += istep * np.sum(delta, dtype=np.float64)
            sumsq += istep * np.sum(delta ** 2, dtype=np.float64)
            redo = True
            maximm = newmax

        # Compute new mean and sigma
        nsky_iter = maximm - minimm

        if nsky_iter < minsky:
            if not silent:
                warnings.warn("Outlier rejection left too few sky elements")
            return 0.0, -1.0, 0.0, 0

        skymn = sum_delta / nsky_iter
        sigma = float(np.sqrt(max(sumsq / nsky_iter - skymn ** 2, 0)))
        skymn = skymn + skymid

        # Compute robust median using central 20%
        center = (minimm + 1 + maximm) / 2.0
        side = round(0.2 * (maximm - minimm)) / 2.0 + 0.25
        j = int(round(center - side))
        k = int(round(center + side))

        # Adjust for read noise if provided
        if readnoise is not None:
            l = int(round(center - 0.25))
            m = int(round(center + 0.25))
            r_noise = 0.25 * readnoise

            while (j > 0 and k < nsky - 1 and
                   ((sky[l] - sky[j] < r_noise) or (sky[k] - sky[m] < r_noise))):
                j -= 1
                k += 1

        skymed = np.mean(sky[j:k + 1])

        # Update mode
        dmod = 3.0 * skymed - 2.0 * skymn - skymod if skymed < skymn else skymn - skymod

        # Prevent oscillations
        if dmod * old < 0:
            clamp = 0.5 * clamp

        skymod = skymod + clamp * dmod
        old = dmod

        if not redo:
            break

    # Final calculations
    skew = float((skymn - skymod) / max(1.0, sigma))
    nsky_final = maximm - minimm

    if debug:
        print(f"Number of unrejected sky elements: {nsky_final}")
        print(f"Number of iterations: {niter}")
        print(f"Mode, Sigma, Skew of sky vector: {skymod}, {sigma}, {skew}")

    return skymod, sigma, skew, nsky_final

def arc(x: Union[float, np.ndarray],
        y0: Union[float, np.ndarray],
        y1: Union[float, np.ndarray],
        r: float) -> Union[float, np.ndarray]:
    """
    Compute the area within an arc of a circle.

    Parameters:
    -----------
    x : float or array
        X coordinate
    y0, y1 : float or array
        Y coordinates defining the arc
    r : float
        Radius of circle

    Returns:
    --------
    area : float or array
        Area of the wedge
    """
    return 0.5 * r * r * (np.arctan(y1 / x) - np.arctan(y0 / x))


def chord(x: Union[float, np.ndarray],
          y0: Union[float, np.ndarray],
          y1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the area of a triangle defined by origin and two points.

    Parameters:
    -----------
    x : float or array
        X coordinate
    y0, y1 : float or array
        Y coordinates

    Returns:
    --------
    area : float or array
        Signed area of triangle
    """
    return 0.5 * x * (y1 - y0)


def oneside(x: Union[float, np.ndarray],
            y0: Union[float, np.ndarray],
            y1: Union[float, np.ndarray],
            r: float) -> Union[float, np.ndarray]:
    """
    Compute area of intersection between triangle and circle.

    Parameters:
    -----------
    x : float or array
        X coordinate
    y0, y1 : float or array
        Y coordinates defining triangle vertices
    r : float
        Circle radius

    Returns:
    --------
    area : float or array
        Intersection area
    """
    # Handle scalar case
    if np.isscalar(x):
        if x == 0:
            return 0.0

        if abs(x) >= r:
            return arc(x, y0, y1, r)

        yh = np.sqrt(r * r - x * x)

        if y0 <= -yh:
            if y1 <= -yh:
                return arc(x, y0, y1, r)
            elif y1 <= yh:
                return arc(x, y0, -yh, r) + chord(x, -yh, y1)
            else:
                return arc(x, y0, -yh, r) + chord(x, -yh, yh) + arc(x, yh, y1, r)
        elif y0 < yh:
            if y1 <= -yh:
                return chord(x, y0, -yh) + arc(x, -yh, y1, r)
            elif y1 <= yh:
                return chord(x, y0, y1)
            else:
                return chord(x, y0, yh) + arc(x, yh, y1, r)
        else:  # y0 >= yh
            if y1 <= -yh:
                return arc(x, y0, yh, r) + chord(x, yh, -yh) + arc(x, -yh, y1, r)
            elif y1 <= yh:
                return arc(x, y0, yh, r) + chord(x, yh, y1)
            else:
                return arc(x, y0, y1, r)

    # Handle array case
    x = np.asarray(x)
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)

    # Initialize result array
    ans = np.zeros_like(x, dtype=float)

    # Handle x = 0 case
    zero_mask = (x == 0)
    ans[zero_mask] = 0.0

    # Handle |x| >= r case
    outside_mask = (np.abs(x) >= r) & ~zero_mask
    if np.any(outside_mask):
        ans[outside_mask] = arc(x[outside_mask], y0[outside_mask], y1[outside_mask], r)

    # Handle |x| < r case
    inside_mask = (np.abs(x) < r) & ~zero_mask
    if not np.any(inside_mask):
        return ans

    xi = x[inside_mask]
    y0i = y0[inside_mask]
    y1i = y1[inside_mask]

    yh = np.sqrt(r * r - xi * xi)

    # Case 1: y0 <= -yh
    case1_mask = (y0i <= -yh)
    if np.any(case1_mask):
        y0_c1 = y0i[case1_mask]
        y1_c1 = y1i[case1_mask]
        x_c1 = xi[case1_mask]
        yh_c1 = yh[case1_mask]

        # Subcase 1a: y1 <= -yh
        subcase1a = (y1_c1 <= -yh_c1)
        if np.any(subcase1a):
            idx = np.where(inside_mask)[0][np.where(case1_mask)[0][subcase1a]]
            ans[idx] = arc(x_c1[subcase1a], y0_c1[subcase1a], y1_c1[subcase1a], r)

        # Subcase 1b: -yh < y1 <= yh
        subcase1b = (-yh_c1 < y1_c1) & (y1_c1 <= yh_c1)
        if np.any(subcase1b):
            idx = np.where(inside_mask)[0][np.where(case1_mask)[0][subcase1b]]
            ans[idx] = (arc(x_c1[subcase1b], y0_c1[subcase1b], -yh_c1[subcase1b], r) +
                        chord(x_c1[subcase1b], -yh_c1[subcase1b], y1_c1[subcase1b]))

        # Subcase 1c: y1 > yh
        subcase1c = (y1_c1 > yh_c1)
        if np.any(subcase1c):
            idx = np.where(inside_mask)[0][np.where(case1_mask)[0][subcase1c]]
            ans[idx] = (arc(x_c1[subcase1c], y0_c1[subcase1c], -yh_c1[subcase1c], r) +
                        chord(x_c1[subcase1c], -yh_c1[subcase1c], yh_c1[subcase1c]) +
                        arc(x_c1[subcase1c], yh_c1[subcase1c], y1_c1[subcase1c], r))

    # Case 2: -yh < y0 < yh
    case2_mask = (-yh < y0i) & (y0i < yh)
    if np.any(case2_mask):
        y0_c2 = y0i[case2_mask]
        y1_c2 = y1i[case2_mask]
        x_c2 = xi[case2_mask]
        yh_c2 = yh[case2_mask]

        # Subcase 2a: y1 <= -yh
        subcase2a = (y1_c2 <= -yh_c2)
        if np.any(subcase2a):
            idx = np.where(inside_mask)[0][np.where(case2_mask)[0][subcase2a]]
            ans[idx] = (chord(x_c2[subcase2a], y0_c2[subcase2a], -yh_c2[subcase2a]) +
                        arc(x_c2[subcase2a], -yh_c2[subcase2a], y1_c2[subcase2a], r))

        # Subcase 2b: -yh < y1 <= yh
        subcase2b = (-yh_c2 < y1_c2) & (y1_c2 <= yh_c2)
        if np.any(subcase2b):
            idx = np.where(inside_mask)[0][np.where(case2_mask)[0][subcase2b]]
            ans[idx] = chord(x_c2[subcase2b], y0_c2[subcase2b], y1_c2[subcase2b])

        # Subcase 2c: y1 > yh
        subcase2c = (y1_c2 > yh_c2)
        if np.any(subcase2c):
            idx = np.where(inside_mask)[0][np.where(case2_mask)[0][subcase2c]]
            ans[idx] = (chord(x_c2[subcase2c], y0_c2[subcase2c], yh_c2[subcase2c]) +
                        arc(x_c2[subcase2c], yh_c2[subcase2c], y1_c2[subcase2c], r))

    # Case 3: y0 >= yh
    case3_mask = (y0i >= yh)
    if np.any(case3_mask):
        y0_c3 = y0i[case3_mask]
        y1_c3 = y1i[case3_mask]
        x_c3 = xi[case3_mask]
        yh_c3 = yh[case3_mask]

        # Subcase 3a: y1 <= -yh
        subcase3a = (y1_c3 <= -yh_c3)
        if np.any(subcase3a):
            idx = np.where(inside_mask)[0][np.where(case3_mask)[0][subcase3a]]
            ans[idx] = (arc(x_c3[subcase3a], y0_c3[subcase3a], yh_c3[subcase3a], r) +
                        chord(x_c3[subcase3a], yh_c3[subcase3a], -yh_c3[subcase3a]) +
                        arc(x_c3[subcase3a], -yh_c3[subcase3a], y1_c3[subcase3a], r))

        # Subcase 3b: -yh < y1 <= yh
        subcase3b = (-yh_c3 < y1_c3) & (y1_c3 <= yh_c3)
        if np.any(subcase3b):
            idx = np.where(inside_mask)[0][np.where(case3_mask)[0][subcase3b]]
            ans[idx] = (arc(x_c3[subcase3b], y0_c3[subcase3b], yh_c3[subcase3b], r) +
                        chord(x_c3[subcase3b], yh_c3[subcase3b], y1_c3[subcase3b]))

        # Subcase 3c: y1 > yh
        subcase3c = (y1_c3 > yh_c3)
        if np.any(subcase3c):
            idx = np.where(inside_mask)[0][np.where(case3_mask)[0][subcase3c]]
            ans[idx] = arc(x_c3[subcase3c], y0_c3[subcase3c], y1_c3[subcase3c], r)

    return ans


def intarea(xc: float, yc: float, r: float,
            x0: Union[float, np.ndarray], x1: Union[float, np.ndarray],
            y0: Union[float, np.ndarray], y1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute area of overlap between circle and rectangle.

    Parameters:
    -----------
    xc, yc : float
        Center of circle
    r : float
        Radius of circle
    x0, y0 : float or array
        One corner of rectangle
    x1, y1 : float or array
        Opposite corner of rectangle

    Returns:
    --------
    area : float or array
        Overlap area
    """
    # Shift so circle is at origin
    x0_shifted = x0 - xc
    y0_shifted = y0 - yc
    x1_shifted = x1 - xc
    y1_shifted = y1 - yc

    return (oneside(x1_shifted, y0_shifted, y1_shifted, r) +
            oneside(y1_shifted, -x1_shifted, -x0_shifted, r) +
            oneside(-x0_shifted, -y1_shifted, -y0_shifted, r) +
            oneside(-y0_shifted, x0_shifted, x1_shifted, r))


def pixwt(xc: float, yc: float, r: float,
          x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute fraction of unit pixel interior to circle.

    The circle has radius r and is centered at (xc, yc). The center of
    the unit pixel (length of sides = 1) is at (x, y).

    Parameters:
    -----------
    xc, yc : float
        Center of circle
    r : float
        Radius of circle
    x, y : float or array
        Center(s) of unit pixel(s)

    Returns:
    --------
    fraction : float or array
        Fraction of pixel area inside circle
    """
    return intarea(xc, yc, r, x - 0.5, x + 0.5, y - 0.5, y + 0.5)

def meanclip(data: np.ndarray,
             clipsig: float = 3.0,
             maxiter: int = 5,
             converge_num: float = 0.02,
             verbose: bool = False,
             double: bool = False) -> Tuple[float, float, np.ndarray]:
    """
    Computes an iteratively sigma-clipped mean on a data set.

    Clipping is done about median, but mean is returned.

    Parameters:
    -----------
    data : np.ndarray
        Input data array
    clipsig : float, optional
        Number of sigma at which to clip (default: 3.0)
    maxiter : int, optional
        Maximum number of iterations (default: 5)
    converge_num : float, optional
        Convergence criterion - stop if fraction of rejected pixels
        is less than this value (default: 0.02)
    verbose : bool, optional
        Print diagnostic messages
    double : bool, optional
        Use double precision for all computations

    Returns:
    --------
    mean : float
        Sigma-clipped mean
    sigma : float
        Standard deviation of remaining pixels
    subs : np.ndarray
        Indices of pixels used in final computation
    """

    if len(data) == 0:
        raise ValueError("Input data array is empty")

    # Set precision
    dtype = np.float64 if double else np.float32

    # Initialize with finite values
    subs = np.where(np.isfinite(data))[0]
    if len(subs) == 0:
        raise ValueError("No finite values in input data")

    iter_count = 0

    while True:
        skpix = data[subs]
        iter_count += 1
        lastct = len(subs)

        # Calculate median and moments
        medval = np.median(skpix)
        mean_val = np.mean(skpix, dtype=dtype)
        var_val = np.var(skpix, dtype=dtype, ddof=1)
        sig = np.sqrt(var_val)

        # Find pixels within clipsig*sigma of median
        deviation = np.abs(skpix - medval)
        wsm = np.where(deviation < clipsig * sig)[0]
        ct = len(wsm)

        if ct > 0:
            subs = subs[wsm]
        else:
            # No pixels left after clipping
            break

        # Check convergence criteria
        if ct == 0:
            break

        convergence = float(abs(ct - lastct)) / lastct
        if convergence <= converge_num or iter_count >= maxiter:
            break

    # Final calculation
    if len(subs) > 0:
        final_data = data[subs]
        mean_final = np.mean(final_data, dtype=dtype)
        var_final = np.var(final_data, dtype=dtype, ddof=1)
        sigma_final = np.sqrt(var_final)
    else:
        mean_final = np.nan
        sigma_final = np.nan
        subs = np.array([], dtype=int)

    if verbose:
        print(f"MEANCLIP: {clipsig}-sigma clipped mean")
        print(f"MEANCLIP: Mean computed in {iter_count} iterations")
        print(f"MEANCLIP: Mean = {mean_final}, sigma = {sigma_final}")
        print(f"MEANCLIP: {len(subs)} pixels used")

    return mean_final, sigma_final, subs


import numpy as np
import warnings
from typing import Optional, Union, Tuple, List


def aper(image: np.ndarray,
         xc: Union[float, np.ndarray],
         yc: Union[float, np.ndarray],
         mags: Optional[np.ndarray] = None,
         errap: Optional[np.ndarray] = None,
         sky: Optional[np.ndarray] = None,
         skyerr: Optional[np.ndarray] = None,
         phpadu: float = 1.0,
         apr: Optional[np.ndarray] = None,
         skyrad: Optional[np.ndarray] = None,
         badpix: Optional[np.ndarray] = None,
         setskyval: Optional[Union[float, np.ndarray]] = None,
         print_results: Union[bool, str] = False,
         silent: bool = False,
         flux: bool = False,
         exact: bool = False,
         nan: bool = False,
         readnoise: float = 0.0,
         meanback: bool = False,
         clipsig: float = 3.0,
         maxiter: int = 5,
         converge_num: float = 0.02,
         minsky: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute concentric aperture photometry (adapted from DAOPHOT)

    APER can compute photometry in several user-specified aperture radii.
    A separate sky value is computed for each source using specified inner
    and outer sky radii.

    Parameters:
    -----------
    image : np.ndarray
        Input image array (2D)
    xc : float or np.ndarray
        X coordinates of sources
    yc : float or np.ndarray
        Y coordinates of sources
    phpadu : float, optional
        Photons per Analog Digital Units (default: 1.0)
    apr : np.ndarray, optional
        Vector of up to 12 photometry aperture radii
    skyrad : np.ndarray, optional
        Two element array [inner_radius, outer_radius] for sky annulus
    badpix : np.ndarray, optional
        Two element array [min_good, max_good] pixel values
    setskyval : float or np.ndarray, optional
        Force sky to specified value(s)
    print_results : bool or str, optional
        Print results to file if string filename given
    silent : bool, optional
        Suppress terminal output
    flux : bool, optional
        Return results in flux units instead of magnitudes
    exact : bool, optional
        Use exact pixel intersection calculations
    nan : bool, optional
        Check for NaN values in image
    readnoise : float, optional
        Read noise value
    meanback : bool, optional
        Use clipped mean for background instead of mode
    clipsig : float, optional
        Sigma clipping value for background (default: 3.0)
    maxiter : int, optional
        Maximum iterations for background clipping (default: 5)
    converge_num : float, optional
        Convergence criterion for background clipping (default: 0.02)
    minsky : int, optional
        Minimum number of sky pixels required (default: 20)

    Returns:
    --------
    mags : np.ndarray
        NAPER by NSTAR array of magnitudes/fluxes
    errap : np.ndarray
        NAPER by NSTAR array of errors
    sky : np.ndarray
        NSTAR array of sky values
    skyerr : np.ndarray
        NSTAR array of sky errors
    """

    # Input validation
    if image.ndim != 2:
        raise ValueError("Image array must be 2-dimensional")

    nrow, ncol = image.shape

    # Convert inputs to arrays
    xc = np.atleast_1d(xc)
    yc = np.atleast_1d(yc)
    nstars = min(len(xc), len(yc))

    # Set default parameters
    maxsky = 10000  # Maximum pixels in sky annulus

    # Handle bad pixel specification
    if not nan:
        if badpix is None:
            badpix = np.array([0, 0])
        chk_badpix = badpix[0] < badpix[1]
    else:
        chk_badpix = False

    # Handle aperture radii
    if apr is None:
        # Default aperture radii - in real implementation would be interactive
        apr = np.array([3.0, 5.0, 7.0])
    apr = np.atleast_1d(apr)
    naper = len(apr)

    # Handle sky radii
    if setskyval is not None:
        if np.isscalar(setskyval):
            setskyval = np.array([setskyval, 0., 1.])
        elif len(setskyval) == 1:
            setskyval = np.array([setskyval[0], 0., 1.])
        elif len(setskyval) != 3:
            raise ValueError("SETSKYVAL must contain 1 or 3 elements")
        skyrad = np.array([0., np.max(apr) + 1])
    else:
        if skyrad is None:
            # Default sky radii - in real implementation would be interactive
            skyrad = np.array([np.max(apr) + 2, np.max(apr) + 5])
        skyrad = np.array(skyrad, dtype=float)

    # Initialize output arrays
    mags_out = np.full((naper, nstars), np.nan if flux else 99.999)
    errap_out = np.full((naper, nstars), np.nan if flux else 9.999)
    sky_out = np.zeros(nstars)
    skyerr_out = np.zeros(nstars)

    # Calculate aperture areas
    area = np.pi * apr ** 2

    # Set up exact photometry parameters if needed
    if exact:
        bigrad = apr + 0.5
        smallrad = apr / np.sqrt(2) - 0.5

    # Set up sky annulus parameters
    if setskyval is None:
        rinsq = max(skyrad[0], 0.0) ** 2
        routsq = skyrad[1] ** 2

    # Set up output file if requested
    if print_results:
        if isinstance(print_results, str):
            filename = print_results
        else:
            filename = 'aper.prt'

        # In real implementation, would write header to file
        if not silent:
            print(f"Results will be written to file {filename}")

    # Print header
    if not silent:
        if flux:
            print(f"\n{'Star':>4} {'X':>7} {'Y':>7} {'Sky':>6}        Fluxes")
        else:
            print(f"\n{'Star':>4} {'X':>7} {'Y':>7} {'Sky':>6}        Magnitudes")

    # Compute limits of submatrix for each star
    lx = np.maximum(np.floor(xc - skyrad[1]).astype(int), 0)
    ux = np.minimum(np.floor(xc + skyrad[1]).astype(int), ncol - 1)
    nx = ux - lx + 1
    ly = np.maximum(np.floor(yc - skyrad[1]).astype(int), 0)
    uy = np.minimum(np.floor(yc + skyrad[1]).astype(int), nrow - 1)
    ny = uy - ly + 1

    # Star coordinates in subarray
    dx = xc - lx
    dy = yc - ly

    # Check for stars too close to edge
    edge = np.minimum.reduce([dx - 0.5, nx + 0.5 - dx, dy - 0.5, ny + 0.5 - dy])
    badstar = ((xc < 0.5) | (xc > ncol - 1.5) |
               (yc < 0.5) | (yc > nrow - 1.5))

    # Set bad values
    if flux:
        badval = np.nan
        baderr = np.nan
    else:
        badval = 99.999
        baderr = 9.999

    # Main loop over stars
    for i in range(nstars):
        apmag = np.full(naper, badval)
        magerr = np.full(naper, baderr)
        skymod = 0.0
        skysig = 0.0
        skyskw = 0.0

        if badstar[i]:
            if not silent:
                print(f"Star {i}: too close to edge")
            sky_out[i] = skymod
            skyerr_out[i] = skysig
            mags_out[:, i] = apmag
            errap_out[:, i] = magerr
            continue

        # Extract subarray
        rotbuf = image[ly[i]:uy[i] + 1, lx[i]:ux[i] + 1]

        # Create distance arrays
        y_indices = np.arange(ny[i])  # y first
        x_indices = np.arange(nx[i])  # x second
        dxsq = (x_indices - dx[i]) ** 2
        dysq = (y_indices - dy[i]) ** 2
        rsq = np.zeros((ny[i], nx[i]))
        for ii in range(ny[i]):
            rsq[ii, :] = dxsq + (ii - dy[i]) ** 2

        if exact:
            x1 = np.abs(xx - dx[i])
            y1 = np.abs(yy - dy[i])
        else:
            r = np.sqrt(rsq) - 0.5

        # Compute sky value if not set
        if setskyval is None:
            # Select sky pixels
            skypix = (rsq >= rinsq) & (rsq <= routsq)

            if nan:
                skypix = skypix & np.isfinite(rotbuf)
            elif chk_badpix:
                skypix = skypix & (rotbuf > badpix[0]) & (rotbuf < badpix[1])

            sky_indices = np.where(skypix)
            nsky = min(len(sky_indices[0]), maxsky)

            if nsky < minsky:
                if not silent:
                    warnings.warn(f"Star {i}: Not enough sky pixels ({nsky} < {minsky})")
                sky_out[i] = skymod
                skyerr_out[i] = skysig
                mags_out[:, i] = apmag
                errap_out[:, i] = magerr
                continue

            skybuf = rotbuf[sky_indices[0][:nsky], sky_indices[1][:nsky]]

            if meanback:
                # Use clipped mean
                skymod, skysig, _ = meanclip(skybuf, clipsig=clipsig, maxiter=maxiter,
                                             converge_num=converge_num, verbose=False)
                skyskw = 0.0
            else:
                # Use MMM for robust sky estimation
                skymod, skysig, skyskw, _ = mmm(skybuf, readnoise=readnoise,
                                                minsky=minsky, silent=silent)

            if skysig < 0.0:
                if not silent:
                    warnings.warn(f"Star {i}: Could not determine sky")
                sky_out[i] = skymod
                skyerr_out[i] = skysig
                mags_out[:, i] = apmag
                errap_out[:, i] = magerr
                continue

            skyvar = skysig ** 2
            sigsq = skyvar / nsky
            skysig = min(skysig, 999.99)
            skyskw = np.clip(skyskw, -99, 999.9)

        else:
            skymod = setskyval[0]
            skysig = setskyval[1]
            nsky = setskyval[2]
            skyvar = skysig ** 2
            sigsq = skyvar / nsky
            skyskw = 0.0

        # Compute photometry for each aperture
        for k in range(naper):
            if edge[i] >= apr[k]:  # Aperture fits within image

                if exact:
                    # Exact photometry using PIXWT
                    mask = np.zeros((nx[i], ny[i]))
                    good = (x1 < smallrad[k]) & (y1 < smallrad[k])
                    mask[good] = 1.0
                    bad = (x1 > bigrad[k]) | (y1 > bigrad[k])
                    mask[bad] = -1.0

                    # Calculate fractional pixels using PIXWT
                    fract_pixels = mask == 0.0
                    if np.any(fract_pixels):
                        xx_fract = xx[fract_pixels]
                        yy_fract = yy[fract_pixels]
                        weights = pixwt(dx[i], dy[i], apr[k], xx_fract, yy_fract)
                        mask[fract_pixels] = np.maximum(weights, 0.0)

                    thisap = mask > 0.0
                    thisapd = rotbuf[thisap]
                    fractn = mask[thisap]

                else:
                    # Approximate photometry
                    thisap = r < apr[k]
                    thisapd = rotbuf[thisap]
                    thisapr = r[thisap]
                    fractn = np.clip(apr[k] - thisapr, 0.0, 1.0)

                    # Normalize fractional pixels
                    full = fractn == 1.0
                    nfull = np.sum(full)
                    if np.sum(~full) > 0:
                        factor = (area[k] - nfull) / np.sum(fractn[~full])
                        fractn[~full] *= factor

                # Check for bad pixels in aperture
                if nan:
                    badflux = not np.all(np.isfinite(thisapd))
                elif chk_badpix:
                    badflux = (np.min(thisapd) <= badpix[0]) or (np.max(thisapd) >= badpix[1])
                else:
                    badflux = False

                if not badflux:
                    apmag[k] = np.sum(thisapd * fractn)

        # Process valid apertures
        if flux:
            good = np.isfinite(apmag)
        else:
            good = np.abs(apmag - badval) > 0.01

        if np.any(good):
            # Subtract sky
            apmag[good] -= skymod * area[good]

            # Calculate errors
            error1 = area[good] * skyvar  # Sky scatter
            error2 = np.maximum(apmag[good], 0) / phpadu  # Photon noise
            error3 = sigsq * area[good] ** 2  # Sky uncertainty
            magerr[good] = np.sqrt(error1 + error2 + error3)

            if not flux:
                # Convert to magnitudes
                flux_good = apmag[good] > 0.0
                if np.any(flux_good):
                    good_flux_idx = np.where(good)[0][flux_good]
                    magerr[good_flux_idx] = 1.0857 * magerr[good_flux_idx] / apmag[good_flux_idx]
                    apmag[good_flux_idx] = 25.0 - 2.5 * np.log10(apmag[good_flux_idx])

        # Store results
        sky_out[i] = skymod
        skyerr_out[i] = skysig
        mags_out[:, i] = apmag
        errap_out[:, i] = magerr

        # Print results
        if not silent:
            mag_strings = []
            for j in range(naper):
                if flux:
                    mag_strings.append(f"{apmag[j]:8.1f}±{magerr[j]:7.1f}")
                else:
                    mag_strings.append(f"{apmag[j]:9.3f}±{magerr[j]:5.3f}")

            print(f"{i:4d} {xc[i]:7.2f} {yc[i]:7.2f} {skymod:6.2f} {' '.join(mag_strings)}")

    return mags_out, errap_out, sky_out, skyerr_out