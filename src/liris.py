def runset_filter(filenames, runset_cut):
    """
    Filter filenames based on their RUNSET header value.

    Parameters:
    -----------
    filenames : list
        List of filenames to check
    runset_cut : float or int
        The minimum RUNSET value to accept

    Returns:
    --------
    list
        Filenames that have RUNSET values greater than runset_cut
    """

    n_filenames = len(filenames)
    accepted_filenames = []

    for file_loop in range(n_filenames):
        with fits.open(filenames[file_loop]) as hdul:
            header = hdul[0].header

            # Extract the RUNSET value from the header and compare with runset_cut
            runset_value = header.get('RUNSET', '0:0')  # Default to '0:0' if RUNSET not found
            runset_first_part = runset_value.split(':')[0]  # Get the part before the colon

            if float(runset_first_part) > runset_cut:
                accepted_filenames.append(filenames[file_loop])

    return accepted_filenames
