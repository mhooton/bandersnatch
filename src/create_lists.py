#!/usr/bin/env python3

# Small python script to create list of images for NGTS Zero-Level-Pipeline
# Output goes to Reduction Module
# Philipp Eigmueller Feb 6 2014
# Converted to Python 3.10

import glob
import os
import time
import sys
import logging
# Tix is rarely used in Python 3, removing this import
# from Tix import IMAGE
from multiprocessing.dummy import Pool as ThreadPool
# from collections import defaultdict
from functools import partial
from datetime import datetime, timedelta
from astropy.io import fits

# No need for this in Python 3 as print function is standard
# from __future__ import print_function

# Set up logger for this module
logger = logging.getLogger(__name__)

DIRECTORY = os.getcwd()
DEBUG = 0


def get_liste(directory, ext):
    """Search for all image files in the directories"""
    liste = glob.glob(os.path.join(directory, f"*.{ext}"))
    # search for images inside folders in the image folder (such as Calibration or AutoFlat)
    add_liste = glob.glob(os.path.join(directory, "*", f"*.{ext}"))

    # if len(liste) == 0:
    #     new_ext = 'fts'
    #     liste = glob.glob(os.path.join(directory, f"*.{new_ext}"))
    #     add_liste = glob.glob(os.path.join(directory, "*", f"*.{new_ext}"))

    if len(add_liste) != 0:
        liste = liste + add_liste

    if DEBUG:
        logger.debug("get_liste search pattern: %s", os.path.join(directory, f"*.{ext}"))

    liste.sort()
    logger.debug("Found %d files with extension %s in %s", len(liste), ext, directory)
    return liste  # Parentheses unnecessary in Python 3 return


def classify(files, runnumber):
    pool = ThreadPool()
    fn = partial(classify_file, runnumber=runnumber)
    results = pool.map(fn, files)
    keys = ['science', 'dark', 'bias', 'flat']
    pool.close()

    return {key: sum([entry[key] for entry in results], []) for key in keys}


def classify_file(filename, runnumber):
    out = {}
    for field in ['science', 'dark', 'bias', 'flat']:
        out[field] = []

    try:
        with fits.open(filename) as hdulist:
            # header name is different for ngts (IMGTYPE) and trappist (IMAGETYP)
            try:
                imtype = hdulist[0].header['IMAGETYP']
                # naxis1 = hdulist[0].header['NAXIS1']
                # filter_val = hdulist[0].header['FILTER']  # Renamed to avoid shadowing built-in 'filter'
            except Exception:  # Better exception handling
                if filename.endswith(".fz"):
                    imtype = hdulist[1].header['IMAGETYP']
                    # naxis1 = hdulist[1].header['NAXIS1']
                    # filter_val = hdulist[1].header['FILTER']
                else:
                    raise  # Re-raise if it's not a .fz file

        # changed because there is inconsistency across the image types: dark frame, light image, bias frame, flat
        if 'LIGHT' in imtype.upper():  # Fixed str.upper() call
            out['science'].append(filename)
        elif 'DARK' in imtype.upper():
            out['dark'].append(filename)
        elif 'BIAS' in imtype.upper():
            out['bias'].append(filename)
        elif 'FLAT' in imtype.upper():
            out['flat'].append(filename)
    except Exception as e:  # Better exception handling
        logger.warning("Unable to classify FITS file %s: %s", filename, e)

    return out


def get_observation_time(filename):
    """Extract observation time from FITS header and classify as dawn or dusk"""
    try:
        with fits.open(filename) as hdulist:
            # Try different common time keywords
            time_keywords = ['DATE-OBS', 'UTC', 'UT', 'TIME-OBS']
            date_obs = None

            header = hdulist[0].header
            if filename.endswith(".fz"):
                header = hdulist[1].header

            for keyword in time_keywords:
                if keyword in header:
                    date_obs = header[keyword]
                    break

            if date_obs is None:
                logger.warning("No time information found in %s", filename)
                return 'unknown'

            # Parse the time string - handle various formats
            if isinstance(date_obs, str):
                # Common formats: '2025-04-29T08:45:12.123' or '2025-04-29 08:45:12'
                if 'T' in date_obs:
                    time_str = date_obs.split('T')[1]
                elif ' ' in date_obs:
                    time_str = date_obs.split(' ')[1]
                else:
                    # Assume it's just a time string
                    time_str = date_obs

                # Extract hour
                hour = int(time_str.split(':')[0])
            else:
                logger.warning("Unexpected time format in %s: %s", filename, date_obs)
                return 'unknown'

            # Classify as dawn or dusk
            # Dawn: 4 AM to 10 AM (04:00 - 10:00)
            # Dusk: 6 PM to 11 PM (18:00 - 23:00)
            # Also consider early morning hours (00:00 - 02:00) as dusk from previous evening
            if 4 <= hour <= 10:
                return 'dawn'
            elif 18 <= hour <= 23 or 0 <= hour <= 2:
                return 'dusk'
            else:
                logger.warning("Flat taken at unusual time %d:xx in %s", hour, filename)
                return 'unknown'

    except Exception as e:
        logger.error("Error extracting time from %s: %s", filename, e)
        return 'unknown'


def sort_liste(liste, logroot, runnumber=None):
    """Sort images for bias, darks, flats, and scientific frames"""
    classification = classify(liste, runnumber)
    biaslist = classification['bias']
    darklist = classification['dark']
    flatlist = classification['flat']
    sciencelist = classification['science']

    # Log image counts
    logger.info("Image classification results:")
    logger.info("  Bias images: %d", len(biaslist))
    logger.info("  Dark images: %d", len(darklist))
    logger.info("  Flat images: %d", len(flatlist))
    logger.info("  Science images: %d", len(sciencelist))

    return (biaslist, darklist, flatlist, sciencelist)


def sort_flatlist(liste):
    """Sort flat images based on filter and observation time (dawn/dusk)"""
    # Dictionary structure: {filter: {'dawn': [], 'dusk': [], 'unknown': []}}
    flatdict = {}
    filters = []

    for item in liste:
        try:
            with fits.open(item) as hdulist:
                header = hdulist[0].header
                if item.endswith(".fz"):
                    header = hdulist[1].header
                filter_val = header['FILTER']
        except Exception as e:
            logger.error("Error reading filter from %s: %s", item, e)
            continue

        # Get observation time classification
        time_class = get_observation_time(item)

        # Initialize filter entry if not exists
        if filter_val not in flatdict:
            filters.append(filter_val)
            flatdict[filter_val] = {'dawn': [], 'dusk': [], 'unknown': []}

        # Add to appropriate time category
        flatdict[filter_val][time_class].append(item)

    logger.debug("Sorted flat images by filter and time")
    return (filters, flatdict)


def sort_scilist(liste):
    # sort science images based on the object we're targeting
    fields = []
    scilists = []
    filters = []
    scifilt = {}

    for item in liste:
        with fits.open(item) as hdulist:
            field = hdulist[0].header['OBJECT']
            filter_val = hdulist[0].header['FILTER']  # Renamed

        if field not in fields:  # More Pythonic than using count()
            # for every object targeted add this object to the array 'fields'
            fields.append(field)
            # and add an empty entry to scilists for each new object
            scilists.append([])

        if filter_val not in filters:
            filters.append(filter_val)

        # replace this entry in scilists with the data from hdulist
        idx = fields.index(field)
        scilists[idx].append(item)
        scifilt[field] = filter_val

    logger.debug("Sorted science images by target field")
    return (fields, scilists, filters, scifilt)


def replace_spaces_with_double_hyphens(text):
    """Replace spaces with double hyphens in a string"""
    return text.replace(' ', '--')


def write_liste(liste, filename, outdir):
    """Write output files"""
    logger.debug("Writing list file: %s to %s", filename, outdir)

    # Replace spaces with double hyphens in filename
    filename = filename.replace(' ', '--')

    # Create the reduction directory inside the date directory
    reduction_dir = os.path.join(outdir, "calib")
    os.makedirs(reduction_dir, exist_ok=True)

    # Full path to the output file
    output = os.path.join(reduction_dir, filename)

    try:
        # Use with statement for file operations
        with open(output, 'w') as f:
            for item in liste:
                f.write(f"{item}\n")
        logger.info("Created list file %s with %d items", output, len(liste))
    except Exception as e:
        logger.error("Error writing file %s: %s", output, e)
        raise

    return 0


def write_log(logroot, runnumber, string, lognumber=2):
    if DEBUG:
        logger.debug("write_log called: %s %s %d", logroot, runnumber, lognumber)
    if lognumber == 1:
        logtype = 'short'
    elif lognumber == 2:
        logtype = 'long'

    # Commented out in original code
    # log_path = os.path.join(DIRECTORY, "logfiles", f"{logroot}_{logtype}_{runnumber:03d}.log")
    # with open(log_path, 'a') as f:
    #     f.write(string)

    if lognumber == 1:
        logger.info(string.strip())
    return 0


def check_log(logroot):
    runnumber = 1
    while os.path.exists(os.path.join(DIRECTORY, "logfiles", f"{logroot}_short_{runnumber:03d}.log")):
        runnumber += 1
    return runnumber


def write_logstart(directory, imageroot, ext, run, runnumber):
    if DEBUG:
        logger.debug("write_logstart called: %s %s %s %s %s", directory, imageroot, ext, run, runnumber)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Use f-strings for more readable string formatting
    string = "-------------------------------------------------\n"
    string += f"{timestamp:>20s} Creating lists of images for Zero Level pipeline\n "
    string += f"{timestamp:>20s} using the script createlists.py\n "
    string += f"{timestamp:>20s} \n "
    string += f"{timestamp:>20s} All {imageroot}*.{ext} files in the directory \n "
    string += f"{timestamp:>20s} {directory} \n "
    string += f"{timestamp:>20s} will be sorted.\n "
    string += f"{timestamp:>20s} \n "
    string += f"{timestamp:>20s} Working Directory is {DIRECTORY}\n "
    string += f"{timestamp:>20s} \n "
    string += " -----------------------------------------------\n "
    string += "# \n "

    write_log(run, runnumber, string, 1)
    write_log(run, runnumber, string, 2)
    return 0


def create_lists(rawdir, outdir, run, ext='fits', discard_n_first_science=None):
    logger.info("Creating file lists from %s", rawdir)
    logger.debug("Parameters: run=%s, ext=%s, discard_first=%s", run, ext, discard_n_first_science)

    # Get full list of images in rawdir
    liste = get_liste(rawdir, ext)

    if DEBUG:
        logger.debug("Main function file list: %s", liste)

    # Extract date from directory
    logger.debug("Raw directory: %s", rawdir)
    extracted_date = os.path.basename(rawdir)
    logger.debug("Extracted date string: %s", extracted_date)

    try:
        date = int(extracted_date)
        logger.debug("Converted date: %d", date)
    except ValueError as e:
        logger.warning("Date conversion error: %s", e)

    # separate into lists based on whether the image is bias/dark/flat/science
    biaslist, darklist, flatlist, sciencelist = sort_liste(liste, run)

    fnames, scilists, filters, scifilt = sort_scilist(sciencelist)
    filtnames, flatdict = sort_flatlist(flatlist)  # Now returns dict with dawn/dusk

    logger.info("Found filters in flat images: %s", filters)
    logger.info("Science image filters by target: %s", scifilt)

    # Log flat statistics
    for filt in filtnames:
        dawn_count = len(flatdict[filt]['dawn'])
        dusk_count = len(flatdict[filt]['dusk'])
        unknown_count = len(flatdict[filt]['unknown'])
        logger.debug("Filter %s: %d dawn, %d dusk, %d unknown flats",
                     filt, dawn_count, dusk_count, unknown_count)

    # Try to extract date from the directory structure
    try:
        # Your path format appears to be .../SPIRIT2/20250429/raw
        dir_parts = os.path.normpath(rawdir).split(os.sep)
        # Find the part that looks like a date (likely the second-to-last part)
        for i, part in enumerate(dir_parts):
            if len(part) == 8 and part.isdigit() and i < len(dir_parts) - 1:
                date = int(part)
                logger.debug("Found date in path: %d", date)
                break
        else:
            date = int(time.strftime("%Y%m%d"))
            logger.warning("No date found in path structure, using current date: %d", date)
    except Exception as e:
        logger.error("Error extracting date, using current date: %s", e)
        date = int(time.strftime("%Y%m%d"))

    # if there are no calibration images found on this night then look for them on previous nights
    if len(biaslist) < 2:
        logger.warning("Insufficient bias images (%d found), searching previous dates", len(biaslist))
        new_biaslist = []
        new_date = str(date)
        bcount = 0
        while len(new_biaslist) < 2:
            bcount = bcount + 1
            if bcount < 100:
                logger.debug("Searching for bias images on %s (attempt %d)", new_date, bcount)
            else:
                logger.warning("Stopped bias search after 100 attempts")
                break

            try:
                date_form = (datetime.strptime(new_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                # Use path manipulation instead of string replacement
                new_dir = os.path.join(os.path.dirname(rawdir), date_form)
                new_date = date_form

                if os.path.exists(new_dir):
                    logger.debug("Searching in directory: %s", new_dir)
                    liste = get_liste(new_dir, ext)
                    new_biaslist, new_darklist, new_flatlist, new_sciencelist = sort_liste(liste, run)
            except Exception as e:
                logger.error("Error processing previous date: %s", e)
                break

        biaslist = new_biaslist
        logger.info("Found %d bias images from previous dates", len(biaslist))

    # Similar pattern for darklist handling
    if len(darklist) == 0:
        logger.warning("No dark images found, searching previous dates")
        new_darklist = []
        new_date = str(date)
        dcount = 0
        while len(new_darklist) == 0:
            dcount = dcount + 1
            if dcount < 100:
                logger.debug("Searching for dark images on %s (attempt %d)", new_date, dcount)
            else:
                logger.warning("Stopped dark search after 100 attempts")
                break

            try:
                date_form = (datetime.strptime(new_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                new_dir = os.path.join(os.path.dirname(rawdir), date_form)
                new_date = date_form

                if os.path.exists(new_dir):
                    logger.debug("Searching in directory: %s", new_dir)
                    liste = get_liste(new_dir, ext)
                    new_biaslist2, new_darklist, new_flatlist, new_sciencelist = sort_liste(liste, run)
            except Exception as e:
                logger.error("Error processing previous date: %s", e)
                break

        darklist = new_darklist
        logger.info("Found %d dark images from previous dates", len(darklist))

    # Enhanced flat finding logic that maintains dawn/dusk separation
    # Find if there are any science images with filters which do not have sufficient flats
    for f in filters:
        logger.info("Searching for flats in filter %s", f)

        # Check if we have sufficient flats for this filter
        if f not in flatdict:
            flatdict[f] = {'dawn': [], 'dusk': [], 'unknown': []}

        dawn_count = len(flatdict[f]['dawn'])
        dusk_count = len(flatdict[f]['dusk'])
        total_count = dawn_count + dusk_count + len(flatdict[f]['unknown'])

        # We need at least 2 flats per session for robust star rejection
        # Ideally 4+ total flats
        needs_more_flats = (dawn_count < 2 and dusk_count < 2) or total_count < 4

        if needs_more_flats:
            logger.warning("Insufficient flats for filter %s (%d dawn, %d dusk), searching previous dates",
                           f, dawn_count, dusk_count)
            new_date = str(date)
            fcount = 0

            while needs_more_flats and fcount < 30:
                fcount = fcount + 1
                logger.debug("Flat search attempt %d for filter %s on %s", fcount, f, new_date)

                try:
                    date_form = (datetime.strptime(new_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                    new_dir = os.path.join(os.path.dirname(rawdir), date_form)
                    new_date = date_form

                    if os.path.exists(new_dir):
                        logger.debug("Searching in directory: %s", new_dir)
                        liste = get_liste(new_dir, ext)
                        new_biaslist, new_darklist, new_flatlist, new_sciencelist = sort_liste(liste, run)
                        new_filtnames, new_flatdict = sort_flatlist(new_flatlist)

                        logger.debug("Found filters in %s: %s", new_dir, new_filtnames)

                        # Merge flats from previous night
                        if f in new_flatdict:
                            logger.debug("Found filter %s in %s, merging flats", f, new_dir)
                            for time_class in ['dawn', 'dusk', 'unknown']:
                                before_count = len(flatdict[f][time_class])
                                flatdict[f][time_class].extend(new_flatdict[f][time_class])
                                after_count = len(flatdict[f][time_class])
                                if after_count > before_count:
                                    logger.debug("Added %d %s flats for filter %s",
                                                 after_count - before_count, time_class, f)

                            # Update counts
                            dawn_count = len(flatdict[f]['dawn'])
                            dusk_count = len(flatdict[f]['dusk'])
                            total_count = dawn_count + dusk_count + len(flatdict[f]['unknown'])
                            needs_more_flats = (dawn_count < 2 and dusk_count < 2) or total_count < 4
                            logger.debug("Updated counts for %s: %d dawn, %d dusk, %d total",
                                         f, dawn_count, dusk_count, total_count)

                except Exception as e:
                    logger.error("Error processing previous date: %s", e)
                    break

            if fcount >= 30:
                logger.warning("Insufficient flat images found for %s after 30 days. Final: %d dawn, %d dusk",
                               f, dawn_count, dusk_count)

    # Log final flat counts
    logger.info("Final flat image counts:")
    for filt in flatdict.keys():
        dawn_count = len(flatdict[filt]['dawn'])
        dusk_count = len(flatdict[filt]['dusk'])
        unknown_count = len(flatdict[filt]['unknown'])
        total = dawn_count + dusk_count + unknown_count
        logger.info("  Filter %s: %d dawn, %d dusk, %d unknown (%d total)",
                    filt, dawn_count, dusk_count, unknown_count, total)

    # Write output files
    logger.debug("Writing output files to %s", outdir)

    write_liste(biaslist, f"{run}_bias.list", outdir)
    write_liste(darklist, f"{run}_dark.list", outdir)

    # Write flat image lists - now separated by dawn/dusk
    max_flat = 0
    for filt in flatdict.keys():
        dawn_flats = flatdict[filt]['dawn']
        dusk_flats = flatdict[filt]['dusk']
        unknown_flats = flatdict[filt]['unknown']

        total_flats = len(dawn_flats) + len(dusk_flats) + len(unknown_flats)

        # Write dawn/dusk lists if we have sufficient separation
        if len(dawn_flats) >= 2 and len(dusk_flats) >= 2:
            write_liste(dawn_flats, f"{run}_flat_{filt}_dawn.list", outdir)
            write_liste(dusk_flats, f"{run}_flat_{filt}_dusk.list", outdir)
            logger.info("Created dawn/dusk flat lists for filter %s", filt)
        elif total_flats >= 4:
            # Fall back to combined list if we don't have good dawn/dusk separation
            all_flats = dawn_flats + dusk_flats + unknown_flats
            write_liste(all_flats, f"{run}_flat_{filt}.list", outdir)
            logger.info("Created combined flat list for filter %s (insufficient dawn/dusk separation)", filt)
        else:
            logger.warning("Very few flat images for filter %s (%d total)", filt, total_flats)
            if total_flats > 0:
                all_flats = dawn_flats + dusk_flats + unknown_flats
                write_liste(all_flats, f"{run}_flat_{filt}.list", outdir)

        if total_flats > max_flat:
            max_flat = total_flats

    # Write science image lists
    for i, field in enumerate(fnames):
        # Apply discard_n_first_science if specified
        science_images = scilists[i]
        if discard_n_first_science is not None and isinstance(discard_n_first_science,
                                                              int) and discard_n_first_science > 0:
            science_images = science_images[discard_n_first_science:]
            logger.info("Discarded first %d science images for field %s", discard_n_first_science, field)

        # if there are less than 5 science images for a field then don't create list file
        if len(science_images) > 4:
            write_liste(science_images, f"{run}_image_{field}.list", outdir)
        else:
            logger.warning("Too few science images for %s (%d after discarding)", field, len(science_images))

    logger.info("File list creation completed successfully")


# When script is run directly
if __name__ == "__main__":
    # Set up basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if correct number of arguments are provided
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python create_lists.py <raw_directory> <output_directory> <run_name> [extension] [discard_n_first_science]")
        sys.exit(1)

    # Parse command line arguments
    rawdir = sys.argv[1]
    outdir = sys.argv[2]
    run = sys.argv[3]

    # Optional extension argument
    ext = 'fits'
    if len(sys.argv) > 4:
        ext = sys.argv[4]

    # Optional discard_n_first_science argument
    discard_n_first_science = None
    if len(sys.argv) > 5:
        try:
            discard_n_first_science = int(sys.argv[5])
        except ValueError:
            logger.warning("discard_n_first_science must be an integer. Using default (None).")

    # Run the main function
    create_lists(rawdir, outdir, run, ext, discard_n_first_science)