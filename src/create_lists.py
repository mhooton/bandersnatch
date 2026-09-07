#!/usr/bin/env python3
"""Sort a night's raw frames into per-type list files.

Originally written for the NGTS zero-level pipeline by Philipp Eigmueller,
February 2014; converted to Python 3 and since rewritten around the
:mod:`instrument` abstraction.

All knowledge of how to find frames, what type each one is, which filter it
used and whether a flat was taken at dusk or dawn now lives in the instrument
configuration.  This module only groups the results and writes the lists.

Each frame's header is read exactly once, into a
:class:`~instrument.FrameMeta`, rather than the file being reopened by each
sorting pass in turn.
"""

import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from instrument import FrameType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def scan_directory(instrument, directory):
    """Read metadata for every raw frame in ``directory``.

    Returns ``{FrameType: [FrameMeta, ...]}``, sorted by filename within each
    type.  Files that cannot be read are logged and skipped, matching the old
    behaviour of carrying on rather than failing the night.
    """
    paths = instrument.find_raw_files(directory)
    logger.debug("Found %d candidate files in %s", len(paths), directory)

    grouped = defaultdict(list)
    unreadable = 0
    ignored = defaultdict(int)
    for path in paths:
        try:
            meta = instrument.read_meta(path)
        except Exception as exc:
            logger.warning("Unable to read FITS file %s: %s", path, exc)
            unreadable += 1
            continue
        if meta.frame_type is FrameType.IGNORE:
            ignored[meta.object] += 1
            continue
        grouped[meta.frame_type].append(meta)

    for obj, count in sorted(ignored.items()):
        logger.info("Ignoring %d frame(s) with OBJECT=%r "
                    "(no classification rule matches)", count, obj)
    if unreadable:
        logger.warning("%d file(s) could not be read and were skipped", unreadable)
    return grouped


def log_counts(grouped):
    logger.info("Image classification results:")
    for ftype in (FrameType.BIAS, FrameType.DARK, FrameType.FLAT,
                  FrameType.FRINGE, FrameType.SCIENCE):
        logger.info("  %-8s images: %d", ftype.value.capitalize(),
                    len(grouped.get(ftype, [])))


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------

def group_flats(instrument, metas):
    """``{filter: {session: [FrameMeta, ...]}}`` for dusk, dawn, dome, unknown."""
    out = defaultdict(lambda: defaultdict(list))
    for meta in metas:
        out[meta.filter][instrument.flat_session(meta)].append(meta)
    return out


def group_by_filter(metas):
    out = defaultdict(list)
    for meta in metas:
        out[meta.filter].append(meta)
    return out


def group_science(metas):
    """``({target: [FrameMeta, ...]}, {target: filter})``."""
    by_target = defaultdict(list)
    filters = {}
    for meta in metas:
        by_target[meta.object].append(meta)
        filters.setdefault(meta.object, meta.filter)
    return by_target, filters


def flat_sessions_wanted(instrument):
    """Which flat sessions the instrument wants combined."""
    allowed = instrument.flat_sources
    if not allowed:
        return ("dusk", "dawn", "dome", "unknown")
    wanted = []
    for source in allowed:
        if source == "sky":
            wanted.extend(["dusk", "dawn", "unknown"])
        elif source == "dome":
            wanted.append("dome")
        else:
            wanted.append(source)
    return tuple(dict.fromkeys(wanted))


def _flat_counts(sessions, wanted):
    return {s: len(sessions.get(s, [])) for s in wanted}


def needs_more_flats(sessions, wanted):
    """The historical sufficiency rule, applied to the wanted sessions only.

    At least two frames in one session for robust star rejection, and at least
    four in total.
    """
    counts = _flat_counts(sessions, wanted)
    total = sum(counts.values())
    return not (any(n >= 2 for n in counts.values()) and total >= 4)


# --------------------------------------------------------------------------
# previous-night search
# --------------------------------------------------------------------------

def _previous_dir(instrument, topdir, date_str, days_back, raw_dir_template=None):
    prev = (datetime.strptime(str(date_str), "%Y%m%d")
            - timedelta(days=days_back)).strftime("%Y%m%d")
    return prev, instrument.raw_dir(topdir, prev, template=raw_dir_template)


def search_previous_nights(instrument, topdir, date_str, want, is_enough,
                           max_days, raw_dir_template=None, merge=None):
    """Walk back night by night until ``is_enough(collected)`` is true.

    ``want`` names what is being looked for, for logging only.  ``merge`` is
    called with each night's grouped metadata and should accumulate into the
    caller's structure; it returns the current collection for ``is_enough``.
    """
    collected = None
    for days_back in range(1, max_days + 1):
        prev_date, prev_dir = _previous_dir(
            instrument, topdir, date_str, days_back, raw_dir_template)
        if not Path(prev_dir).exists():
            continue
        logger.debug("Searching %s for %s (%d night(s) back)",
                     prev_dir, want, days_back)
        try:
            grouped = scan_directory(instrument, prev_dir)
        except Exception as exc:
            logger.error("Error scanning %s: %s", prev_dir, exc)
            continue
        collected = merge(grouped, prev_date)
        if is_enough(collected):
            logger.info("Found sufficient %s after searching back to %s",
                        want, prev_date)
            return collected
    logger.warning("Still insufficient %s after searching back %d night(s)",
                   want, max_days)
    return collected


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def replace_spaces_with_double_hyphens(text):
    """Replace spaces with double hyphens in a string."""
    return text.replace(' ', '--')


def write_liste(liste, filename, outdir):
    """Write one list file into ``{outdir}/calib``."""
    logger.debug("Writing list file: %s to %s", filename, outdir)
    filename = filename.replace(' ', '--')
    reduction_dir = os.path.join(outdir, "calib")
    os.makedirs(reduction_dir, exist_ok=True)
    output = os.path.join(reduction_dir, filename)
    try:
        with open(output, 'w') as f:
            for item in liste:
                f.write(f"{item}\n")
        logger.info("Created list file %s with %d items", output, len(liste))
    except Exception as e:
        logger.error("Error writing file %s: %s", output, e)
        raise
    return 0


def _paths(metas):
    return [str(m.path) for m in metas]


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def create_lists(instrument, rawdir, outdir, run, topdir=None,
                 discard_n_first_science=None, raw_dir_template=None,
                 max_search_days=30):
    """Sort a night into list files under ``{outdir}/calib``.

    Which calibration types are looked for follows the instrument's reduction
    recipe: a night with no ``dark`` step is not searched for darks, and only
    an instrument whose recipe includes ``fringe`` gets a fringe list.
    """
    logger.info("Creating file lists from %s", rawdir)
    grouped = scan_directory(instrument, rawdir)
    log_counts(grouped)

    date_str = os.path.basename(str(rawdir).rstrip(os.sep))
    if not (len(date_str) == 8 and date_str.isdigit()):
        for part in Path(rawdir).parts[::-1]:
            if len(part) == 8 and part.isdigit():
                date_str = part
                break
        else:
            date_str = time.strftime("%Y%m%d")
            logger.warning("No date found in %s, using today: %s", rawdir, date_str)
    logger.debug("Observation date for calibration search: %s", date_str)

    topdir = topdir if topdir is not None else Path(rawdir).parent
    steps = instrument.steps

    bias = list(grouped.get(FrameType.BIAS, []))
    dark = list(grouped.get(FrameType.DARK, []))
    flats = group_flats(instrument, grouped.get(FrameType.FLAT, []))
    fringe = group_by_filter(grouped.get(FrameType.FRINGE, []))
    science, science_filters = group_science(grouped.get(FrameType.SCIENCE, []))

    wanted_sessions = flat_sessions_wanted(instrument)
    if instrument.flat_sources:
        logger.info("Flat sources restricted to %s (sessions: %s)",
                    instrument.flat_sources, list(wanted_sessions))

    # -- borrow calibrations from previous nights where needed --------------

    if "bias" in steps and len(bias) < 2:
        logger.warning("Insufficient bias frames (%d found), searching previous nights",
                       len(bias))

        def merge_bias(g, _date):
            bias.extend(g.get(FrameType.BIAS, []))
            return bias

        search_previous_nights(
            instrument, topdir, date_str, "bias frames",
            lambda c: len(c) >= 2, max_search_days, raw_dir_template, merge_bias)
        logger.info("Using %d bias frames", len(bias))

    if "dark" in steps and not dark:
        logger.warning("No dark frames found, searching previous nights")

        def merge_dark(g, _date):
            dark.extend(g.get(FrameType.DARK, []))
            return dark

        search_previous_nights(
            instrument, topdir, date_str, "dark frames",
            lambda c: len(c) > 0, max_search_days, raw_dir_template, merge_dark)
        logger.info("Using %d dark frames", len(dark))
    elif "dark" not in steps:
        logger.info("Reduction recipe has no dark step; not searching for darks")

    for filt in sorted(set(science_filters.values())):
        sessions = flats.setdefault(filt, defaultdict(list))
        if not needs_more_flats(sessions, wanted_sessions):
            continue
        counts = _flat_counts(sessions, wanted_sessions)
        logger.warning("Insufficient flats for filter %s (%s), searching previous nights",
                       filt, counts)

        def merge_flat(g, _date, _filt=filt, _sessions=sessions):
            for meta in g.get(FrameType.FLAT, []):
                if meta.filter != _filt:
                    continue
                _sessions[instrument.flat_session(meta)].append(meta)
            return _sessions

        search_previous_nights(
            instrument, topdir, date_str, f"flats in filter {filt}",
            lambda c: not needs_more_flats(c, wanted_sessions),
            max_search_days, raw_dir_template, merge_flat)

    # -- write ---------------------------------------------------------------

    if "bias" in steps or bias:
        write_liste(_paths(bias), f"{run}_bias.list", outdir)
    if "dark" in steps or dark:
        write_liste(_paths(dark), f"{run}_dark.list", outdir)

    logger.info("Final flat frame counts:")
    for filt in sorted(flats):
        sessions = flats[filt]
        counts = _flat_counts(sessions, wanted_sessions)
        total = sum(counts.values())
        logger.info("  Filter %s: %s (%d usable)", filt, counts, total)

        usable = {s: sessions.get(s, []) for s in wanted_sessions}
        multi = [s for s, v in usable.items() if len(v) >= 2]
        if len(multi) >= 2:
            for session in multi:
                write_liste(_paths(usable[session]),
                            f"{run}_flat_{filt}_{session}.list", outdir)
            logger.info("Created per-session flat lists for filter %s: %s",
                        filt, multi)
        elif total > 0:
            combined = [m for s in wanted_sessions for m in usable.get(s, [])]
            write_liste(_paths(combined), f"{run}_flat_{filt}.list", outdir)
            if total < 4:
                logger.warning("Very few flat frames for filter %s (%d)", filt, total)
            else:
                logger.info("Created combined flat list for filter %s "
                            "(insufficient session separation)", filt)

    if "fringe" in steps:
        if not fringe:
            logger.warning("Reduction recipe includes 'fringe' but no fringe "
                           "frames were classified in %s", rawdir)
        for filt, metas in sorted(fringe.items()):
            write_liste(_paths(metas), f"{run}_fringe_{filt}.list", outdir)
            logger.info("Created fringe list for filter %s (%d frames)",
                        filt, len(metas))
    elif fringe:
        logger.info("Found %d fringe frame(s) but the recipe has no fringe step",
                    sum(len(v) for v in fringe.values()))

    for target in sorted(science):
        metas = science[target]
        if discard_n_first_science:
            n = int(discard_n_first_science)
            if n > 0:
                metas = metas[n:]
                logger.info("Discarded first %d science frames for target %s",
                            n, target)
        if len(metas) > 4:
            write_liste(_paths(metas), f"{run}_image_{target}.list", outdir)
        else:
            logger.warning("Too few science frames for %s (%d after discarding)",
                           target, len(metas))

    logger.info("File list creation completed successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if len(sys.argv) < 5:
        logger.error("Usage: python create_lists.py <instrument> <config_dir> "
                     "<raw_directory> <output_directory> <run> "
                     "[discard_n_first_science]")
        sys.exit(1)
    from instrument import load_instrument
    inst = load_instrument(sys.argv[1], sys.argv[2])
    discard = int(sys.argv[6]) if len(sys.argv) > 6 else None
    create_lists(inst, sys.argv[3], Path(sys.argv[4]), sys.argv[5],
                 discard_n_first_science=discard)
