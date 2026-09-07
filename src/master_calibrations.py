"""Build master calibration frames: bias, dark, flat and fringe map.

Frames are read through the :class:`~instrument.Instrument`, so multi-extension
files, overscan and trimming are handled the same way here as in the science
reduction.  Each product is built from frames reduced exactly as far as the
step that will consume it: a master flat from frames taken through everything
before the flat step, a fringe map from frames taken through everything before
the fringe step.  A product built any other way would not match the frames it
is applied to.
"""

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from instrument import find_dated_entry
from reduction import (Reducer, combine_fringe_frames, highpass,
                       steps_before)
from utils import medabsdevclip

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_list(list_file):
    with open(list_file) as f:
        return [line.strip() for line in f if line.strip()]


def _load_if_exists(path):
    if Path(path).exists():
        with fits.open(path) as hdul:
            return hdul[0].data
    return None


def _calib_for_steps(outdir, run, steps, filter=None):
    """Load whichever master frames a truncated reduction chain needs."""
    calib = {}
    caldir = Path(outdir) / "calib"
    if "bias" in steps:
        calib["bias"] = _load_if_exists(caldir / f"{run}_master_bias.fits")
    if "dark" in steps:
        calib["dark"] = _load_if_exists(caldir / f"{run}_master_dark.fits")
    if "flat" in steps and filter is not None:
        calib["flat"] = _load_if_exists(caldir / f"{run}_master_flat_{filter}.fits")
    return calib


def make_reducer_upto(instrument, outdir, run, step, filter=None):
    """A :class:`~reduction.Reducer` running the chain that precedes ``step``."""
    steps = steps_before(instrument, step)
    return Reducer(instrument, _calib_for_steps(outdir, run, steps, filter),
                   steps=steps)


def calculate_readout_noise(instrument, filenames, reducer=None):
    """Read-out noise in electrons, from the difference of two bias frames."""
    if len(filenames) < 2:
        logger.warning("Need at least 2 bias frames for readout noise")
        return 0.0
    try:
        frames = []
        for name in filenames[:2]:
            frame = instrument.open_frame(name)
            data = (reducer.reduce(frame)[0] if reducer is not None
                    else instrument.trim(frame.data))
            frames.append(np.asarray(data, dtype=np.float64))
        diff = frames[0] - frames[1]
        adu_diff = np.std(diff)
        ron = adu_diff * instrument.gain() / np.sqrt(2)
        logger.info("Calculated readout noise: %.3f electrons", ron)
        return float(ron)
    except Exception as exc:
        logger.error("Failed to calculate readout noise: %s", exc)
        return 0.0


def copy_backup_calibration(backup_path, output_file, calib_type, filter=None):
    """Copy a pre-made calibration into this run's calib directory."""
    if not os.path.exists(backup_path):
        filter_str = f" for filter {filter}" if filter else ""
        logger.error("Backup %s file not found: %s%s",
                     calib_type, backup_path, filter_str)
        raise FileNotFoundError(
            f"Backup {calib_type} file not found: {backup_path}")
    os.makedirs(Path(output_file).parent, exist_ok=True)
    shutil.copy2(backup_path, output_file)
    logger.info("Copied backup %s to %s", calib_type, output_file)


# --------------------------------------------------------------------------
# dated backup configuration
# --------------------------------------------------------------------------

def _load_yaml(config_dir, name):
    path = Path(config_dir) / name
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.debug("Optional config not found: %s", path)
        return None
    except yaml.YAMLError as exc:
        logger.error("Error parsing %s: %s", path, exc)
        return None


def load_master_flats_config(config_dir):
    return _load_yaml(config_dir, "master_flats.yaml")


def load_fringe_maps_config(config_dir):
    return _load_yaml(config_dir, "fringe_maps.yaml")


def find_backup_flat_for_date(master_flats_config, filter_name, observation_date):
    entry = find_dated_entry(master_flats_config, [filter_name],
                             observation_date, root="filters")
    return entry["path"] if entry else None


def find_backup_fringe_map(fringe_maps_config, filter_name, detector_name,
                           observation_date):
    entry = find_dated_entry(fringe_maps_config, [filter_name, detector_name],
                            observation_date, root="maps")
    return entry["path"] if entry else None


# --------------------------------------------------------------------------
# fringe map
# --------------------------------------------------------------------------

FRINGE_PROVENANCE = ("INSTRUME", "DETECTOR", "FILTER")


def write_fringe_map(path, template, raw_template, instrument, filter_name,
                     n_frames):
    """Write a fringe map with the provenance needed to verify it on load."""
    hdu = fits.PrimaryHDU(np.asarray(template, dtype=np.float32))
    hdu.header["INSTRUME"] = instrument.name
    hdu.header["DETECTOR"] = instrument.detector.name
    hdu.header["FILTER"] = filter_name
    hdu.header["NFRAMES"] = (n_frames, "Dithered frames combined")
    if instrument.detector.trim:
        hdu.header["TRIMSEC"] = instrument.detector.trim
    hdu.header["HPSIGMA"] = (instrument.fringe.highpass_sigma,
                             "Large-scale removal sigma (px)")
    extra = fits.ImageHDU(np.asarray(raw_template, dtype=np.float32),
                          name="UNFILTERED")
    fits.HDUList([hdu, extra]).writeto(path, overwrite=True)
    logger.info("Fringe map written to %s (%d frames)", path, n_frames)


def load_fringe_map(path, instrument, filter_name, expected_shape=None):
    """Load a fringe map, refusing one built for a different configuration.

    A map from the wrong detector or filter would produce a plausible-looking
    but wrong correction, so a mismatch is an error rather than a warning.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header

    expected = {"INSTRUME": instrument.name,
                "DETECTOR": instrument.detector.name,
                "FILTER": filter_name}
    for key, want in expected.items():
        have = header.get(key)
        if have is None:
            logger.warning("Fringe map %s has no %s keyword; cannot verify "
                           "provenance", path, key)
            continue
        if str(have).strip() != str(want).strip():
            raise ValueError(
                f"Fringe map {path} was built for {key}={have!r} but this run "
                f"needs {want!r}. Applying it would give a wrong correction.")
    if expected_shape is not None and data.shape != tuple(expected_shape):
        raise ValueError(
            f"Fringe map {path} has shape {data.shape}, but this detector's "
            f"trimmed frames are {tuple(expected_shape)}")
    return data


def make_fringe_map(instrument, outdir, run, config, filter, config_dir=None):
    """Build, or fetch, the fringe map for one filter."""
    caldir = Path(outdir) / "calib"
    os.makedirs(caldir, exist_ok=True)
    output_file = caldir / f"{run}_fringe_map_{filter}.fits"
    calib_params = config.get("calibration_params", {}) or {}
    observation_date = config["instrument_settings"]["date"]
    config_dir = Path(config_dir or config.get("_config_dir", "."))

    def from_config():
        maps_cfg = load_fringe_maps_config(config_dir)
        path = find_backup_fringe_map(maps_cfg, filter,
                                      instrument.detector.name, observation_date)
        if path:
            copy_backup_calibration(path, output_file, "fringe map", filter)
            return True
        direct = (calib_params.get("backup_fringe_maps") or {}).get(filter)
        if direct:
            copy_backup_calibration(direct, output_file, "fringe map", filter)
            return True
        return False

    if calib_params.get("force_backup_fringe_map", False):
        logger.info("force_backup_fringe_map set; taking the map from config")
        if from_config():
            return
        logger.warning("No configured fringe map found; building from this night")

    list_file = caldir / f"{run}_fringe_{filter}.list"
    if not list_file.exists():
        logger.warning("No fringe frame list for filter %s: %s", filter, list_file)
        if from_config():
            return
        raise FileNotFoundError(
            f"No fringe frames for filter {filter} on {observation_date}, and no "
            f"entry in fringe_maps.yaml for detector "
            f"{instrument.detector.name}, nor a backup_fringe_maps path. "
            f"The reduction recipe includes a 'fringe' step, so a map is required.")

    filenames = read_list(list_file)
    logger.info("Building fringe map for filter %s from %d dithered frames",
                filter, len(filenames))
    reducer = make_reducer_upto(instrument, outdir, run, "fringe", filter)

    images = []
    for name in filenames:
        try:
            image, _ = reducer.reduce(instrument.open_frame(name))
            images.append(image)
        except Exception as exc:
            logger.error("Failed to reduce fringe frame %s: %s", name, exc)
    if len(images) < 3:
        raise ValueError(
            f"Only {len(images)} usable fringe frames for filter {filter}; "
            "a median combine needs several dithered frames to reject stars")

    n_frames = len(images)
    raw_template = combine_fringe_frames(images)
    del images                      # free the stack before filtering
    # The high pass is applied after the median in both cases, so the filtered
    # template follows from the unfiltered one; combining twice would double
    # peak memory for no gain.
    template = highpass(raw_template, instrument.fringe.highpass_sigma)
    write_fringe_map(output_file, template, raw_template, instrument, filter,
                     n_frames)


# --------------------------------------------------------------------------
# master bias, dark, flat
# --------------------------------------------------------------------------

def process_flat_session(instrument, filenames, run, outdir, filter,
                         clip, nlimit, session_name):
    """Combine one flat session (dusk, dawn or dome) into a normalised master."""
    logger.info("Processing %s session for filter %s", session_name, filter)
    if not filenames:
        logger.warning("No files found for %s session", session_name)
        return None

    n_files = len(filenames)
    logger.info("Processing %s session with %d files", session_name, n_files)
    reducer = make_reducer_upto(instrument, outdir, run, "flat", filter)

    data_cube = None
    for file_loop, filename in enumerate(filenames):
        logger.debug("Reading %s flat %d/%d: %s",
                     session_name, file_loop + 1, n_files, filename)
        try:
            image, _ = reducer.reduce(instrument.open_frame(filename))
        except Exception as exc:
            logger.error("Failed to process flat file %s: %s", filename, exc)
            raise
        if data_cube is None:
            ny, nx = image.shape
            logger.debug("Image dimensions: %d x %d pixels", nx, ny)
            data_cube = np.zeros((ny, nx, n_files))
        data_cube[:, :, file_loop] = image

    logger.debug("Creating %s session master flat with clip=%.1f, nlimit=%d",
                 session_name, clip, nlimit)
    session_master = medabsdevclip(data_cube, clip, nlimit)
    session_master /= np.median(session_master)
    logger.info("%s session master flat created successfully",
                session_name.capitalize())
    return session_master


def _make_master_flat(instrument, outdir, run, config, filter, clip, nlimit,
                      config_dir):
    caldir = Path(outdir) / "calib"
    output_file = caldir / f"{run}_master_flat_{filter}.fits"
    calib_params = config.get("calibration_params", {}) or {}
    observation_date = config["instrument_settings"]["date"]

    def from_config():
        flats_cfg = load_master_flats_config(config_dir)
        path = find_backup_flat_for_date(flats_cfg, filter, observation_date)
        if path:
            copy_backup_calibration(path, output_file, "flat", filter)
            return True
        direct = (calib_params.get("backup_master_flats") or {}).get(filter)
        if direct:
            copy_backup_calibration(direct, output_file, "flat", filter)
            return True
        return False

    if calib_params.get("force_backup_flats", False):
        logger.info("Forced backup flats enabled for filter %s", filter)
        if from_config():
            return None
        logger.warning("No suitable backup flat found, falling back to creation")

    sessions = {}
    for name in ("dusk", "dawn", "dome"):
        path = caldir / f"{run}_flat_{filter}_{name}.list"
        if path.exists():
            sessions[name] = read_list(path)
            logger.info("Found %d %s flat files", len(sessions[name]), name)

    combined_list = caldir / f"{run}_flat_{filter}.list"
    combined = read_list(combined_list) if combined_list.exists() else []
    if not sessions and combined:
        logger.warning("Using combined flat list %s", combined_list)
        logger.warning("Consider separate sessions for better star rejection")

    if not sessions and not combined:
        logger.warning("No flat field lists found for filter %s", filter)
        if from_config():
            return None
        raise ValueError(
            f"No flat field files or backup found for filter {filter}")

    if sessions:
        masters = []
        for name, files in sessions.items():
            master = process_flat_session(instrument, files, run, outdir, filter,
                                          clip, nlimit, name)
            if master is not None:
                masters.append(master)
        if not masters:
            raise ValueError(f"No valid flat sessions found for filter {filter}")
        if len(masters) == 1:
            logger.info("Only one session available, using that as master flat")
            master_frame = masters[0]
        else:
            logger.info("Combining %d session masters", len(masters))
            master_frame = np.median(np.stack(masters, axis=2), axis=2)
            master_frame /= np.median(master_frame)
        logger.info("Final master flat created from %d session(s)", len(masters))
    else:
        logger.info("Processing %d flat files from a combined list", len(combined))
        master_frame = process_flat_session(
            instrument, combined, run, outdir, filter, clip, nlimit, "combined")
    return master_frame


def _make_master_bias_or_dark(instrument, outdir, run, config, type, clip,
                              nlimit):
    caldir = Path(outdir) / "calib"
    output_file = caldir / f"{run}_master_{type}.fits"
    list_file = caldir / f"{run}_{type}.list"
    logger.info("Processing %s calibration from %s", type, list_file)

    if not list_file.exists():
        logger.warning("No %s list file found: %s", type, list_file)
        backup_path = (config.get("calibration_params", {}) or {}).get(
            f"backup_master_{type}")
        if backup_path:
            logger.info("Attempting to use backup master %s", type)
            copy_backup_calibration(backup_path, output_file, type)
            return None
        raise FileNotFoundError(
            f"No {type} list file found and no backup specified: {list_file}")

    filenames = read_list(list_file)
    logger.info("Found %d %s files", len(filenames), type)
    if not filenames:
        raise ValueError(f"{type} list {list_file} is empty")

    reducer = make_reducer_upto(instrument, outdir, run, type)

    if type == "bias" and len(filenames) >= 2:
        logger.info("Calculating readout noise from bias frames...")
        ron = calculate_readout_noise(instrument, filenames, reducer)
        try:
            with open(caldir / "readoutnoise.txt", "w") as f:
                f.write(f"{ron:.3f}\n")
            logger.info("Saved readout noise to: %s", caldir / "readoutnoise.txt")
        except Exception as exc:
            logger.error("Failed to save readout noise file: %s", exc)

    data_cube = None
    for file_loop, filename in enumerate(filenames):
        logger.debug("Reading %s %d/%d: %s",
                     type, file_loop + 1, len(filenames), filename)
        try:
            frame = instrument.open_frame(filename)
            image, _ = reducer.reduce(frame)
            if type == "dark":
                image = image / frame.meta.exptime
                logger.debug("Normalised dark by exposure time: %.2f",
                             frame.meta.exptime)
        except Exception as exc:
            logger.error("Failed to process %s file %s: %s", type, filename, exc)
            raise
        if data_cube is None:
            ny, nx = image.shape
            logger.debug("Image dimensions: %d x %d pixels", nx, ny)
            data_cube = np.zeros((ny, nx, len(filenames)))
        data_cube[:, :, file_loop] = image

    logger.debug("Creating master %s with clip=%.1f, nlimit=%d", type, clip, nlimit)
    master_frame = medabsdevclip(data_cube, clip, nlimit)

    if type == "dark":
        dark_current = instrument.gain() * np.median(master_frame)
        logger.info("Calculated dark current: %.3f electrons/sec/pixel", dark_current)
        try:
            with open(caldir / "darkcurrent.txt", "w") as f:
                f.write(f"{dark_current:.3f}\n")
        except Exception as exc:
            logger.error("Failed to save dark current file: %s", exc)
    return master_frame


def make_master_calibration(type, outdir, run, instrument, config,
                            filter="zYJ", clip=5, nlimit=5, config_dir=None):
    """Build one master calibration frame and write it to ``{outdir}/calib``."""
    logger.info("Creating master %s calibration", type)
    caldir = Path(outdir) / "calib"
    os.makedirs(caldir, exist_ok=True)
    config_dir = Path(config_dir or config.get("_config_dir", "."))

    if type == "fringe":
        return make_fringe_map(instrument, outdir, run, config, filter, config_dir)

    if type == "flat":
        output_file = caldir / f"{run}_master_flat_{filter}.fits"
        master_frame = _make_master_flat(instrument, outdir, run, config, filter,
                                         clip, nlimit, config_dir)
    elif type in ("bias", "dark"):
        output_file = caldir / f"{run}_master_{type}.fits"
        master_frame = _make_master_bias_or_dark(instrument, outdir, run, config,
                                                 type, clip, nlimit)
    else:
        raise ValueError(f"Unknown calibration type {type!r}")

    if master_frame is None:
        return None                      # a backup was copied into place

    logger.debug("Writing master %s to: %s", type, output_file)
    try:
        fits.PrimaryHDU(master_frame).writeto(output_file, overwrite=True)
        logger.info("Master %s frame created successfully: %s", type, output_file)
        logger.debug("Master %s statistics: mean=%.2f, median=%.2f, std=%.2f",
                     type, np.mean(master_frame), np.median(master_frame),
                     np.std(master_frame))
    except Exception as exc:
        logger.error("Failed to write master %s file %s: %s", type, output_file, exc)
        raise
    return master_frame


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Running master_calibrations.py as standalone script")
