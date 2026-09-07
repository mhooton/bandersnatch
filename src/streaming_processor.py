"""Single-pass reduction, centroiding and photometry.

Each frame is read once, reduced, centroided and measured, then discarded.
Nothing intermediate is written unless asked for, which keeps a night's
processing within memory even for large detectors.
"""

import logging
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from aper import aper
from centroid import centroid_loop, write_centroiding_results
from photometry import count_bad_pixels_in_apertures, write_photometry_results
from reduce_science import annotate_header

logger = logging.getLogger(__name__)


#: Per-frame quantities recorded for every science frame, so that later stages
#: never need to reopen a raw file to read a header.
FRAME_TABLE_COLUMNS = [
    "file", "bjd_mid", "exptime", "airmass", "altitude", "filter",
    "sky_level", "overscan_level", "fringe_scale", "fringe_intercept",
]


def write_frames_table(target_dir, records):
    """Write ``frames.fits``: one row per science frame."""
    if not records:
        return None
    table = Table()
    for column in FRAME_TABLE_COLUMNS:
        values = [r.get(column) for r in records]
        if all(v is None for v in values):
            continue
        if all(isinstance(v, str) or v is None for v in values):
            table[column] = [("" if v is None else str(v)) for v in values]
        else:
            table[column] = [np.nan if v is None else float(v) for v in values]
    path = Path(target_dir) / "frames.fits"
    table.write(path, format="fits", overwrite=True)
    logger.info("Wrote per-frame table %s (%d rows, columns: %s)",
                path, len(table), ", ".join(table.colnames))
    return table


def process_images_streaming(instrument, reducer, outdir, run, target, config,
                             initial_positions, centroid_params,
                             photometry_params, bad_pixel_map=None,
                             save_processed_images=False, target_coord=None):
    """Reduce, centroid and measure every science frame for one target."""
    logger.info("Starting streaming processing for target %s", target)

    gain = instrument.gain()
    phpadu = instrument.phpadu()

    list_file = outdir / "calib" / f"{run}_image_{target}.list"
    try:
        with open(list_file) as f:
            filenames = [line.strip() for line in f if line.strip()]
        logger.info("Found %d images for streaming processing", len(filenames))
    except Exception as exc:
        logger.error("Failed to read image list %s: %s", list_file, exc)
        raise
    if not filenames:
        raise ValueError(f"No images found for target {target}")

    centroid_results = {k: [] for k in (
        "BJD", "File", "airmass", "xc", "yc", "x_bright", "y_bright",
        "val_bright", "x_width", "y_width", "n_pixels_above",
        "sig3_pixels", "sig5_pixels", "sig10_pixels")}
    photometry_results = {}
    all_poststamps = []
    processed_filenames = []
    frame_records = []

    star_x_input = np.array(initial_positions)[:, 0]
    star_y_input = np.array(initial_positions)[:, 1]
    n_stars = len(star_x_input)

    apr = np.arange(photometry_params["aper_min"],
                    photometry_params["aper_max"] + 1, dtype=float)
    skyrad = np.array([photometry_params["SKYRAD_inner"],
                       photometry_params["SKYRAD_outer"]])
    setskyval = 1e-20 if photometry_params["sky_suppress"] else None

    for aper_radius in apr:
        photometry_results[f"aper{int(aper_radius)}"] = {
            k: [] for k in ("BJD", "File", "flux", "flux_err", "sky", "sky_err",
                            "n_bad_pixels_in_aperture")}

    target_dir = outdir / target
    run_dir = target_dir / run
    if save_processed_images:
        os.makedirs(run_dir, exist_ok=True)
        logger.info("Created directories for saving processed images")

    successful_images = 0
    failed_images = []

    for i, filename in enumerate(filenames):
        if (i + 1) % 10 == 0:
            logger.info("Processing image %d/%d", i + 1, len(filenames))
        try:
            frame = instrument.open_frame(filename)
            image, record = reducer.reduce(frame)
            meta = frame.meta
            bjd_obs = instrument.compute_bjd(meta, target_coord)

            if save_processed_images:
                processed_filename = f"proc{Path(filename).name}"
                header = annotate_header(meta.header, instrument, record)
                fits.PrimaryHDU(data=image, header=header).writeto(
                    run_dir / processed_filename, overwrite=True)
                processed_filenames.append(processed_filename)

            centroid_result = centroid_loop(
                star_x_input, star_y_input,
                centroid_params["boxsize"], centroid_params["nlimit_centroid"],
                centroid_params["clip_centroid"], centroid_params["sky_sigma"],
                centroid_params["tracking_star"],
                centroid_params["flux_above_value"],
                image=image, header=meta.header,
                mask_centroid_pixels=centroid_params["mask_centroid_pixels"])

            centroid_results["BJD"].append(bjd_obs)
            centroid_results["File"].append(str(filename))
            centroid_results["airmass"].append(meta.airmass)
            for key in ("xc", "yc", "x_bright", "y_bright", "val_bright",
                        "x_width", "y_width", "n_pixels_above",
                        "sig3_pixels", "sig5_pixels", "sig10_pixels"):
                centroid_results[key].append(centroid_result[key])
            all_poststamps.append(centroid_result["poststamps"])

            xc_frame = centroid_result["xc"]
            yc_frame = centroid_result["yc"]
            mags, errap, sky, skyerr = aper(
                image=image, xc=xc_frame, yc=yc_frame, phpadu=phpadu, apr=apr,
                skyrad=skyrad, setskyval=setskyval, flux=True, silent=True)

            n_bad_pix_aperture = count_bad_pixels_in_apertures(
                bad_pixel_map, image.shape, xc_frame, yc_frame, apr, n_stars)

            for aper_idx, aper_radius in enumerate(apr):
                res = photometry_results[f"aper{int(aper_radius)}"]
                res["BJD"].append(bjd_obs)
                res["File"].append(str(filename))
                res["flux"].append(mags[aper_idx, :] * gain)
                res["flux_err"].append(errap[aper_idx, :] * gain)
                res["sky"].append(sky * gain)
                res["sky_err"].append(skyerr * gain)
                res["n_bad_pixels_in_aperture"].append(
                    n_bad_pix_aperture[aper_idx, :])

            record = dict(record)
            record.update(file=str(filename), bjd_mid=bjd_obs,
                          exptime=meta.exptime, airmass=meta.airmass,
                          altitude=meta.altitude, filter=meta.filter)
            frame_records.append(record)
            successful_images += 1
        except Exception as exc:
            logger.error("Failed to process image %s: %s", filename, exc)
            failed_images.append(filename)
            continue

    logger.info("Streaming processing complete: %d/%d images successful",
                successful_images, len(filenames))
    if successful_images == 0:
        raise RuntimeError("No images were successfully processed")

    if save_processed_images and processed_filenames:
        from create_lists import write_liste
        try:
            write_liste(processed_filenames, f"{run}_proc_{target}.list", outdir)
            logger.info("Created processed frame list with %d files",
                        len(processed_filenames))
        except Exception as exc:
            logger.error("Failed to write processed frame list: %s", exc)

    logger.info("Writing accumulated results to disk")
    centroiding_dir = target_dir / "centroiding"
    photometry_dir = target_dir / "photometry"
    os.makedirs(centroiding_dir, exist_ok=True)
    os.makedirs(photometry_dir, exist_ok=True)

    write_centroiding_results(centroiding_dir, centroid_results, all_poststamps,
                              n_stars, centroid_params["boxsize"])
    median_filter_window = config.get("photometry_settings", {}).get(
        "median_filter_window", 21)
    write_photometry_results(photometry_dir, photometry_results, config, target,
                             outdir, median_filter_window)
    write_frames_table(target_dir, frame_records)
    logger.info("All results written successfully")
    return successful_images, failed_images


def load_calibration_frames(instrument, outdir, run, filter_name):
    """Load whichever master frames this instrument's recipe consumes."""
    caldir = Path(outdir) / "calib"
    wanted = {
        "bias": caldir / f"{run}_master_bias.fits",
        "dark": caldir / f"{run}_master_dark.fits",
        "flat": caldir / f"{run}_master_flat_{filter_name}.fits",
    }
    frames = {}
    for step, path in wanted.items():
        if step not in instrument.steps:
            continue
        try:
            with fits.open(path) as hdul:
                frames[step] = hdul[0].data
            logger.debug("Loaded master %s: shape %s", step, frames[step].shape)
        except Exception as exc:
            logger.error("Failed to load master %s from %s: %s", step, path, exc)
            raise
    return frames
