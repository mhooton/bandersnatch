"""Apply the reduction chain to a target's science frames and write them out.

This is the sequential path, used when the pipeline is asked for only part of
the workflow.  When reduction, centroiding and photometry are all requested,
:mod:`streaming_processor` does the same reduction in one pass without writing
intermediate images.
"""

import logging
import os

import numpy as np
from astropy.io import fits

from create_lists import write_liste

logger = logging.getLogger(__name__)


def annotate_header(header, instrument, record):
    """Record what was done to a frame, and where it sits on the raw detector.

    ``LTV1``/``LTV2`` are the IRAF convention for a section offset.  DS9 and
    friends use them to report the original, untrimmed pixel coordinates in
    their *physical* readout, so a processed frame and a raw frame can be
    compared position for position.
    """
    header = header.copy()
    dx, dy = instrument.detector.trim_offset
    if dx or dy:
        header["LTV1"] = (-dx, "Offset to raw detector x")
        header["LTV2"] = (-dy, "Offset to raw detector y")
    header["BANDINST"] = (instrument.name, "Instrument config used")
    header["BANDDET"] = (instrument.detector.name, "Detector processed")
    header["BANDSTEP"] = (",".join(instrument.steps), "Reduction steps applied")
    for key, value in (record or {}).items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            header[key[:8].upper()] = float(value)
    return header


def reduce_science_frames(instrument, reducer, outdir, run, target,
                          target_coord=None):
    """Reduce every science frame for ``target``; return the per-frame records."""
    logger.info("Starting science frame reduction for target %s", target)

    list_file = outdir / "calib" / f"{run}_image_{target}.list"
    try:
        with open(list_file) as f:
            filenames = [line.strip() for line in f if line.strip()]
        logger.info("Found %d science frames for target %s", len(filenames), target)
    except Exception as exc:
        logger.error("Failed to read science image list %s: %s", list_file, exc)
        raise
    if not filenames:
        raise ValueError(f"No science frames found for target {target}")

    run_dir = outdir / target / run
    os.makedirs(run_dir, exist_ok=True)
    logger.debug("Created output directory: %s", run_dir)

    output_filenames = []
    records = []
    for i, filename in enumerate(filenames):
        logger.debug("Processing science frame %d/%d: %s",
                     i + 1, len(filenames), filename)
        try:
            frame = instrument.open_frame(filename)
            image, record = reducer.reduce(frame)

            output_filename = f"proc{os.path.basename(filename)}"
            header = annotate_header(frame.meta.header, instrument, record)
            fits.PrimaryHDU(data=image, header=header).writeto(
                run_dir / output_filename, overwrite=True)

            output_filenames.append(output_filename)
            record = dict(record)
            record.update(file=str(filename), processed=output_filename,
                          bjd_mid=instrument.compute_bjd(frame.meta, target_coord),
                          exptime=frame.meta.exptime, airmass=frame.meta.airmass,
                          altitude=frame.meta.altitude, filter=frame.meta.filter)
            records.append(record)
        except Exception as exc:
            logger.error("Failed to process science frame %s: %s", filename, exc)
            logger.warning("Skipping failed frame and continuing")
            continue

    logger.info("Successfully reduced %d/%d science frames for target %s",
                len(output_filenames), len(filenames), target)
    if not output_filenames:
        raise RuntimeError(
            f"No science frames were successfully reduced for target {target}")

    # The per-frame table is what later stages read instead of reopening raw
    # files, so the sequential path must write it too.
    from streaming_processor import write_frames_table
    write_frames_table(outdir / target, records)

    write_liste(output_filenames, f"{run}_proc_{target}.list", outdir)
    logger.info("Created processed frame list: %s_proc_%s.list with %d files",
                run, target, len(output_filenames))
    logger.info("Science frame reduction completed for target %s", target)
    return records


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("Running reduce_science.py as standalone script")
