# Instrument abstraction for bandersnatch

Design document, September 2026. Motivated by adding the INT Wide Field Camera (WFC), with the
general aim that a further instrument should need one YAML file and no edits to pipeline modules.

## 1. Purpose and scope

Today every pipeline module reads FITS files directly and assumes the SPECULOOS conventions:
a single-extension file, `IMAGETYP` containing LIGHT/DARK/BIAS/FLAT, `FILTER`, `OBJECT`,
`BJD-OBS`, `AIRMASS` or `ALTITUDE`, a fixed `bias, dark, flat` chain, and a fixed raw-data
directory layout. WFC violates every one of those. Rather than add WFC branches, this design
moves all instrument knowledge behind one object, `Instrument`, and makes the reduction chain a
configured list of steps.

In scope: frame discovery and classification, per-detector geometry (extension, trim,
overscan), header normalisation including time and BJD, the ordered reduction recipe, and two
new steps (overscan, fringe subtraction). Out of scope for now: multi-detector processing in a
single run, non-linearity correction, the IDL polynomial sky regression.

## 2. Principles

1. **One place touches raw FITS.** Only `Instrument.open_frame` opens a raw file. Every stage
   receives a `Frame` with a 2-D float array and normalised metadata.
2. **YAML first, Python second.** Instrument behaviour is declared in
   `configs/{INSTRUMENT}.yaml`. A Python subclass is allowed for behaviour YAML cannot express,
   selected by an `adapter:` key, but WFC and the SPECULOOS cameras must not need one.
3. **Defaults are the legacy behaviour.** Existing SPIRIT, Callisto, SINISTRO and Io_ANDOR YAMLs
   keep working unchanged. Every new key has a default equal to what the code does today.
4. **Trimmed detector coordinates everywhere.** All processed images, master frames, star
   positions and photometry use the coordinates of the trimmed image. Instruments without a
   trim section are unaffected.
5. **Record, do not recompute.** Per-frame quantities that stages need later (time, exposure
   time, airmass, altitude, sky level, overscan level, fringe scale) are written once to a
   per-frame table. No stage re-opens raw files to read headers.

## 3. Where instrument knowledge lives today

| Module | Assumption | WFC reality |
|---|---|---|
| `create_lists.get_liste` | `*.fits` in dir and one subdir | `*.fits.fz` in `download_dir` |
| `create_lists.classify_file` | `IMAGETYP` contains LIGHT/DARK/BIAS/FLAT | `zero`/`object`; types only in `OBSTYPE` + free-text `OBJECT` |
| `create_lists.sort_*` | `FILTER`, `OBJECT` in primary header | `WFFBAND`; `OBJECT` also names flats and fringe frames |
| `create_lists.get_observation_time` | `DATE-OBS` has a time; Paranal hour windows for dawn/dusk | date only; La Palma |
| `create_lists.create_lists` | previous nights are sibling directories named by date; darks must exist | no darks taken |
| `run.setup_paths` | `{topdir}/Observations/{inst}/images/{date}` | `{topdir}/WFC/{date}/download_dir` |
| `run.main` flat loading | `hdul[0].header['FILTER']` | `WFFBAND` |
| `master_calibrations` | `hdul[image_extension].data`, `hdul[0].header['EXPTIME']`, LIRIS branch, one gain | 4 extensions with per-chip gain; overscan needed |
| `reduce_science.apply_calibrations` | bias, dark, flat, fixed | overscan, bias, flat, fringe |
| `streaming_processor` | as above; `image_extension` read from the wrong section; broken `overscan_corr` call | |
| `centroid`, `streaming_processor` | `BJD-OBS` else `MJD-OBS` else frame index; `AIRMASS` else `ALTITUDE` | `MJD-OBS` at start, no BJD; `AIRMASS` present |
| `photometry_analysis` | re-opens files for `EXPTIME`, `ALTITUDE` | no `ALTITUDE`; derive from `ZDSTART`/`ZDEND` |
| `bad_pixel_map` | needs a master dark | none |

## 3.5 How data is reached, locally and on the server

There is a single root, `paths.topdir` in the date config, and both locations are the same code
with a different string:

| | Path | Local | Server |
|---|---|---|---|
| Raw frames | `{topdir}/Observations/{inst}/images/{date}` | `topdir: /Users/matthewhooton` | `topdir: /data/SPECULOOSPipeline` |
| Outputs | `{topdir}/bandersnatch_runs/{run_name}` | | |
| Configs | `{topdir}/bandersnatch_runs/configs` | | |

On the server, `docker/docker-compose.server` bind-mounts `/export/data/SPECULOOSPipeline` onto
`/data/SPECULOOSPipeline` and mounts the source tree live, so container paths equal the
configured `topdir` and no translation is required.

Two mechanisms exist for this and are stale. They are not part of this design, but they should
not be trusted or extended:

- `photometry.translate_path_for_docker` rewrites stored paths when `RUNNING_IN_DOCKER` is set.
  Nothing sets that variable, so it returns its input unchanged in every context, and the
  `/app/output` layout it targets no longer exists. The bind mount is what makes Docker work.
- `batch_run.sh --staged` reads raw data from `{topdir}/data/{inst}/{date}`, the layout replaced
  in commit 0784982. Staged batch running is therefore broken against the current `run.py`;
  unstaged batch running is unaffected.
- `run_modular.py` is a stale parallel copy of `run.py` carrying its own staging and Docker
  branches. It is not the entry point in use.

**Consequence for the abstraction.** The raw-directory template moves into the instrument YAML
as `raw_dir`, with `{topdir}`, `{inst}` and `{date}` placeholders, and may be overridden per
night in the date config's `paths` block. WFC then works unchanged in both places:

```yaml
# WFC.yaml
raw_dir: "{topdir}/WFC/{date}/download_dir"
# a date config may override, e.g. on the server
paths:
  topdir: "/data/SPECULOOSPipeline"
  raw_dir: "{topdir}/Observations/WFC/images/{date}"
```

The default template is the current `{topdir}/Observations/{inst}/images/{date}`, so no existing
instrument config changes.

## 4. The abstraction

### 4.1 `Instrument`

Built by `load_instrument(name, config_dir)` from `configs/{name}.yaml`. Public surface:

```python
class Instrument:
    name: str
    detector: Detector                     # the selected detector (see 4.4)
    observatory: EarthLocation | None
    steps: list[str]                       # ordered reduction recipe (see 4.6)
    fringe: FringeParams | None

    def raw_dir(self, topdir: Path, date: str) -> Path
    def find_raw_files(self, raw_dir: Path) -> list[Path]
    def open_frame(self, path: Path) -> Frame          # raw array of the selected detector + metadata
    def read_meta(self, path: Path) -> FrameMeta       # headers only, no pixel data
    def classify(self, meta: FrameMeta) -> FrameType   # bias | dark | flat | fringe | science | ignore
    def flat_session(self, meta: FrameMeta) -> str     # 'dusk' | 'dawn' | 'dome' | 'unknown'
    def gain(self) -> float; def read_noise(self) -> float | None; def phpadu(self) -> float
```

Legacy keys `gain`, `phpadu`, `saturation_threshold`, `telescope_diameter`,
`observatory_altitude`, `scintillation`, `bad_pixel_correction` remain valid at the top level
and populate a single default detector when no `detectors:` block is present.

### 4.2 `Frame` and `FrameMeta`

```python
@dataclass
class FrameMeta:
    path: Path
    frame_type: FrameType
    object: str                 # cleaned target name (spaces -> '--' as today)
    filter: str
    exptime: float              # seconds
    time_start: Time | None     # UTC start, when derivable
    bjd_mid: float              # BJD_TDB at mid-exposure (see 4.5)
    airmass: float              # -1 if unknown
    altitude: float | None      # degrees
    header: fits.Header         # merged header, for provenance only
    extra: dict                 # any additional keywords requested in YAML

@dataclass
class Frame:
    meta: FrameMeta
    data: np.ndarray            # float64, full raw array of the selected detector, untrimmed
```

Header merging: `header_hdus` lists the HDUs whose headers are combined, later entries
overriding earlier ones. WFC uses `[0, data_hdu]` so per-chip `GAIN`, `BIASSEC` and `TRIMSEC`
are visible alongside the primary keywords. Default `[0]`.

Keyword resolution supports three forms, so that fallbacks and simple derivations need no code:

```yaml
keywords:
  exptime: EXPTIME                                   # single keyword
  airmass: {keywords: [AIRMASS, AMSTART]}            # first present wins
  altitude: {expr: "90 - 0.5*(ZDSTART + ZDEND)"}     # arithmetic on header values
  filter: {keyword: WFFBAND, strip: true}
```

Expressions are evaluated with a restricted namespace: header values, `min`, `max`, `abs`,
and the `math` module. Nothing else.

### 4.3 Frame classification

An ordered list of rules. The first match wins; a frame matching nothing is `ignore` and is
logged once per run with its `OBJECT`.

```yaml
classification:
  - {type: bias,    where: {OBSTYPE: BIAS}}
  - {type: ignore,  where: {OBJECT: {regex: "(?i)^NLtest"}}}
  - {type: flat,    where: {OBJECT: {regex: "(?i)flat"}}}
  - {type: fringe,  where: {OBJECT: {regex: "(?i)fringe"}}}
  - {type: science, where: {OBSTYPE: TARGET}}
```

`where` values are exact strings after stripping, or `{regex: ...}`, or `{contains: ...}`.
Multiple keys in one `where` are ANDed. The default rule set reproduces today's behaviour:

```yaml
classification:
  - {type: science, where: {IMAGETYP: {contains: LIGHT}}}
  - {type: dark,    where: {IMAGETYP: {contains: DARK}}}
  - {type: bias,    where: {IMAGETYP: {contains: BIAS}}}
  - {type: flat,    where: {IMAGETYP: {contains: FLAT}}}
```

Matching is case-insensitive for `contains`, as today.

Flat sessions: `flat_session` returns `dome` when a `dome_flat` rule matches (WFC: `OBJECT`
contains "Domeflat"), otherwise `dusk` or `dawn` by comparing the frame time with local solar
midnight at the observatory. This replaces the fixed 04:00 to 10:00 and 18:00 to 23:00 UT
windows, which happen to work at Paranal and La Palma but would not elsewhere. When no
observatory is configured, the current hour windows remain as the fallback.

List files gain a fringe list, `{run}_fringe_{filter}.list`, and dome flats are written to
`{run}_flat_{filter}_dome.list`. `master_calibrations` treats a dome list like a dusk or dawn
session. An instrument or date config may set `flat_sources: [sky]` or `[dome]` to restrict
which sessions are combined; the default is all available.

### 4.4 Geometry: detectors, trim, overscan

```yaml
detectors:
  CCD4:
    hdu: 4                        # int index or EXTNAME
    trim:     "[54:2101,1:4096]"  # FITS 1-based inclusive section, IRAF convention
    overscan: "[2102:2154,1:4096]"
    gain: 2.9                     # e-/ADU
    read_noise: 5.8               # e-
    saturation: 65535
    bad_columns: [1780]           # trimmed x, optional seed for the bad pixel map
default_detector: CCD4
```

The date config may override with `instrument_settings.detector: CCD2`. One detector is
processed per run. Processing several chips is a loop over runs, not a change to the
pipeline, and is deliberately left for later.

`Frame.data` is the untrimmed array so the overscan step can see its region. The `trim` step
is implicit: it is applied immediately after `overscan` (or first, if there is no overscan
step) and everything downstream, including master frames, is in trimmed coordinates. For
instruments with no `trim`, nothing changes.

The overscan region is taken from the YAML, never from `BIASSEC`. On WFC chip 2 that keyword
points at illuminated pixels.

### 4.5 Time and BJD

```yaml
time:
  keyword: MJD-OBS          # or BJD-OBS, JD, DATE-OBS
  format: mjd               # mjd | jd | isot
  scale: utc                # utc | tdb
  reference: start          # start | mid ; 'start' adds exptime/2
observatory: {lat: 28.7619, lon: -17.8776, height: 2348}
```

`bjd_mid` is computed as `Time(...) + exptime/2`, converted to BJD_TDB with astropy's
`light_travel_time` towards the target.

**Target coordinates come from `targets.yaml`.** Each target entry gains `ra` and `dec`
(sexagesimal or degrees), which are the authoritative source for the barycentric correction.
Where they are absent the telescope pointing in the header is used as a fallback, and a warning
names the target and the resulting uncertainty. For a target a few arcminutes off axis that
approximation costs about a millisecond, so the fallback is usable but should not become the
norm; `ra`/`dec` are added to entries as nights are processed.

For SPECULOOS the YAML default is `{keyword: BJD-OBS, format: jd, scale: tdb, reference: mid}`,
which is a pass-through and preserves the current tables exactly. The existing fallback
chain (BJD-OBS, then MJD-OBS, then frame index) becomes a hard error with a clear message when
no configured time keyword is present; a frame index is not a time.

### 4.6 Reduction recipe

```yaml
reduction:
  steps: [overscan, bias, flat, fringe]     # default: [bias, dark, flat]
  overscan: {method: median}                # median | row_median | polynomial (order: 1)
```

A `Reducer` is constructed once per target from the instrument and the loaded calibration
products, and exposes `reduce(frame) -> (image, record)`. `record` is a dict of per-frame
scalars produced by the steps (`overscan_level`, `fringe_scale`, `sky_level`, ...). The
current `apply_calibrations` becomes the `[bias, dark, flat]` special case and is removed.

Steps and their products:

| Step | Needs | Produces in `record` |
|---|---|---|
| `overscan` | detector overscan section | `overscan_level` |
| `bias` | `{run}_master_bias.fits` | |
| `dark` | `{run}_master_dark.fits`, exptime | |
| `flat` | `{run}_master_flat_{filter}.fits` | |
| `bpm` | `bad_pixel_map.fits` | (interpolation, as today, when a map exists) |
| `fringe` | `{run}_fringe_map_{filter}.fits` | `fringe_scale`, `fringe_intercept`, `sky_level` |

Step order is validated at start-up: `overscan` must be first, `fringe` must follow `flat`,
`dark` requires `bias`. Missing products are reported before any frame is processed, naming
the list file or backup key that would supply them.

The `fringe` step implements the whole-image fit agreed after the chip 4 experiment:

1. Star mask: smooth by `star_mask.smooth` px, flag above median plus `star_mask.threshold_mad`
   MAD, dilate by `star_mask.dilate` px; also mask configured bad columns and a border.
2. Fit `image = scale * template + a + b*x + c*y` on masked pixels subsampled every
   `fit.subsample` pixels, `fit.iterations` rounds of `fit.clip_sigma` clipping.
3. Subtract `scale * template` only. Record `scale`, `a`, and the residual slope against the
   template on held-out pixels as a diagnostic.

```yaml
fringe:
  highpass_sigma: 120        # px, separates large-scale sky from the fringe pattern
  template_smooth_sigma: 2   # px, removes the noise that otherwise attenuates the fitted scale
  star_mask: {smooth: 3, threshold_mad: 4, dilate: 12, border: 40}
  fit: {plane: true, clip_sigma: 3, iterations: 3, subsample: 7}
```

### 4.7 Calibration products

`master_calibrations.make_master_calibration` keeps its role and gains:

- `instrument.open_frame` in place of `fits.open` + `image_extension`, so masters are built
  from the selected detector, overscan-corrected and trimmed when the recipe says so.
- A `fringe` type. It reduces each listed fringe frame through the steps preceding `fringe`,
  subtracts each frame's median, MAD-clip combines, high-passes with `highpass_sigma`, smooths
  with `template_smooth_sigma`, and writes `{run}_fringe_map_{filter}.fits` with the
  unsmoothed template in a second extension for inspection and the provenance keywords of
  section 4.9 in the header. Falls back to `fringe_maps.yaml` per that section.
- Read-out noise from bias pairs uses the detector gain; dark current is only computed when
  `dark` is in the recipe.
- The LIRIS `RUNSET` filter and `liris.py` are removed.

`bad_pixel_map.make_bad_pixel_map` accepts `master_dark=None` and then uses hot pixels from
the master bias scatter plus flat-based detection, and seeds from `bad_columns`.

### 4.8 Per-frame record table

Every processed target gets `{target}/frames.fits`, one row per science frame:

`file, bjd_mid, exptime, airmass, altitude, filter, sky_level, overscan_level, fringe_scale,
fringe_intercept, n_bad_pixels_in_box`

`centroids.fits` and `photometry_aper*.fits` keep their current columns so downstream tools
are unaffected. `photometry_analysis` reads `exptime` and `altitude` from `frames.fits`
instead of re-opening raw files, and `fringe_scale` becomes available as a decorrelation
vector alongside airmass.

### 4.9 Reuse of fringe maps across nights: `fringe_maps.yaml`

The fringe pattern is fixed by the CCD; only its amplitude varies, and the per-frame scale
handles that. One map can therefore serve a whole observing run, and a night with too few
dithered blank-sky frames can borrow one. This mirrors `master_flats.yaml`, with one addition:
a fringe map belongs to a **detector** as well as a filter, so the file is keyed by both.

```yaml
# ~/bandersnatch_runs/configs/fringe_maps.yaml
maps:
  z:                              # filter
    CCD4:                         # detector name; use 'default' for single-detector instruments
      - start_date: "20170801"
        path: "/Users/matthewhooton/bandersnatch_runs/WFC_20170805/calib/1_fringe_map_z.fits"
      - start_date: "20200101"
        path: "/Users/matthewhooton/bandersnatch_runs/WFC_20200118/calib/1_fringe_map_z.fits"
  i:
    CCD4:
      - start_date: "20170801"
        path: "/Users/matthewhooton/bandersnatch_runs/WFC_20170805/calib/1_fringe_map_i.fits"
```

Selection is the existing rule: the most recent `start_date` that is less than or equal to the
observation date. The date-matching helper currently inside
`master_calibrations.find_backup_flat_for_date` is generalised to
`find_dated_entry(config, keys, observation_date)` and used by both flats and fringe maps, so
there is one implementation of this logic rather than two.

Resolution order for the fringe map, matching how flats already behave:

1. `calibration_params.force_backup_fringe_map: true` forces the config lookup, skipping
   creation.
2. Otherwise build from this night's `{run}_fringe_{filter}.list` when `to_make_fringe_map` is
   set and the list exists.
3. Otherwise `fringe_maps.yaml` for this filter, detector and date.
4. Otherwise `calibration_params.backup_fringe_maps: {z: "/path"}`, the direct-path form kept
   for symmetry with `backup_master_flats`.
5. Otherwise a clear error naming the filter, detector and date. The `fringe` step is in the
   recipe by explicit choice, so a silently skipped correction is worse than a stop.

**Provenance checking.** Every fringe map is written with `INSTRUME`, `DETECTOR`, `FILTER`,
`NFRAMES` and `TRIMSEC` in its header, and these are verified on load against the current run.
A map built for another detector or filter is refused rather than applied. Applying, say, a CCD3
map to CCD4 would produce a plausible-looking but wrong correction, so this check is not
optional. Array shape is checked against the trimmed detector shape for the same reason.

### 4.10 Coordinate convention for star positions

Star positions are identified by eye on **raw, untrimmed** frames, because that is what exists
before the pipeline has run. Internally the pipeline works on trimmed images. The instrument
YAML therefore declares which convention its configured positions use:

```yaml
coordinates: raw        # raw | trimmed. Default 'trimmed' (equivalently, no trim at all)
```

With `coordinates: raw`, the `Instrument` adds the trim offset to `star0_positions` and to
`all_star_positions` exactly once, at config load. Everything after that point, including
`centroids.fits`, `photometry_aper*.fits`, poststamps and diagnostic plots, is in trimmed
coordinates. Instruments with no trim section are unaffected either way.

Two safeguards, because a silently misapplied offset would be hard to diagnose:

1. The conversion is logged at INFO on every run, naming each target and both coordinate pairs,
   for example `KELT16 star 0: raw (263.0, 1924.6) -> trimmed (210.0, 1924.6), offset (-53, 0)`.
2. Processed images are written with `LTV1`/`LTV2` set to the negated trim offset, so DS9 and
   IRAF-aware tools report raw coordinates in their *physical* readout while the image
   coordinates remain trimmed. Opening a raw frame and a processed frame side by side then
   gives matching physical positions.

**Note on pixel indexing.** bandersnatch positions are numpy 0-based, whereas DS9 displays
1-based FITS coordinates. A position read from DS9 needs 1 subtracted from each axis. This
predates the present work and is unchanged by it; against a centroid box of 15 to 30 pixels the
one-pixel offset is harmless, but it is worth stating in the README.

## 5. Instrument YAML examples

### 5.1 SPIRIT, unchanged file, shown with the defaults it inherits

```yaml
instrument_name: "SPIRIT"
telescope_diameter: 1.0
observatory_altitude: 2440
gain: 5.092
phpadu: 6.36
saturation_threshold: 11000
bad_pixel_correction: {hot_sigma_threshold: 5, cold_sigma_threshold: 5, flat_threshold: 0.1, overwrite_existing: false}
scintillation: {C_Y: 1.56, H: 8000}
# implicit defaults:
# raw_dir: "{topdir}/Observations/{inst}/images/{date}"
# file_patterns: ["*.fits"]; search_subdirectories: true
# detectors: {default: {hdu: 0}}; header_hdus: [0]
# keywords: {exptime: EXPTIME, filter: FILTER, object: OBJECT, airmass: {keywords: [AIRMASS]}, altitude: ALTITUDE}
# time: {keyword: BJD-OBS, format: jd, scale: tdb, reference: mid}
# classification: IMAGETYP contains LIGHT/DARK/BIAS/FLAT
# reduction: {steps: [bias, dark, flat]}
```

### 5.2 WFC

```yaml
instrument_name: "WFC"
telescope_diameter: 2.5
observatory: {lat: 28.7619, lon: -17.8776, height: 2348}
observatory_altitude: 2348
scintillation: {C_Y: 1.56, H: 8000}
phpadu: 1.0

raw_dir: "{topdir}/WFC/{date}/download_dir"
file_patterns: ["*.fits.fz", "*.fits", "*.fit"]
header_hdus: [0, data]              # 'data' means the selected detector's HDU

detectors:
  CCD1: {hdu: extension1, trim: "[54:2101,1:4096]", overscan: "[2102:2154,1:4096]", gain: 2.8, read_noise: 6.4, saturation: 65535}
  CCD2: {hdu: extension2, trim: "[54:2101,1:4096]", overscan: "[2102:2154,1:4096]", gain: 3.0, read_noise: 6.9, saturation: 65535, bad_columns: [834, 2041]}
  CCD3: {hdu: extension3, trim: "[54:2101,1:4096]", overscan: "[2102:2154,1:4096]", gain: 2.5, read_noise: 5.5, saturation: 65535, bad_columns: [1221, 1230, 1240, 1241, 1242]}
  CCD4: {hdu: extension4, trim: "[54:2101,1:4096]", overscan: "[2102:2154,1:4096]", gain: 2.9, read_noise: 5.8, saturation: 65535, bad_columns: [1780]}
default_detector: CCD4
coordinates: raw                    # star positions in targets.yaml are untrimmed

keywords:
  exptime: EXPTIME
  filter: {keyword: WFFBAND, strip: true}
  object: OBJECT
  airmass: {keywords: [AIRMASS, AMSTART]}
  altitude: {expr: "90 - 0.5*(ZDSTART + ZDEND)"}
  pointing_ra: RA
  pointing_dec: DEC

time: {keyword: MJD-OBS, format: mjd, scale: utc, reference: start}

classification:
  - {type: bias,    where: {OBSTYPE: BIAS}}
  - {type: ignore,  where: {OBJECT: {regex: "(?i)^NLtest"}}}
  - {type: flat,    where: {OBJECT: {regex: "(?i)flat"}}}
  - {type: fringe,  where: {OBJECT: {regex: "(?i)fringe"}}}
  - {type: science, where: {OBSTYPE: TARGET}}
dome_flat: {OBJECT: {regex: "(?i)domeflat"}}
flat_sources: [sky]

reduction:
  steps: [overscan, bias, flat, fringe]
  overscan: {method: median}

fringe:
  highpass_sigma: 120
  template_smooth_sigma: 2
  star_mask: {smooth: 3, threshold_mad: 4, dilate: 12, border: 40}
  fit: {plane: true, clip_sigma: 3, iterations: 3, subsample: 7}

bad_pixel_correction: {hot_sigma_threshold: 5, cold_sigma_threshold: 5, flat_threshold: 0.5, overwrite_existing: false}
```

Bad-column values above are the raw-array columns I measured, converted to trimmed
coordinates; they should be checked against a master flat before use.

## 6. Changes by module

- **`instrument.py`** (new): `Instrument`, `Detector`, `Frame`, `FrameMeta`, `FrameType`,
  `load_instrument`, FITS section parsing, keyword resolution, classification, time
  conversion, `raw_dir`, `find_raw_files`, solar-midnight session split.
- **`reduction.py`** (new): `Reducer`, step functions `overscan_correct`, `trim`,
  `fringe_subtract`, `fit_fringe_scale`, `build_star_mask`. `apply_calibrations` and the
  duplicate `overscan_corr` definitions are deleted.
- **`create_lists.py`**: takes an `Instrument`; discovery, classification, filter, object and
  session logic delegate to it; writes fringe and dome lists; previous-night search uses
  `instrument.raw_dir(topdir, date)`; darks are only searched for when `dark` is a step.
- **`run.py` target loading**: applies the trim offset to configured star positions once, per
  section 4.10, and logs the conversion.
- **`master_calibrations.py`**: `open_frame` everywhere; `fringe` type; per-detector gain;
  drop `image_extension`, `runset_cut`, overscan column arguments; `liris.py` deleted.
- **`bad_pixel_map.py`**: optional dark; bad-column seeds.
- **`reduce_science.py`**: uses `Reducer`; writes trimmed images with the merged header plus
  `BANDSTEP` history cards and the record values.
- **`streaming_processor.py`**: uses `open_frame` and `Reducer`; fixes the `image_extension`
  section bug by construction; writes `frames.fits`.
- **`centroid.py`, `photometry.py`**: metadata from `FrameMeta` or `frames.fits`; the
  `BJD-OBS`/`MJD-OBS`/`ALTITUDE` lookups are removed. `extract_airmass` moves into
  `instrument.py`.
- **`photometry_analysis.py`**: `exptime`, `altitude` from `frames.fits`.
- **`precision_plots.py`**: takes `plate_scale`, `read_noise`, `saturation` and the instrument
  name from the instrument config instead of the hardcoded `plate_scale: 0.35` and
  `--instrument-name` default of `SPIRIT2`. `plate_scale` is documented in the README as an
  instrument key but is absent from every instrument YAML; it is added to the schema with no
  default, and `precision_plots` reports a clear error when it is missing rather than silently
  assuming a SPIRIT value.
- **`run.py`**: `load_instrument`; `setup_paths` via `instrument.raw_dir`; flat filter via
  `read_meta`; `to_make_fringe_map` flag; step and product validation before processing;
  passes `instrument` to every stage.
- **`utils.py`**: unchanged.
- **README**: instrument configuration section rewritten around this schema.

## 6.5 Cleanup carried out with this work

Three stale mechanisms are removed rather than carried through the refactor. Each was checked
before being listed here.

**Delete `photometry.translate_path_for_docker`** and its three use sites: `photometry.py:267`,
`photometry_analysis.py:532`, and the import in `streaming_processor.py:13`. It rewrites stored
paths only when `RUNNING_IN_DOCKER` is set; nothing sets that variable, in the Dockerfile, the
compose file or the batch script, so it returns its input unchanged in every context that
exists. It also targets an `/app/output` layout that no longer exists. Deleting it is a
behavioural no-op today. Container paths are made to work by the bind mount in
`docker-compose.server`, which maps the host tree onto the `topdir` the config names; if a
future mount disagrees, the fix belongs in the mount or in `topdir`, not in path rewriting.

**Delete `run_modular.py`.** It is a 695-line parallel copy of `run.py` last touched in June
2025, carrying its own `--local-staging` flag and Docker branch. Nothing imports it and it is
not the entry point in use. Keeping two divergent pipeline drivers through an abstraction
change guarantees they diverge further.

**Repair `batch_run.sh --staged`.** It reads raw data from `{topdir}/data/{inst}/{date}`, the
layout replaced in commit 0784982, so staged mode cannot find input against the current
`run.py`. Rather than update the hardcoded path, and so reintroduce a second copy of the layout
rules that this design has just centralised, `run.py` gains:

```
python run.py --print-paths 20170805.yaml
INST=WFC
DATE=20170805
RUN_NAME=WFC_20170805
RAWDIR=/Users/matthewhooton/data/WFC/20170805/download_dir
OUTDIR=/Users/matthewhooton/bandersnatch_runs/WFC_20170805
```

`get_config_paths()` in the batch script calls this instead of parsing YAML with three inline
`python3 -c` invocations and rebuilding paths itself. The script then has no knowledge of the
directory layout, and instrument-specific `raw_dir` templates work in staged mode for free.

**Ignore compiled Python.** `.gitignore` currently has `/__pycache__`, which matches only the
repository root, so `src/__pycache__/*.pyc` files are tracked. Replace with `__pycache__/` and
`*.pyc`, and remove the tracked files from the index.

## 7. Configuration changes for users

Date config additions, all optional:

```yaml
instrument_settings:
  detector: CCD4                    # overrides default_detector
processing_flags:
  to_make_fringe_map: true
calibration_params:
  force_backup_fringe_map: false    # true = always take the map from fringe_maps.yaml
  backup_fringe_maps:               # direct-path form, symmetric with backup_master_flats
    z: "/Users/matthewhooton/bandersnatch_runs/WFC_20170805/calib/1_fringe_map_z.fits"
paths:
  raw_dir: "{topdir}/WFC/{date}/download_dir"   # optional per-night override of the template
```

A new `configs/fringe_maps.yaml` provides date-based map reuse; see section 4.9. Like
`master_flats.yaml` it is optional, and its absence is not an error.

`calibration_params.image_extension` is deprecated; if present it is honoured as
`detectors.default.hdu` with a warning. `paths.topdir` for the WFC night is `~/data`.

`targets.yaml` entries gain `ra` and `dec`, used for the barycentric correction. Positions stay
in the convention the instrument declares, which for WFC is raw, untrimmed CCD4 pixels:

```yaml
KELT16:
  - start_date: "20170101"
    ra: "20:57:04.44"
    dec: "+31:39:39.6"
    tracking_star: 1
    all_star_positions:
      - [263.0, 1924.6]        # raw CCD4 coordinates; pipeline converts to (210.0, 1924.6)
      - ...
```

Existing SPECULOOS entries need no change: those instruments have no trim, so raw and trimmed
coordinates are identical, and `ra`/`dec` are optional with the pointing-centre fallback.

## 8. Testing and validation

- **Unit tests** (`tests/`, pytest) on synthetic FITS written in the test: a single-extension
  SPECULOOS-like file and a four-extension WFC-like file with `INHERIT`. Cover: classification
  rules, header merging, keyword fallbacks and expressions, section parsing, time conversion
  round trips, step-order validation, `Reducer` on a known image, `fit_fringe_scale` recovering
  a planted scale to better than 1 percent with and without template noise, the raw-to-trimmed
  coordinate conversion, `find_dated_entry` boundary cases, and refusal of a fringe map whose
  provenance keywords or shape do not match the run.
- **Regression**: run an existing SPIRIT or SINISTRO night before and after the refactor and
  diff `centroids.fits` and `photometry_aper*.fits` to numerical precision. This is the
  guarantee that the defaults are the legacy behaviour.
- **WFC end to end**: the 2017-08-05 night. Check masters against the scratch products from
  the design phase, confirm KELT-16 lands at (210, 1925) on the first frame, confirm the
  fringe scale runs from about 0.9 to 2 through the night, and compare light-curve scatter with
  and without the fringe step.

## 9. Implementation order

0. **Cleanup first**, since it is independent of the abstraction and shrinks what has to be
   refactored: delete `translate_path_for_docker` and `run_modular.py`, fix `.gitignore` and
   untrack the compiled Python. Verify an existing SPECULOOS night still runs. Commit
   separately so the refactor diff stays readable.
1. `instrument.py` with legacy defaults and tests; wire into `create_lists`, `run`,
   `master_calibrations`, `reduce_science`, `streaming_processor`, `centroid`, `photometry`.
   Regression run on a SPECULOOS night.
2. Geometry and `Reducer`: trim and overscan steps, coordinate conversion, product validation,
   delete `apply_calibrations` and the duplicate `overscan_corr`.
3. `WFC.yaml` and a date config for 2017-08-05; run lists, bias, flats; compare masters against
   the design-phase products.
4. Fringe map builder, `fringe_maps.yaml` and the shared date-selection helper, fringe step;
   full WFC run; light-curve comparison with and without the step.
5. `frames.fits`, `photometry_analysis` and `precision_plots` changes, `run.py --print-paths`
   and the `batch_run.sh` repair, README.

## 10. Decisions

Settled, September 2026:

1. **Raw directory.** `raw_dir` is a per-instrument template, overridable in the date config,
   so the existing `~/data/WFC/{date}/download_dir` layout is used as it stands and the same
   config works on the server. See section 3.5.
2. **Dark step for SPECULOOS.** The default recipe remains `[bias, dark, flat]`. Nothing
   changes for existing instruments.
3. **Target coordinates.** `ra` and `dec` go in `targets.yaml` and are the authoritative source
   for the BJD conversion. The pointing centre is a warned fallback only.
4. **Star position coordinates.** Positions are given in raw, untrimmed detector coordinates,
   because that is what is eyeballed before a first reduction. The pipeline converts once at
   load and works in trimmed coordinates thereafter. See section 4.10.
5. **One detector per run.** Multi-chip reduction is out of scope. A second chip is a second
   run with `instrument_settings.detector` changed.

Recommended, proceeding unless corrected:

6. **WFC flats from twilight sky frames only** (`flat_sources: [sky]`). Both dome and sky flats
   exist for 2017-08-05. Their small-scale structure agrees at a correlation of 0.945, and
   neither correlates with the fringe pattern (0.003 for sky, -0.014 for dome), so the sky flats
   are not fringe-contaminated and either would serve. Sky flats are preferred for their
   illumination pattern. The dome frames are still classified and listed, so switching is a
   one-line config change.

7. **Fringe maps are reusable across nights** via `configs/fringe_maps.yaml`, keyed by filter
   and detector, with the same date-range rule as `master_flats.yaml` and a shared
   implementation of that rule. Maps carry provenance keywords that are verified on load. See
   section 4.9.
8. **The stale-path cleanup is in scope for this work**: delete `translate_path_for_docker` and
   `run_modular.py`, repair `batch_run.sh --staged` via a new `run.py --print-paths`, and fix
   `.gitignore` so compiled Python is untracked. See section 6.5. `precision_plots.py` loses its
   hardcoded plate scale and instrument name at the same time.

No open questions remain. Implementation can proceed through section 9.

---

## 11. As built

Implemented September 2026 on branch `wfc-instrument-abstraction`. The design
above was followed; this section records where the implementation went beyond
it, and why.

### Verification

- **68 unit tests** in `tests/test_instrument.py`, covering FITS section
  parsing, keyword resolution and expressions, classification rule order, the
  detector selection and header merge, time conversion, step-order validation,
  the raw-to-trimmed coordinate conversion, dated-entry lookup, the reduction
  chain, fringe fitting and map construction, fringe-map provenance refusal,
  and the clipped stack combination.
- **Regression**: `tests/make_synthetic_night.py` generates a deterministic
  synthetic night in SPECULOOS and INT/WFC flavours. A full run on the
  SPECULOOS flavour was captured before any change and compared after. Master
  bias, master dark, master flat, bad pixel map, centroids and six aperture
  tables agree to float64 rounding: at worst 2e-15 relative in any table
  column and 4e-16 absolute in any master frame, against a float64 epsilon of
  2.2e-16. The only source of that difference is summation order in the
  vectorised stack combination.
- **INT/WFC end to end**: the real night of 2017-08-05, 514 frames, chip 4.
  Every check against an independent measurement passed:

  | Quantity | Pipeline | Independent measurement |
  |---|---|---|
  | KELT-16 position, trimmed | (210.1, 1926.0) | (210, 1925) from the Gaia plate solve |
  | Fringe template amplitude | 3.10 ADU robust std | 3.5 ADU |
  | Fringe scale through the night | 0.60 to 2.09, smoothly | 0.7 to 2.1, rising |
  | Bias drift across the night | 36 ADU | 10 to 50 ADU |
  | Airmass range | 1.00 to 1.97 | 1.00 to 1.97 |

  The fitted fringe scale correlates with the recorded sky level at +0.71 and
  varies smoothly frame to frame, which is the behaviour expected of OH
  airglow and evidence the fit is tracking signal rather than noise.

- **Does the fringe step earn its place?** The same night was reduced again
  with `fringe` removed from the recipe and nothing else changed. The best
  aperture improves from 1.381 to 1.325 parts per thousand, about 4 per cent,
  and the gain is concentrated at the larger apertures (5 to 7 per cent at
  r = 17 to 19 px) where more fringed sky falls inside the aperture. Across
  all apertures the median change is -1.1 per cent, with a few apertures
  marginally worse within the noise. That is a real improvement of the
  expected size and sign: the design-phase estimate of the uncorrected bias
  was 0.1 to 0.3 mmag per star at the start of the night, against a
  light-curve scatter of 1.4 mmag.

### Beyond the design

**Performance work, forced by frame size.** A WFC chip is 8.4 megapixels
against SPIRIT's 0.26, and three separate costs only became visible at that
scale:

- `medabsdevclip` looped in Python over every pixel, at four minutes per
  master frame. Vectorised in row chunks sized to bound the working set;
  about six seconds now.
- The fringe map stacked every frame twice, once for the filtered template and
  once for the unfiltered one. The high pass is applied after the median in
  both, so the filtered map follows from the unfiltered one.
- `gaussian_filter` at sigma 120 is a 961-tap convolution along each axis. The
  large-scale estimate is now computed on a decimated grid, 50 times faster,
  differing by under 2 per cent of the fringe amplitude.

**Bugs found and fixed in passing**, all in paths the streaming processor
bypassed and none introduced by this work:

- `centroid()` counted bad pixels using `results` before `centroid_loop` had
  assigned it, so the sequential path raised `NameError` on its first frame
  whenever a bad pixel map existed.
- `make_bad_pixel_map` subtracted the master bias from the master dark, but
  `make_master_calibration` already returns the dark bias-subtracted and
  normalised per second, so the bias frame's structure was folded into the
  detection thresholds.
- `medabsdevclip` with `nlimit >= nz - 1` produced an all-NaN master frame that
  propagated silently to NaN photometry. Now an error naming both numbers.
- `reduce_science_frames` collected per-frame records that `run.py` discarded.

**Plate scales.** `plate_scale` is now read from the instrument config rather
than hardcoded to 0.35 in `precision_plots.py`. It was absent from every existing
config, and the README's example value of 0.35 turned out to be the Andor
camera's, not SPIRIT's. Values were added from evidence rather than memory:
SPIRIT, SPIRIT2 and Callisto all describe the 1280 by 1024 SPIRIT array, whose
frame headers give 12 µm pixels at an 8.0 m focal length, hence 0.309; Io_ANDOR
is the 2088 by 2048 Andor iKon-L at the published 0.35; SINISTRO carries
`PIXSCALE = 0.389` in its headers. A one-character typo in Io_ANDOR.yaml
(`Io_ANDIR`) was corrected at the same time.
