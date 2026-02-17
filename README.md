# Bandersnatch Pipeline

A Python-based astronomical data reduction pipeline for time-series photometry of exoplanet targets. Performs automated calibration, centroiding, and aperture photometry on telescope imaging data.

## Features

- **Automated calibration**: Master bias, dark, and flat field creation with sigma-clipping
- **Intelligent flat field handling**: Separate dawn/dusk processing for improved star rejection
- **Precision centroiding**: Connected-component analysis with tracking star reference
- **Multi-aperture photometry**: Automated aperture optimization with differential photometry
- **Streaming processing**: Memory-efficient single-pass reduction+centroiding+photometry
- **Batch processing**: Process multiple nights with robust error handling
- **Comprehensive outputs**: Light curves, precision metrics, diagnostic plots

## Table of Contents

- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
  - [Date Configuration Files](#date-configuration-files)
  - [Instrument Configuration](#instrument-configuration)
  - [Targets Configuration](#targets-configuration)
  - [Master Flats Configuration](#master-flats-configuration)
- [Usage](#usage)
  - [Single Night Processing](#single-night-processing)
  - [Batch Processing](#batch-processing)
  - [Partial Pipeline Runs](#partial-pipeline-runs)
  - [Reprocessing Stages](#reprocessing-stages)
- [Pipeline Stages](#pipeline-stages)
- [Output Structure](#output-structure)
- [Post-Processing Tools](#post-processing-tools)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.11 or higher
- Conda or Mamba package manager
- ~50GB free disk space (for typical night's processing)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/mhooton/bandersnatch
cd bandersnatch
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate bandersnatch
```

3. **Verify installation**:
```bash
python src/run.py --help
```

## Directory Structure

The pipeline expects and creates the following directory structure:
```
{topdir}/                             # Base directory (set in config)
├── Observations/                    # Raw telescope data
│   └── {INSTRUMENT}/
│       └── images/
│           └── {YYYYMMDD}/
│               ├── file001.fits
│               ├── file002.fits
│               └── ...
│
├── bandersnatch_runs/               # Pipeline outputs
│   ├── configs/                     # Configuration files
│   │   ├── {YYYYMMDD}.yaml         # Date-specific configs
│   │   ├── SPIRIT2.yaml            # Instrument config
│   │   ├── targets.yaml            # Target positions/tracking stars
│   │   ├── master_flats.yaml       # Backup flat field config
│   │   ├── zYJ.csv                 # Filter transmission
│   │   ├── SPIRIT2.csv             # Detector QE curve
│   │   └── ml_40pc.txt             # Stellar catalog
│   │
│   └── {INSTRUMENT}_{YYYYMMDD}/    # Run output directory
│       ├── calib/                   # Master calibrations
│       │   ├── {RUN}_master_bias.fits
│       │   ├── {RUN}_master_dark.fits
│       │   ├── {RUN}_master_flat_{FILTER}.fits
│       │   ├── {RUN}_bias.list
│       │   ├── {RUN}_dark.list
│       │   ├── {RUN}_flat_{FILTER}.list
│       │   ├── {RUN}_image_{TARGET}.list
│       │   └── {RUN}_proc_{TARGET}.list
│       │
│       ├── {TARGET}/               # Per-target results
│       │   ├── {RUN}/              # Processed science frames
│       │   │   └── proc*.fits
│       │   ├── centroiding/
│       │   │   ├── centroids.fits
│       │   │   ├── poststamps_{STAR}.fits
│       │   │   └── plots/
│       │   └── photometry/
│       │       ├── photometry_aper{N}.fits
│       │       ├── differential_summary.txt
│       │       ├── precision_metrics.fits
│       │       └── plots/
│       │
│       └── logs/                   # Pipeline logs
│           └── {INSTRUMENT}_{YYYYMMDD}_{RUN}.log
```

### Key Directories

- **`data/`**: Raw FITS files from the telescope (organized by instrument and date)
- **`bandersnatch_runs/configs/`**: All configuration files
- **`bandersnatch_runs/{RUN_NAME}/`**: Output for each processed night
- **`calib/`**: Master calibration frames and file lists
- **`{TARGET}/`**: Per-target science products

## Configuration

The pipeline uses YAML configuration files. You need several configuration files:

### Date Configuration Files

**Location**: `~/bandersnatch_runs/configs/{YYYYMMDD}.yaml`

This is the main configuration file for each observing night. Example:
```yaml
# Date configuration for observations on 2025-12-15
instrument_settings:
  inst: "SPIRIT"              # Instrument name (must match instrument config)
  date: "20251215"            # Observation date (YYYYMMDD format)
  run: "1"                    # Run number (usually "1")
  discard_n_first_science: 0  # Number of initial science frames to discard

processing_flags:
  to_make_lists: true                # Create file lists from raw data
  to_make_master_bias: true         # Create master bias frame
  to_make_master_dark: true         # Create master dark frame
  to_make_master_flat: true         # Create master flat fields
  to_make_bad_pixel_map: false      # Create bad pixel map from calibrations
  to_reduce_science_images: true    # Apply calibrations to science frames
  to_centroid: true                 # Perform centroiding
  to_aperture_photometry: true      # Perform aperture photometry
  to_run_analysis: true             # Run precision analysis and diagnostic plots

calibration_params:
  image_extension: 0           # FITS extension containing image data
  bias_MAD_clip: 5            # Sigma clipping for bias combination
  bias_MAD_nlimit: 0          # Pixel limit for bias clipping
  dark_MAD_clip: 5            # Sigma clipping for dark combination
  dark_MAD_nlimit: 5          # Pixel limit for dark clipping
  flat_MAD_clip: 3            # Sigma clipping for flat combination
  flat_MAD_nlimit: 5          # Pixel limit for flat clipping
  
  # Skip creating calibrations if they already exist (optional)
  skip_existing_bias: false   # Skip bias if master file exists
  skip_existing_dark: false   # Skip dark if master file exists
  skip_existing_flat: false   # Skip flat if master file exists
  skip_existing_lists: false  # Skip lists if they already exist
  
  # Force using date-based backup flats from master_flats.yaml
  force_backup_flats: false   # true = always use backup flats
  
  # Backup calibration frames (optional - for nights with insufficient calibrations)
  backup_master_bias: "/path/to/backup_master_bias.fits"
  backup_master_dark: "/path/to/backup_master_dark.fits"
  backup_master_flats:
    zYJ: "/path/to/backup_master_flat_zYJ.fits"

centroid_settings:
  boxsize: 17                  # Half-width of centroiding box (pixels)
  nlimit_centroid: 5           # Pixel limit for sky background clipping
  clip_centroid: 5             # Sigma clipping for sky background
  sky_sigma: 5                 # Threshold for pixel selection (sigma above sky)
  flux_above_value: 500        # Flux threshold for diagnostics (ADU)
  mask_centroid_pixels: true   # Mask pixels used for centroiding in photometry
  
  # Star 0 positions for each target (DICTIONARY FORMAT - NOT LIST!)
  # Keys must match target names found in image lists
  star0_positions:
    TOI-696c: [465.20539, 661.97652]    # [x, y] position of target star
    Sp1255-5928: [512.3, 489.1]         # Another target if present

photometry_settings:
  PHPADU: 1.0                  # Photons per ADU (usually 1.0 if using gain)
  aper_min: 2                  # Smallest aperture radius (pixels)
  aper_max: 12                 # Largest aperture radius (pixels)
  SKYRAD_inner: 15             # Inner radius of sky annulus (pixels)
  SKYRAD_outer: 25             # Outer radius of sky annulus (pixels)
  sky_suppress: false          # Force sky to near-zero (for testing)

paths:
  topdir: "/data/SPECULOOSPipeline"  # Server path, or local mirror e.g. "/Volumes/my_drive"

# Optional: advanced streaming settings
save_processed_images_streaming: false  # Save intermediate processed images

logging:
  level: "INFO"                          # DEBUG, INFO, WARNING, ERROR
  console_output: true                   # Print to console
  file_output: true                      # Write to log file
  log_dir: "logs"                        # Log directory (relative to run output)
  log_filename: "{inst}_{date}_{run}.log"
  max_file_size_mb: 50                   # Log rotation size
  backup_count: 5                        # Number of log backups
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Instrument Configuration

**Location**: `~/bandersnatch_runs/configs/{INSTRUMENT}.yaml`

Defines instrument-specific parameters. Example for SPIRIT2:
```yaml
instrument_name: "SPIRIT2"
telescope_diameter: 0.2       # Telescope diameter (meters)
observatory_altitude: 2400    # Observatory altitude (meters)
gain: 1.5                     # Detector gain (electrons/ADU)
phpadu: 1.0                   # Photons per ADU
plate_scale: 0.35             # Arcseconds per pixel

scintillation:
  C_Y: 0.09                   # Scintillation constant
  H: 8000                     # Scale height (meters)
```

### Targets Configuration

**Location**: `~/bandersnatch_runs/configs/targets.yaml`

Defines reference star positions and tracking stars for each target. Now supports **date-based configurations** for targets that shift over time:
```yaml
targets:
  TOI-696c:
    - start_date: "20250101"              # Config valid from this date
      all_star_positions:                  # Reference positions [x, y]
        - [465.205, 661.977]               # Star 0 (target)
        - [523.45, 702.31]                 # Star 1 (comparison)
        - [412.89, 598.23]                 # Star 2 (comparison)
        - [550.12, 720.45]                 # Star 3 (comparison)
      tracking_star: 1                     # Index of tracking star (0-based)
    
    - start_date: "20251201"              # New config from Dec 2025
      all_star_positions:                  # Updated positions
        - [466.123, 663.045]               # Positions may drift over time
        - [524.234, 703.456]
        - [413.567, 599.123]
        - [551.234, 721.567]
      tracking_star: 1
  
  Sp1255-5928:
    - start_date: "20220101"              # Single config (no drift)
      all_star_positions:
        - [512.3, 489.1]                   # Star 0 (target)
        - [478.2, 512.8]                   # Star 1 (comparison)
        - [545.6, 456.3]                   # Star 2 (comparison)
      tracking_star: 1
```

**How it works**:
1. Pipeline finds which target(s) are in your data (from image filenames)
2. Looks up the target in `targets.yaml`
3. Finds the most recent `start_date` that is ≤ your observation date
4. Uses those reference positions
5. Calculates offset between your `star0_positions` (from date config) and reference
6. Applies that offset to all stars

**Legacy format** (single config, no dates):
```yaml
targets:
  TOI-696c:
    all_star_positions:
      - [465.205, 661.977]
      - [523.45, 702.31]
    tracking_star: 1
```

### Master Flats Configuration

**Location**: `~/bandersnatch_runs/configs/master_flats.yaml`

Defines date ranges for backup flat fields (similar to targets.yaml date-based system):
```yaml
filters:
  zYJ:
    - start_date: "20250101"
      path: "/Users/matthewhooton/bandersnatch_runs/SPIRIT2_20250115/calib/1_master_flat_zYJ.fits"
    
    - start_date: "20250501"
      path: "/Users/matthewhooton/bandersnatch_runs/SPIRIT2_20250515/calib/1_master_flat_zYJ.fits"
  
  I+z:
    - start_date: "20250101"
      path: "/Users/matthewhooton/bandersnatch_runs/SPIRIT2_20250120/calib/1_master_flat_I+z.fits"
```

The pipeline will:
1. Try to create flats from the current night's data
2. If insufficient flats, use the most recent backup from `master_flats.yaml`
3. If `force_backup_flats: true` in date config, always use backup flats

## Usage

### Single Night Processing

**Basic usage** (process everything):
```bash
conda activate bandersnatch
cd /path/to/bandersnatch
python src/run.py {topdir}/bandersnatch_runs/configs/20251215.yaml
```

This will:
1. Create file lists from raw data
2. Make master bias, dark, and flat frames
3. Reduce science images
4. Perform centroiding
5. Compute aperture photometry with differential photometry
6. Generate diagnostic plots and precision metrics

**Check outputs**:
```bash
# View log
less ~/bandersnatch_runs/SPIRIT_20251215/logs/SPIRIT_20251215_1.log

# Check calibrations
ls ~/bandersnatch_runs/SPIRIT_20251215/calib/

# View light curve summary
cat ~/bandersnatch_runs/SPIRIT_20251215/TOI-696c/photometry/differential_summary.txt
```

### Batch Processing

**Process multiple nights** from a list:

1. **Create a config list file** (`~/bandersnatch_runs/configs/december_runs.txt`):
```
20251201.yaml
20251202.yaml
20251203.yaml
20251215.yaml
```

2. **Run batch processing**:
```bash
cd /path/to/bandersnatch/src
./batch_run.sh --list december_runs.txt
```

**Alternative methods**:
```bash
# Process specific configs directly
./batch_run.sh 20251201.yaml 20251202.yaml 20251203.yaml

# Auto-discover all configs
./batch_run.sh --auto-discover

# With parallel processing (2 configs at once)
./batch_run.sh --list december_runs.txt --parallel 2

# Dry run (validate without processing)
./batch_run.sh --list december_runs.txt --dry-run

# Verbose output
./batch_run.sh --list december_runs.txt --verbose

# Resume from last failed run
./batch_run.sh --list december_runs.txt --resume
```

**Batch options**:
```bash
# Skip existing calibrations
./batch_run.sh --skip-existing-bias --skip-existing-dark --skip-existing-flat december_runs.txt

# Override aperture ranges
./batch_run.sh --aper-min 3 --aper-max 15 december_runs.txt

# Override centroiding boxsize
./batch_run.sh --boxsize 20 december_runs.txt

# Custom timeout (default 2 hours per config)
./batch_run.sh --timeout 3600 december_runs.txt  # 1 hour

# Override topdir
./batch_run.sh --topdir /mnt/external_drive december_runs.txt
```

**Batch output**:
```
~/bandersnatch_runs/batch_runs/batch_YYYYMMDD_HHMMSS/
├── summary_report.txt           # Success/failure summary
├── batch_run.log               # Complete batch log
├── 20251201.log                # Individual run logs
├── 20251201.status             # SUCCESS, FAILED, or TIMEOUT
├── 20251202.log
├── 20251202.status
└── ...
```

### Partial Pipeline Runs

**Run only specific stages**:
```bash
# Only create master calibrations (no science processing)
python src/run.py 20251215.yaml \
  --no-reduce-science \
  --no-centroid \
  --no-photometry

# Only centroiding and photometry (calibrations already exist)
python src/run.py 20251215.yaml \
  --no-make-lists \
  --no-make-bias \
  --no-make-dark \
  --no-make-flat \
  --no-reduce-science

# Only photometry (for reprocessing with different apertures)
python src/run.py 20251215.yaml \
  --no-make-lists \
  --no-make-bias \
  --no-make-dark \
  --no-make-flat \
  --no-reduce-science \
  --no-centroid \
  --aper-min 3 \
  --aper-max 20
```

**Note**: When disabling reduction, centroiding, OR photometry (partial workflow), the pipeline uses **traditional processing**. The memory-efficient **streaming processor** only runs when all three are enabled together.

### Reprocessing Stages

**Skip existing calibrations** (useful when rerunning after failures):
```bash
python src/run.py 20251215.yaml \
  --skip-existing-bias \
  --skip-existing-dark \
  --skip-existing-flat
```

**Force recreation** (opposite of skip):
```bash
python src/run.py 20251215.yaml \
  --no-skip-existing-bias \
  --no-skip-existing-dark \
  --no-skip-existing-flat
```

**Reprocess photometry with different parameters**:
```bash
# First run with default apertures (2-12 pixels)
python src/run.py 20251215.yaml

# Reprocess with larger apertures
python src/run.py 20251215.yaml \
  --no-make-lists \
  --no-make-bias \
  --no-make-dark \
  --no-make-flat \
  --no-reduce-science \
  --no-centroid \
  --aper-min 5 \
  --aper-max 25
```

**Reprocess centroiding with different boxsize**:
```bash
python src/run.py 20251215.yaml \
  --no-make-lists \
  --no-make-bias \
  --no-make-dark \
  --no-make-flat \
  --no-reduce-science \
  --boxsize 25
```

### Command-Line Overrides

**Override configuration settings**:
```bash
# Override date/instrument/run
python src/run.py 20251215.yaml --date 20251216 --inst SPIRIT2 --run 2

# Override logging
python src/run.py 20251215.yaml --verbose           # DEBUG level
python src/run.py 20251215.yaml --quiet             # WARNING level
python src/run.py 20251215.yaml --console-only      # No log file
python src/run.py 20251215.yaml --log-file custom.log

# Override centroiding
python src/run.py 20251215.yaml --boxsize 25

# Override photometry
python src/run.py 20251215.yaml --aper-min 3 --aper-max 20
```

## Pipeline Stages

### 1. File List Creation (`to_make_lists`)

**What it does**: Scans raw data directory and creates `.list` files categorizing images by type.

**Inputs**: 
- Raw FITS files in `~/data/{INSTRUMENT}/{DATE}/`

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/calib/
├── {RUN}_bias.list           # Bias frame paths
├── {RUN}_dark.list           # Dark frame paths
├── {RUN}_flat_{FILTER}.list  # Flat field paths (per filter)
└── {RUN}_image_{TARGET}.list # Science frame paths (per target)
```

**Features**:
- Auto-detects image types from FITS headers (`IMAGETYP`)
- Searches previous nights for calibrations if insufficient frames found
- Supports both `.fits` and `.fits.fz` (compressed) files
- Can discard first N science frames (`discard_n_first_science`)
- Creates separate dawn/dusk flat lists for improved star rejection

**Skip option**: `--skip-existing-lists` or `skip_existing_lists: true`

### 2. Master Bias Creation (`to_make_master_bias`)

**What it does**: Combines bias frames using sigma-clipped median.

**Inputs**: 
- `{RUN}_bias.list`

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/calib/
├── {RUN}_master_bias.fits    # Master bias frame
└── readoutnoise.txt          # Calculated readout noise (electrons)
```

**Algorithm**:
1. Load all bias frames
2. Apply sigma clipping (default: 5σ)
3. Compute median across frames
4. Calculate readout noise from first two bias frames

**Configuration**:
```yaml
calibration_params:
  bias_MAD_clip: 5      # Sigma threshold
  bias_MAD_nlimit: 0    # Pixel limit for clipping
```

**Skip option**: `--skip-existing-bias` or `skip_existing_bias: true`

**Backup**: If insufficient bias frames, uses `backup_master_bias` path from config.

### 3. Master Dark Creation (`to_make_master_dark`)

**What it does**: Combines bias-corrected dark frames, normalized by exposure time.

**Inputs**: 
- `{RUN}_dark.list`
- `{RUN}_master_bias.fits`

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/calib/
├── {RUN}_master_dark.fits    # Master dark frame (e/sec/pixel)
└── darkcurrent.txt           # Calculated dark current (e/sec/pixel)
```

**Algorithm**:
1. Load all dark frames
2. Subtract master bias
3. Normalize by exposure time
4. Apply sigma clipping
5. Compute median across frames

**Configuration**:
```yaml
calibration_params:
  dark_MAD_clip: 5      # Sigma threshold
  dark_MAD_nlimit: 5    # Pixel limit for clipping
```

**Skip option**: `--skip-existing-dark` or `skip_existing_dark: true`

**Backup**: If no dark frames, uses `backup_master_dark` path from config.

### 4. Master Flat Creation (`to_make_master_flat`)

**What it does**: Combines bias/dark-corrected flat fields with advanced star rejection.

**Inputs**: 
- `{RUN}_flat_{FILTER}_dawn.list` and/or `{RUN}_flat_{FILTER}_dusk.list`
- Or legacy: `{RUN}_flat_{FILTER}.list`
- `{RUN}_master_bias.fits`
- `{RUN}_master_dark.fits`

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/calib/
└── {RUN}_master_flat_{FILTER}.fits    # Master flat (one per filter)
```

**Algorithm** (dawn/dusk processing):
1. Process dawn flats separately:
   - Load, apply bias/dark corrections
   - Sigma clip and median combine
   - Normalize to median = 1
2. Process dusk flats separately (same steps)
3. If both sessions have ≥2 flats, median combine the two session masters
4. If only one session is good, use that session
5. Fallback to old-style processing if no dawn/dusk separation

**Why dawn/dusk separation?**: Stars appear in different positions in morning vs evening sky flats, improving rejection when combined.

**Configuration**:
```yaml
calibration_params:
  flat_MAD_clip: 3      # Sigma threshold (more aggressive)
  flat_MAD_nlimit: 5    # Pixel limit for clipping
  force_backup_flats: false  # Always use backup from master_flats.yaml
```

**Skip option**: `--skip-existing-flat` or `skip_existing_flat: true`

**Backup system**:
1. Try to create from current night's data
2. If insufficient flats → use `master_flats.yaml` (date-based lookup)
3. If not in `master_flats.yaml` → use `backup_master_flats` (legacy)
4. If `force_backup_flats: true` → skip step 1, go straight to backup

### 5. Science Frame Reduction (`to_reduce_science_images`)

**What it does**: Applies calibrations to science images.

**Inputs**: 
- `{RUN}_image_{TARGET}.list`
- `{RUN}_master_bias.fits`
- `{RUN}_master_dark.fits`
- `{RUN}_master_flat_{FILTER}.fits`

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/{TARGET}/{RUN}/
├── proc*.fits                        # Calibrated science frames
└── ../calib/{RUN}_proc_{TARGET}.list # List of processed frames
```

**Algorithm**:
```
reduced = (raw - bias - dark×exp_time) / flat
```

**Note**: With **streaming processing** enabled (all three flags: reduce + centroid + photometry), processed images are NOT saved to disk by default. Set `save_processed_images_streaming: true` to save them. Precision analysis and diagnostic plots require `to_run_analysis: true` as a separate step.

### 6. Centroiding (`to_centroid`)

**What it does**: Measures precise stellar positions using connected-component analysis.

**Inputs**: 
- Processed science frames (or raw frames in streaming mode)
- `star0_positions` from date config
- `targets.yaml` for all star positions and tracking star

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/{TARGET}/centroiding/
├── centroids.fits              # Table of positions vs time
├── poststamps_{STAR}.fits      # Image cutouts for each star
└── plots/                      # Diagnostic plots
    ├── xc_star_0.png
    ├── yc_star_0.png
    ├── x_width_star_0.png
    └── ...
```

**Algorithm**:
1. For each frame:
   - Extract postage stamp around each star
   - Measure sky background in annulus
   - Identify pixels >N sigma above sky
   - Use 4-connected component analysis on tracking star
   - Choose largest/central component
   - Calculate centroid of component
   - Apply same offset to all stars
2. Store positions, PSF widths, brightness, etc.

**Configuration**:
```yaml
centroid_settings:
  boxsize: 17                   # Postage stamp half-width
  sky_sigma: 5                  # Threshold for pixel selection
  mask_centroid_pixels: true    # Mask centroiding pixels in photometry
```

**Output table columns**:
- `BJD`: Barycentric Julian Date
- `File`: Path to science frame
- `airmass`: Airmass during observation
- `xc`, `yc`: Centroid positions (all stars)
- `x_bright`, `y_bright`, `val_bright`: Brightest pixel
- `x_width`, `y_width`: PSF FWHM estimates
- `n_pixels_above`: Pixels above flux threshold
- `sig3_pixels`, `sig5_pixels`, `sig10_pixels`: Pixels >N sigma

**Diagnostic plots**: Time series of all centroid metrics for each star.

### 7. Aperture Photometry (`to_aperture_photometry`)

**What it does**: Extracts flux in multiple apertures with differential photometry and outlier rejection.

**Inputs**: 
- Processed science frames
- `centroids.fits`
- Instrument config (gain, scintillation parameters)
- Calibration diagnostics (readout noise, dark current)

**Outputs**: 
```
~/bandersnatch_runs/{RUN_NAME}/{TARGET}/photometry/
├── photometry_aper{N}.fits        # One file per aperture radius
├── differential_summary.txt       # Best aperture, precision metrics
├── precision_metrics.fits         # Detailed table for all apertures
└── plots/                         # Light curves and diagnostics
    ├── diff_flux_aper{N}.png
    ├── flux_aper{N}_all_stars.png
    ├── flux_norm_aper{N}_star_{N}.png
    └── sky_all_stars.png
```

**Algorithm**:
1. For each aperture radius (aper_min to aper_max):
   - Extract flux in circular aperture at centroid position
   - Measure sky in annulus (SKYRAD_inner to SKYRAD_outer)
   - Subtract sky × area from flux
   - Convert ADU to electrons using gain
   
2. **Outlier detection** (rolling window, 4σ):
   - Detect outliers in each star's light curve
   - Remove frames where target is outlier
   - Remove frames where ALL comparison stars are outliers
   - Interpolate comparison star outliers (median filter)

3. **Comparison star optimization**:
   - Test all combinations of comparison stars
   - Choose combination that minimizes scatter
   - Create differential light curve: target / Σ(good_comparisons)

4. **Secondary differential cleaning**:
   - Apply rolling window outlier detection to differential flux
   - Remove remaining outliers

5. **Precision analysis**:
   - Calculate theoretical precision (photon noise, scintillation, sky, readout, dark)
   - Calculate observed precision (rolling 10-min window)
   - Identify best aperture

**Configuration**:
```yaml
photometry_settings:
  aper_min: 2               # Smallest aperture
  aper_max: 12              # Largest aperture
  SKYRAD_inner: 15          # Sky annulus inner radius
  SKYRAD_outer: 25          # Sky annulus outer radius
```

**Output table columns** (per aperture):
- `BJD`: Time
- `File`: Frame path
- `outlier_flag`: Bitmask of which stars were outliers
- `flux`, `flux_err`: Raw flux and error (all stars)
- `flux_norm`, `flux_norm_err`: Normalized flux
- `sky`, `sky_err`: Sky background
- `diff_flux`, `diff_flux_err`: Differential flux (target only)
- `good_star_mask`: Which comparison stars were used

**Differential summary** format:
```
# Main Summary
# Aperture_radius  Diff_flux_std_dev  N_frames_used  sigma_scint_value
              5        0.002345             145         1.234e-05
              6        0.002123             145         1.234e-05  <- Best
              7        0.002456             145         1.234e-05
              ...

# Star_0_data (target)
# Aperture_radius  sigma_prec  sky_background  sigma_obs_10min
              5    0.001234           125.3        0.002345
              6    0.001123           125.3        0.002123
              ...

# Star_1_data (comparison)
# ...
```

### 8. Post-Processing Analysis (`to_run_analysis`)

**What it does**: Runs precision analysis and generates diagnostic plots independently of the photometry extraction step.

**When to use**: This flag is independent of the photometry extraction. It can be re-run without re-extracting photometry — useful for regenerating plots or recalculating precision metrics after adjusting parameters.

**Inputs**:
- Existing `photometry_aper{N}.fits` files from a previous photometry run
- Instrument config (gain, scintillation parameters)
- Calibration diagnostics (readout noise, dark current)

**Outputs**:
~/bandersnatch_runs/{RUN_NAME}/{TARGET}/photometry/
├── differential_summary.txt       # Best aperture, precision metrics
├── precision_metrics.fits         # Detailed table for all apertures/stars
└── plots/                         # Light curves and diagnostics
├── diff_flux_aper{N}.png
├── flux_aper{N}_all_stars.png
├── flux_norm_aper{N}star{N}.png
└── sky_all_stars.png

**Algorithm**:
1. Load all existing `photometry_aper{N}.fits` tables
2. Calculate theoretical precision (photon noise, scintillation, sky, readout, dark)
3. Calculate observed precision (rolling 10-minute window)
4. Create summary files (`differential_summary.txt`, `precision_metrics.fits`)
5. Generate diagnostic plots

**Note**: Neither the streaming nor the traditional photometry extraction path runs analysis or generates plots automatically. You **must** set `to_run_analysis: true` to get precision metrics and diagnostic plots.

**Reprocessing example** (just regenerate analysis and plots):
```bash
python src/run.py 20251215.yaml \
  --no-make-lists \
  --no-make-bias \
  --no-make-dark \
  --no-make-flat \
  --no-reduce-science \
  --no-centroid \
  --no-photometry \
  --run-analysis
```

## Output Structure

### Calibration Outputs
```
~/bandersnatch_runs/{RUN_NAME}/calib/
├── 1_master_bias.fits           # Master bias frame
├── 1_master_dark.fits           # Master dark (e/sec/pixel)
├── 1_master_flat_zYJ.fits       # Master flat for zYJ filter
├── readoutnoise.txt             # Readout noise (electrons)
├── darkcurrent.txt              # Dark current (e/sec/pixel)
├── 1_bias.list                  # List of bias frames used
├── 1_dark.list                  # List of dark frames used
├── 1_flat_zYJ_dawn.list         # Dawn flat frames
├── 1_flat_zYJ_dusk.list         # Dusk flat frames
├── 1_image_TOI-696c.list        # Science frames for target
└── 1_proc_TOI-696c.list         # Processed science frames
```

### Science Outputs (per target)
```
~/bandersnatch_runs/{RUN_NAME}/TOI-696c/
├── 1/                           # Processed frames (if saved)
│   ├── procfile001.fits
│   ├── procfile002.fits
│   └── ...
│
├── centroiding/
│   ├── centroids.fits           # Main centroiding table
│   ├── poststamps_0.fits        # Target star cutouts (3D cube)
│   ├── poststamps_1.fits        # Comparison 1 cutouts
│   ├── poststamps_2.fits        # Comparison 2 cutouts
│   └── plots/
│       ├── xc_star_0.png        # X position vs time
│       ├── yc_star_0.png        # Y position vs time
│       ├── x_width_star_0.png   # PSF width vs time
│       └── ... (all metrics for all stars)
│
└── photometry/
    ├── photometry_aper2.fits    # Aperture 2 results
    ├── photometry_aper3.fits    # Aperture 3 results
    ├── ...
    ├── photometry_aper12.fits   # Aperture 12 results
    ├── differential_summary.txt # Best aperture summary
    ├── precision_metrics.fits   # Complete precision analysis
    └── plots/
        ├── diff_flux_aper6.png         # Differential light curve
        ├── flux_aper6_all_stars.png    # Raw flux (all stars)
        ├── flux_norm_aper6_star_0.png  # Normalized flux (target)
        ├── flux_norm_aper6_star_1.png  # Normalized flux (comp 1)
        └── sky_all_stars.png           # Sky background vs time
```

### Log Files
```
~/bandersnatch_runs/{RUN_NAME}/logs/
├── SPIRIT_20251215_1.log        # Main pipeline log
├── 20251215.yaml                # Copy of date config used
└── targets_used.yaml            # Copy of relevant target configs
```

## Post-Processing Tools

### Convert to Juliet Format

Create light curve files compatible with the `juliet` transit fitting package:
```bash
python src/create_juliet_lc.py \
  ~/                                    # topdir
  SPIRIT_20251215                       # run_name
  TOI-696c                              # target_name
  6                                     # aperture_number
  SPIRIT2                               # instrument_name
  -o lc.dat                             # output file (optional)
```

**Output** (`lc.dat`):
```
# BJD          flux         flux_err     instrument  theta1(sky)  theta2(xc)  theta3(yc)  theta4(x_width)  theta5(y_width)  theta6(airmass)
2459123.45678  1.00234e+05  234.5        SPIRIT2     0.234        -0.123      0.456       -0.234           0.123            0.345
2459123.45789  1.00189e+05  235.1        SPIRIT2     0.245        -0.134      0.467       -0.245           0.134            0.356
...
```

All theta columns are normalized to [-1, 1] range for use in GP regression.

### Generate Precision Plots

Create precision vs brightness plots for all targets:
```bash
python src/precision_plots.py \
  --topdir ~/                           # Base directory
  --config-dir ~/bandersnatch_runs/configs  # Config location
  --filter-name zYJ                     # Filter name
  --instrument-name SPIRIT2             # Instrument name
  --verbose                             # Show debug output
```

This will:
1. Auto-discover all processed runs
2. Find targets with completed photometry
3. Load stellar parameters from `ml_40pc.txt`
4. Generate precision vs flux plots using `mphot`
5. Save plots to `photometry/plots/precision_vs_brightness_{TARGET}_{RUN}.png`

**Options**:
```bash
--dry-run           # Show what would be processed without generating plots
--verbose           # Enable debug logging
```

## Advanced Features

### Streaming Processing

**What it is**: Memory-efficient single-pass processing of reduction → centroiding → photometry.

**When it's used**: Automatically enabled when all three flags are true:
```yaml
processing_flags:
  to_reduce_science_images: true
  to_centroid: true
  to_aperture_photometry: true
```

**Benefits**:
- Reduces disk I/O (doesn't write/read intermediate processed images)
- Lower memory footprint
- Faster processing
- Produces identical photometry tables to traditional processing

**Note**: Streaming processing extracts photometry but does not run precision analysis or generate diagnostic plots. Set `to_run_analysis: true` to run these as a separate step after extraction.

**Optionally save processed images**:
```yaml
save_processed_images_streaming: true  # Saves to {TARGET}/{RUN}/ directory
```

**When traditional processing is used**:
- Any of the three flags is false
- Running only certain stages (e.g., just photometry)

### Staged Pipeline (Batch Processing)

**What it is**: Three-stage pipelined processing for batch runs with slow storage.

**Use case**: Raw data on slow NAS, fast local SSD available for processing.

**How it works**:
```
Stage 1 (Data N):   Copy raw data to local disk
Stage 2 (Data N):   Process on local disk (fast I/O)
Stage 3 (Data N):   Copy results back to NAS

Timeline:
  [Copy N] → [Process N] → [Copy N back]
             [Copy N+1] → [Process N+1] → [Copy N+1 back]
                          [Copy N+2] → [Process N+2] → ...
```

**Usage**:
```bash
cd src/
./batch_run.sh \
  --staged \
  --staging-dir /mnt/fast_local_ssd \
  --list december_runs.txt
```

**Options**:
```bash
--staged                        # Enable staged processing
--staging-dir /path/to/fast/disk  # Local staging location
--max-retries 10               # Copy retry attempts (default: 10)
--retry-interval 60            # Seconds between retries (default: 60)
```

**How paths are handled**:
- Raw data copied from: `{topdir}/data/{inst}/{date}/` → `{staging-dir}/data/{inst}/{date}/`
- Processing happens in: `{staging-dir}/bandersnatch_runs/{run_name}/`
- Results copied back to: `{topdir}/bandersnatch_runs/{run_name}/`

### Force Using Backup Flats

If you have high-quality master flats and want to always use them:

1. Set up `master_flats.yaml` with date ranges
2. In your date config:
```yaml
calibration_params:
  force_backup_flats: true
```

This skips flat creation entirely and uses the date-based backup system.

### Masking Centroid Pixels

To avoid correlation between centroiding and photometry:
```yaml
centroid_settings:
  mask_centroid_pixels: true
```

This masks out the pixels used for centroiding before performing photometry. The masked pixels are replaced with median-filtered values.

## Troubleshooting

### Pipeline fails with "No flat field lists found"

**Problem**: No flat images were taken this night, or they're insufficient.

**Solutions**:
1. Check that flat images exist: `ls ~/data/{INSTRUMENT}/{DATE}/`
2. Verify flat images have correct `IMAGETYP` header
3. Use backup flats:
```yaml
   calibration_params:
     backup_master_flats:
       zYJ: "/path/to/good_master_flat_zYJ.fits"
```
4. Or set up `master_flats.yaml` for automatic date-based fallback

### Pipeline fails with "Target not found in targets.yaml"

**Problem**: Target name from science images doesn't match `targets.yaml`.

**Debug**:
```bash
# Check what targets were found
ls ~/bandersnatch_runs/{RUN_NAME}/calib/*_image_*.list

# Check targets.yaml
grep "^  " ~/bandersnatch_runs/configs/targets.yaml
```

**Solution**: Add target to `targets.yaml` or fix target name in FITS `OBJECT` header.

### Pipeline fails with "star0_positions must be dictionary"

**Problem**: Using old list format for `star0_positions`.

**Fix**: Change from:
```yaml
star0_positions:
  - [465.205, 661.977]
```

To:
```yaml
star0_positions:
  TOI-696c: [465.205, 661.977]
```

### "Mismatch: found N targets but M star0_positions"

**Problem**: Number of targets in data doesn't match `star0_positions` entries.

**Debug**:
```bash
# See what targets pipeline found
grep "Found targets" ~/bandersnatch_runs/{RUN_NAME}/logs/*.log

# Check your star0_positions
grep -A 5 "star0_positions:" ~/bandersnatch_runs/configs/{DATE}.yaml
```

**Solution**: Add/remove entries in `star0_positions` to match discovered targets.

### Centroiding produces NaN values

**Problem**: Stars are too faint or tracking star selection is wrong.

**Solutions**:
1. Check `boxsize` is appropriate (try increasing to 20-25)
2. Check `sky_sigma` threshold (try decreasing to 3-4)
3. Verify tracking star is brightest/most isolated star
4. Check raw images for clouds, cosmic rays, etc.

### Photometry scatter is high

**Problem**: Poor comparison star selection or bad frames.

**Debug**:
```bash
# Check differential_summary.txt
cat ~/bandersnatch_runs/{RUN_NAME}/{TARGET}/photometry/differential_summary.txt

# Look at diagnostic plots
open ~/bandersnatch_runs/{RUN_NAME}/{TARGET}/photometry/plots/diff_flux_aper*.png
```

**Solutions**:
1. Try different aperture radii (check differential_summary.txt for best)
2. Increase sky annulus: `SKYRAD_inner: 20, SKYRAD_outer: 30`
3. Check that comparison stars are similar brightness to target
4. Verify no clouds/moon/twilight in observations
5. Consider masking centroid pixels: `mask_centroid_pixels: true`

### "No images were successfully processed"

**Problem**: All images failed during processing.

**Debug**: Check the log file:
```bash
grep -i "error\|failed" ~/bandersnatch_runs/{RUN_NAME}/logs/*.log
```

**Common causes**:
- Wrong `image_extension` (try 0 or 1)
- Corrupted FITS files
- Missing calibration frames
- Disk space full

### Batch run gets stuck

**Problem**: One config hangs indefinitely.

**Solution**: Use timeout:
```bash
./batch_run.sh --timeout 3600 --list configs.txt  # 1 hour per config
```

### Permission denied errors

**Problem**: Pipeline can't write to output directory.

**Solutions**:
```bash
# Fix permissions
chmod -R u+w ~/bandersnatch_runs/

# Check disk space
df -h ~/

# Verify topdir path is correct
grep topdir ~/bandersnatch_runs/configs/{DATE}.yaml
```

### ImportError: No module named 'mphot'

**Problem**: Environment not properly set up.

**Solution**:
```bash
conda activate bandersnatch
pip install mphot
```

### Streaming processing is not saving processed images

**Expected behavior**: By default, streaming processing doesn't save intermediate files.

**To save them**:
```yaml
save_processed_images_streaming: true
```

Or use traditional processing (disable one of the three main flags).

---

## Questions or Issues?

If you encounter problems not covered here:

1. Check the log file in `~/bandersnatch_runs/{RUN_NAME}/logs/`
2. Run with `--verbose` for detailed debug output
3. Try a `--dry-run` with batch processing to validate configs
4. Check that all config files are present and properly formatted

## Contributing

When reporting issues, please include:
- Full command used
- Relevant sections of log file
- Configuration file (with paths anonymized if needed)
- Python/package versions: `conda list`