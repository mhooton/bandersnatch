#!/bin/bash

# Batch runner for bandersnatch pipeline with staged processing
# Usage: ./batch_run.sh [options] [config_files...]
#
# Examples:
#   ./batch_run.sh config1.yaml config2.yaml config3.yaml
#   ./batch_run.sh --list configs_list.txt
#   ./batch_run.sh --auto-discover
#   ./batch_run.sh --staged --staging-dir /tmp/bandersnatch config*.yaml

set -u  # Exit on undefined variables
set +e  # Don't exit on command failures (we handle them)

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run.py"
TIMEOUT_SECONDS=7200  # 2 hours default timeout
PARALLEL_JOBS=1
DRY_RUN=false
VERBOSE=false
RESUME_MODE=false
AUTO_DISCOVER=false
CONFIG_LIST_FILE=""
TOPDIR="/Users/matthewhooton"  # Default topdir
CONFIGS_DIR="${TOPDIR}/bandersnatch_runs/configs"
BATCH_DIR="${TOPDIR}/bandersnatch_runs/batch_runs"

# Pipeline arguments to pass through
PIPELINE_ARGS=()

# Staging configuration
USE_STAGED_PIPELINE=false
LOCAL_STAGING_DIR="/tmp/bandersnatch_staging"
MAX_COPY_RETRIES=10  # 10 minutes of retries
COPY_RETRY_INTERVAL=60  # 1 minute between retries

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Batch runner for bandersnatch pipeline

Usage: $0 [OPTIONS] [CONFIG_FILES...]

Options:
    -h, --help              Show this help message
    -l, --list FILE         Read config files from FILE (searches in configs directory if not found locally)
    -a, --auto-discover     Auto-discover all .yaml files in configs directory
    -t, --timeout SECONDS  Timeout for each run (default: $TIMEOUT_SECONDS)
    -p, --parallel JOBS     Run JOBS configs in parallel (default: $PARALLEL_JOBS)
    -d, --dry-run           Validate configs without running pipeline
    -v, --verbose           Verbose output
    -r, --resume            Resume from last failed/incomplete run
    --topdir DIR            Override topdir (default: $TOPDIR)
    --python-script PATH    Override python script path (default: $PYTHON_SCRIPT)

Pipeline Options:
    --staged                Enable staged pipeline with local processing
    --staging-dir DIR       Local staging directory (default: $LOCAL_STAGING_DIR)
    --max-retries NUM       Max retries for data copy operations (default: $MAX_COPY_RETRIES)
    --retry-interval SEC    Interval between copy retries (default: $COPY_RETRY_INTERVAL)
    --skip-existing-bias    Skip creating master bias if it already exists
    --no-skip-existing-bias Force create master bias even if it exists
    --skip-existing-dark    Skip creating master dark if it already exists
    --no-skip-existing-dark Force create master dark even if it exists
    --skip-existing-flat    Skip creating master flat if it already exists
    --no-skip-existing-flat Force create master flat even if it exists
    --skip-existing-lists   Skip creating file lists if they already exist
    --no-skip-existing-lists Force create file lists even if they exist
    --boxsize NUM           Override centroiding boxsize
    --aper-min NUM          Override minimum aperture radius
    --aper-max NUM          Override maximum aperture radius

Examples:
    $0 config1.yaml config2.yaml config3.yaml
    $0 --list my_configs.txt
    $0 --auto-discover --parallel 2
    $0 --staged --staging-dir /fast/local/storage config*.yaml

EOF
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -l|--list)
                CONFIG_LIST_FILE="$2"
                shift 2
                ;;
            -a|--auto-discover)
                AUTO_DISCOVER=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT_SECONDS="$2"
                shift 2
                ;;
            -p|--parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -r|--resume)
                RESUME_MODE=true
                shift
                ;;
            --topdir)
                TOPDIR="$2"
                CONFIGS_DIR="${TOPDIR}/bandersnatch_runs/configs"
                BATCH_DIR="${TOPDIR}/bandersnatch_runs/batch_runs"
                shift 2
                ;;
            --python-script)
                PYTHON_SCRIPT="$2"
                shift 2
                ;;
            --staged)
                USE_STAGED_PIPELINE=true
                shift
                ;;
            --staging-dir)
                LOCAL_STAGING_DIR="$2"
                shift 2
                ;;
            --max-retries)
                MAX_COPY_RETRIES="$2"
                shift 2
                ;;
            --retry-interval)
                COPY_RETRY_INTERVAL="$2"
                shift 2
                ;;
            --skip-existing-bias|--no-skip-existing-bias|--skip-existing-dark|--no-skip-existing-dark|--skip-existing-flat|--no-skip-existing-flat|--skip-existing-lists|--no-skip-existing-lists|--boxsize|--aper-min|--aper-max)
                PIPELINE_ARGS+=("$1")
                if [[ "$1" == "--boxsize" || "$1" == "--aper-min" || "$1" == "--aper-max" ]]; then
                    PIPELINE_ARGS+=("$2")
                    shift 2
                else
                    shift
                fi
                ;;
            -*)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                CONFIG_FILES+=("$1")
                shift
                ;;
        esac
    done
}

# Function to validate prerequisites
validate_prerequisites() {
    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi

    # Check if configs directory exists
    if [[ ! -d "$CONFIGS_DIR" ]]; then
        log_error "Configs directory not found: $CONFIGS_DIR"
        exit 1
    fi

    # Create batch runs directory
    mkdir -p "$BATCH_DIR"

    # Create staging directory if using staged pipeline
    if [[ "$USE_STAGED_PIPELINE" == "true" ]]; then
        mkdir -p "$LOCAL_STAGING_DIR"
        log_info "Using staged pipeline with staging directory: $LOCAL_STAGING_DIR"

        # Check available space
        local available_space=$(df "$LOCAL_STAGING_DIR" | awk 'NR==2 {print $4}')
        log_info "Available space in staging directory: ${available_space}KB"
    fi

    # Check if timeout command is available
    if ! command -v timeout >/dev/null 2>&1; then
        log_warning "timeout command not available, timeouts will not work"
        TIMEOUT_SECONDS=0
    fi
}

# State management functions for staged pipeline
get_state_file() {
    local config="$1"
    local stage="$2"  # raw_copying, raw_ready, processing, output_copying, complete
    local config_basename=$(basename "$config" .yaml)
    echo "${BATCH_RUN_DIR}/${config_basename}.${stage}"
}

set_state() {
    local config="$1"
    local stage="$2"
    touch "$(get_state_file "$config" "$stage")"
}

check_state() {
    local config="$1"
    local stage="$2"
    [[ -f "$(get_state_file "$config" "$stage")" ]]
}

clear_state() {
    local config="$1"
    local stage="$2"
    local state_file=$(get_state_file "$config" "$stage")
    [[ -f "$state_file" ]] && rm "$state_file"
}

# Function to extract paths from config file
get_config_paths() {
    local config_file="$1"
    local inst date topdir_config

    # Extract instrument and date from config
    inst=$(python3 -c "import yaml; config=yaml.safe_load(open('$config_file')); print(config['instrument_settings']['inst'])" 2>/dev/null)
    date=$(python3 -c "import yaml; config=yaml.safe_load(open('$config_file')); print(config['instrument_settings']['date'])" 2>/dev/null)
    topdir_config=$(python3 -c "import yaml; config=yaml.safe_load(open('$config_file')); print(config['paths']['topdir'])" 2>/dev/null)

    if [[ -z "$inst" || -z "$date" || -z "$topdir_config" ]]; then
        log_error "Failed to extract paths from config: $config_file"
        return 1
    fi

    # Expand tilde in topdir_config
    topdir_config=$(eval echo "$topdir_config")

    echo "$inst" "$date" "$topdir_config"
}

# Function to copy raw data to local staging
copy_raw_data() {
    local config="$1"
    local config_basename=$(basename "$config" .yaml)

    # Get paths from config
    local paths=($(get_config_paths "$config"))
    if [[ ${#paths[@]} -ne 3 ]]; then
        log_error "Failed to get paths for config: $config"
        return 1
    fi

    local inst="${paths[0]}"
    local date="${paths[1]}"
    local topdir_config="${paths[2]}"

    local raw_source_dir="${topdir_config}/data/${inst}/${date}"
    local local_staging_base="${LOCAL_STAGING_DIR}/${config_basename}"
    local local_raw_dir="${local_staging_base}/raw"

    log_info "Starting raw data copy for $config"
    log_info "  Source: $raw_source_dir"
    log_info "  Destination: $local_raw_dir"

    # Check if source directory exists
    if [[ ! -d "$raw_source_dir" ]]; then
        log_error "Raw data source directory not found: $raw_source_dir"
        return 1
    fi

    # Create staging directory structure
    mkdir -p "$local_raw_dir"
    mkdir -p "${local_staging_base}/data/${inst}/${date}"

    # Copy raw data
    if cp -r "$raw_source_dir"/* "${local_staging_base}/data/${inst}/${date}"/; then
        set_state "$config" "raw_ready"
        log_success "Raw data copy completed for $config"
        return 0
    else
        log_error "Raw data copy failed for $config"
        return 1
    fi
}

# Function to copy output data back to external storage
copy_output_data() {
    local config="$1"
    local config_basename=$(basename "$config" .yaml)

    # Get paths from config
    local paths=($(get_config_paths "$config"))
    if [[ ${#paths[@]} -ne 3 ]]; then
        log_error "Failed to get paths for config: $config"
        return 1
    fi

    local inst="${paths[0]}"
    local date="${paths[1]}"
    local topdir_config="${paths[2]}"

    local run_name="${inst}_${date}"
    local local_output_dir="${LOCAL_STAGING_DIR}/${config_basename}/bandersnatch_runs/${run_name}"
    local output_dest_dir="${topdir_config}/bandersnatch_runs/${run_name}"

    log_info "Starting output data copy for $config"
    log_info "  Source: $local_output_dir"
    log_info "  Destination: $output_dest_dir"

    # Check if local output directory exists
    if [[ ! -d "$local_output_dir" ]]; then
        log_error "Local output directory not found: $local_output_dir"
        return 1
    fi

    # Create destination directory
    mkdir -p "$output_dest_dir"

    # Copy output data
    if cp -r "$local_output_dir"/* "$output_dest_dir"/; then
        # Cleanup local staging after successful copy
        rm -rf "${LOCAL_STAGING_DIR}/${config_basename}"
        set_state "$config" "complete"
        log_success "Output data copy completed and local staging cleaned up for $config"
        return 0
    else
        log_error "Output data copy failed for $config"
        return 1
    fi
}

# Function to wait for raw data to be ready with retry logic
wait_for_raw_data_ready() {
    local config="$1"
    local retries=0

    log_info "Waiting for raw data to be ready for: $config"

    while [[ $retries -lt $MAX_COPY_RETRIES ]]; do
        if check_state "$config" "raw_ready"; then
            log_success "Raw data ready for: $config"
            return 0
        fi

        log_info "Raw data not ready for $config, waiting... (attempt $((retries + 1))/$MAX_COPY_RETRIES)"
        sleep $COPY_RETRY_INTERVAL
        retries=$((retries + 1))
    done

    log_error "Raw data copy timed out for $config after $MAX_COPY_RETRIES attempts"
    log_error "Skipping this config and moving to next"
    return 1
}

# Function to wait for previous output transfer to complete
wait_for_previous_output_complete() {
    local prev_config="$1"

    if [[ -n "$prev_config" ]] && ! check_state "$prev_config" "complete"; then
        log_info "Waiting for previous output transfer to complete: $prev_config"
        while ! check_state "$prev_config" "complete"; do
            sleep 30  # Check every 30 seconds
        done
        log_success "Previous output transfer completed: $prev_config"
    fi
}

# Function to process a config using local staging
process_config_locally() {
    local config="$1"
    local job_id="$2"
    local total_configs="$3"
    local config_basename=$(basename "$config" .yaml)
    local staging_base="${LOCAL_STAGING_DIR}/${config_basename}"

    local status_file="${BATCH_RUN_DIR}/${config_basename}.status"
    local log_file="${BATCH_RUN_DIR}/${config_basename}.log"
    local start_time=$(date +%s)

    log_info "[$job_id/$total_configs] Starting local processing: $config"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "SUCCESS" > "$status_file"
        log_success "[$job_id/$total_configs] DRY RUN: $config"
        return 0
    fi

    # Prepare command with staging arguments
    local cmd="python3 '$PYTHON_SCRIPT' '$config' --local-staging --staging-dir '$staging_base'"

    # Add pipeline arguments
    for arg in "${PIPELINE_ARGS[@]}"; do
        cmd="$cmd '$arg'"
    done

    if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
        cmd="timeout ${TIMEOUT_SECONDS}s $cmd"
    fi

    # Run the command
    local exit_code=0
    if [[ "$VERBOSE" == "true" ]]; then
        eval "$cmd" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
    else
        eval "$cmd" > "$log_file" 2>&1
        exit_code=$?
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Determine status
    local status="SUCCESS"
    local message=""

    if [[ $exit_code -eq 124 ]]; then
        status="TIMEOUT"
        message="Timed out after ${TIMEOUT_SECONDS}s"
    elif [[ $exit_code -ne 0 ]]; then
        status="FAILED"
        message="Exit code: $exit_code"
    else
        message="Completed in ${duration}s"
    fi

    echo "$status" > "$status_file"

    if [[ "$status" == "SUCCESS" ]]; then
        log_success "[$job_id/$total_configs] $config - $message"
    else
        log_error "[$job_id/$total_configs] $config - $status: $message"
    fi

    return $exit_code
}

# Function to run staged pipeline
run_staged_pipeline() {
    local configs=("$@")
    local total_configs=${#configs[@]}
    local current_index=0
    local processing_config=""
    local next_config=""
    local prev_config=""

    log_info "Starting staged pipeline with ${total_configs} configs"

    # Start copying first dataset
    if [[ ${#configs[@]} -gt 0 ]]; then
        next_config="${configs[0]}"
        set_state "$next_config" "raw_copying"
        copy_raw_data "$next_config" &
        log_info "Started copying raw data for first config: $next_config"
    fi

    while [[ $current_index -lt $total_configs ]]; do
        processing_config="$next_config"
        prev_config=""
        if [[ $current_index -gt 0 ]]; then
            prev_config="${configs[$((current_index - 1))]}"
        fi
        current_index=$((current_index + 1))

        # Determine next config to start copying
        if [[ $current_index -lt $total_configs ]]; then
            next_config="${configs[$current_index]}"
        else
            next_config=""
        fi

        # Wait for current config's raw data to be ready
        if ! wait_for_raw_data_ready "$processing_config"; then
            log_error "Skipping processing for $processing_config due to raw data copy failure"
            continue
        fi

        # Wait for previous output transfer to complete before starting next raw copy
        wait_for_previous_output_complete "$prev_config"

        # Start copying next config's raw data (if exists)
        if [[ -n "$next_config" ]]; then
            set_state "$next_config" "raw_copying"
            copy_raw_data "$next_config" &
            log_info "Started copying raw data for next config: $next_config"
        fi

        # Process current config locally
        set_state "$processing_config" "processing"
        if process_config_locally "$processing_config" "$current_index" "$total_configs"; then
            # Start output copy in background
            set_state "$processing_config" "output_copying"
            copy_output_data "$processing_config" &
            log_info "Started copying output data for: $processing_config"
        else
            log_error "Processing failed for $processing_config, skipping output copy"
            # Clean up local staging for failed run
            local config_basename=$(basename "$processing_config" .yaml)
            rm -rf "${LOCAL_STAGING_DIR}/${config_basename}"
        fi
    done

    # Wait for final output copy to complete
    log_info "Waiting for final output transfer to complete..."
    if [[ ${#configs[@]} -gt 0 ]]; then
        local last_config="${configs[$((${#configs[@]} - 1))]}"
        wait_for_previous_output_complete "$last_config"
    fi

    log_success "Staged pipeline completed!"
}

# Function to discover config files
discover_configs() {
    local configs=()

    if [[ "$AUTO_DISCOVER" == "true" ]]; then
        log_info "Auto-discovering config files in $CONFIGS_DIR" >&2
        while IFS= read -r -d '' file; do
            configs+=("$(basename "$file")")
        done < <(find "$CONFIGS_DIR" -name "*.yaml" -o -name "*.yml" -print0 2>/dev/null | sort -z)

    elif [[ -n "$CONFIG_LIST_FILE" ]]; then
        log_info "Reading config files from $CONFIG_LIST_FILE" >&2

        # Check multiple locations for the list file
        local list_file_path=""
        if [[ -f "$CONFIG_LIST_FILE" ]]; then
            list_file_path="$CONFIG_LIST_FILE"
        elif [[ -f "$CONFIGS_DIR/$CONFIG_LIST_FILE" ]]; then
            list_file_path="$CONFIGS_DIR/$CONFIG_LIST_FILE"
        else
            log_error "Config list file not found: $CONFIG_LIST_FILE" >&2
            log_error "  Searched in: current directory, $CONFIGS_DIR" >&2
            return 1
        fi

        log_info "Using list file: $list_file_path" >&2

        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            configs+=("$line")
        done < "$list_file_path"

    elif [[ ${#CONFIG_FILES[@]} -gt 0 ]]; then
        configs=("${CONFIG_FILES[@]}")

    else
        log_error "No config files specified. Use --help for usage information." >&2
        return 1
    fi

    if [[ ${#configs[@]} -eq 0 ]]; then
        log_error "No config files found" >&2
        return 1
    fi

    # Output configs one per line for proper array capture
    printf '%s\n' "${configs[@]}"
}

# Function to validate config files exist
validate_configs() {
    local configs=("$@")
    local valid_configs=()

    for config in "${configs[@]}"; do
        local config_path=""

        # Check multiple locations for config file
        if [[ -f "$config" ]]; then
            config_path="$config"
        elif [[ -f "$CONFIGS_DIR/$config" ]]; then
            config_path="$CONFIGS_DIR/$config"
        elif [[ -f "$(basename "$config")" ]]; then
            config_path="$(basename "$config")"
        else
            log_error "Config file not found: $config" >&2
            log_error "  Searched in: current directory, $CONFIGS_DIR" >&2
            continue
        fi

        # Validate YAML syntax if dry run
        if [[ "$DRY_RUN" == "true" ]]; then
            if python3 -c "import yaml; yaml.safe_load(open('$config_path'))" 2>/dev/null; then
                log_success "Valid YAML: $config" >&2
            else
                log_error "Invalid YAML syntax: $config" >&2
                continue
            fi
        fi

        # Use full path for staging pipeline
        if [[ "$USE_STAGED_PIPELINE" == "true" ]]; then
            if [[ "$config_path" == "$config" && ! "$config" =~ ^/ ]]; then
                # Convert relative path to absolute
                config_path="$CONFIGS_DIR/$config"
            fi
        fi

        valid_configs+=("$config_path")
    done

    if [[ ${#valid_configs[@]} -eq 0 ]]; then
        log_error "No valid config files found" >&2
        return 1
    fi

    # Output configs one per line for proper array capture
    printf '%s\n' "${valid_configs[@]}"
}

# Function to create batch run directory and files
setup_batch_run() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    BATCH_RUN_DIR="${BATCH_DIR}/batch_${timestamp}"

    # If resuming, find the latest batch run directory
    if [[ "$RESUME_MODE" == "true" ]]; then
        local latest_batch=$(find "$BATCH_DIR" -maxdepth 1 -type d -name "batch_*" | sort | tail -1)
        if [[ -n "$latest_batch" ]]; then
            BATCH_RUN_DIR="$latest_batch"
            log_info "Resuming batch run: $BATCH_RUN_DIR"
        else
            log_warning "No previous batch run found, starting new batch"
        fi
    fi

    mkdir -p "$BATCH_RUN_DIR"

    # Initialize summary files
    SUMMARY_FILE="${BATCH_RUN_DIR}/summary_report.txt"
    BATCH_LOG="${BATCH_RUN_DIR}/batch_run.log"

    # Write header to summary file
    cat > "$SUMMARY_FILE" << EOF
Bandersnatch Batch Run Summary
==============================
Start Time: $(date)
Batch Directory: $BATCH_RUN_DIR
Python Script: $PYTHON_SCRIPT
Timeout: ${TIMEOUT_SECONDS}s
Parallel Jobs: $PARALLEL_JOBS
Dry Run: $DRY_RUN
Staged Pipeline: $USE_STAGED_PIPELINE
Staging Directory: $LOCAL_STAGING_DIR

Config Files:
EOF

    # Start batch log
    exec 1> >(tee -a "$BATCH_LOG")
    exec 2> >(tee -a "$BATCH_LOG" >&2)
}

# Function to run a single config (original sequential method)
run_single_config() {
    local config="$1"
    local job_id="$2"
    local total_configs="$3"

    local config_basename=$(basename "$config" .yaml)
    local status_file="${BATCH_RUN_DIR}/${config_basename}.status"
    local log_file="${BATCH_RUN_DIR}/${config_basename}.log"
    local start_time=$(date +%s)

    # Check if already completed in resume mode
    if [[ "$RESUME_MODE" == "true" && -f "$status_file" ]]; then
        local existing_status=$(cat "$status_file" 2>/dev/null)
        if [[ "$existing_status" == "SUCCESS" ]]; then
            log_info "[$job_id/$total_configs] Skipping completed config: $config"
            return 0
        fi
    fi

    log_info "[$job_id/$total_configs] Starting: $config"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "SUCCESS" > "$status_file"
        log_success "[$job_id/$total_configs] DRY RUN: $config"
        return 0
    fi

    # Prepare command
    local cmd="python3 '$PYTHON_SCRIPT' '$config'"

    # Add pipeline arguments
    for arg in "${PIPELINE_ARGS[@]}"; do
        cmd="$cmd '$arg'"
    done

    if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
        cmd="timeout ${TIMEOUT_SECONDS}s $cmd"
    fi

    # Run the command
    local exit_code=0
    if [[ "$VERBOSE" == "true" ]]; then
        eval "$cmd" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}
    else
        eval "$cmd" > "$log_file" 2>&1
        exit_code=$?
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Determine status
    local status="SUCCESS"
    local message=""

    if [[ $exit_code -eq 124 ]]; then
        status="TIMEOUT"
        message="Timed out after ${TIMEOUT_SECONDS}s"
    elif [[ $exit_code -ne 0 ]]; then
        status="FAILED"
        message="Exit code: $exit_code"
    else
        message="Completed in ${duration}s"
    fi

    echo "$status" > "$status_file"

    if [[ "$status" == "SUCCESS" ]]; then
        log_success "[$job_id/$total_configs] $config - $message"
    else
        log_error "[$job_id/$total_configs] $config - $status: $message"
    fi

    return $exit_code
}

# Function to run configs in parallel (original method)
run_configs_parallel() {
    local configs=("$@")
    local total_configs=${#configs[@]}
    local job_count=0
    local pids=()
    local results=()

    for config in "${configs[@]}"; do
        ((job_count++))

        # Wait for available slot if at max parallel jobs
        while [[ ${#pids[@]} -ge $PARALLEL_JOBS ]]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    results[$i]=$?
                    unset pids[$i]
                fi
            done
            pids=("${pids[@]}")  # Reindex array
            sleep 1
        done

        # Start new job
        run_single_config "$config" "$job_count" "$total_configs" &
        pids+=($!)

        if [[ "$VERBOSE" == "true" ]]; then
            log_info "Started job ${#pids[@]}/$PARALLEL_JOBS for config: $config"
        fi
    done

    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
        results+=($?)
    done

    return 0
}

# Function to run configs sequentially (original method)
run_configs_sequential() {
    local configs=("$@")
    local total_configs=${#configs[@]}
    local job_count=0
    local failed_count=0

    for config in "${configs[@]}"; do
        ((job_count++))

        if ! run_single_config "$config" "$job_count" "$total_configs"; then
            ((failed_count++))
        fi

        # Show progress
        local completed_percent=$((job_count * 100 / total_configs))
        log_info "Progress: $job_count/$total_configs ($completed_percent%) completed"
    done

    return $failed_count
}

# Function to generate final summary
generate_summary() {
    local configs=("$@")
    local total_configs=${#configs[@]}
    local success_count=0
    local failed_count=0
    local timeout_count=0
    local skipped_count=0

    echo "" >> "$SUMMARY_FILE"
    echo "Results:" >> "$SUMMARY_FILE"
    echo "--------" >> "$SUMMARY_FILE"

    for config in "${configs[@]}"; do
        local config_basename=$(basename "$config" .yaml)
        local status_file="${BATCH_RUN_DIR}/${config_basename}.status"
        local status="UNKNOWN"

        if [[ -f "$status_file" ]]; then
            status=$(cat "$status_file")
        fi

        case "$status" in
            SUCCESS) ((success_count++)) ;;
            FAILED) ((failed_count++)) ;;
            TIMEOUT) ((timeout_count++)) ;;
            SKIPPED) ((skipped_count++)) ;;
        esac

        printf "%-50s %s\n" "$config" "$status" >> "$SUMMARY_FILE"
    done

    # Summary statistics
    cat >> "$SUMMARY_FILE" << EOF

Summary Statistics:
------------------
Total Configs: $total_configs
Successful: $success_count
Failed: $failed_count
Timeout: $timeout_count
Skipped: $skipped_count

End Time: $(date)
Batch Directory: $BATCH_RUN_DIR
EOF

    # Print summary to console
    log_info "Batch run completed!"
    log_info "Results: $success_count success, $failed_count failed, $timeout_count timeout, $skipped_count skipped"
    log_info "Summary report: $SUMMARY_FILE"
    log_info "Batch log: $BATCH_LOG"

    # Return non-zero if any failures
    if [[ $failed_count -gt 0 || $timeout_count -gt 0 ]]; then
        return 1
    fi
    return 0
}

# Main function
main() {
    local CONFIG_FILES=()

    # Parse arguments
    parse_args "$@"

    # Validate prerequisites
    validate_prerequisites

    # Discover and validate config files
    local discovered_configs=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && discovered_configs+=("$line")
    done < <(discover_configs)

    if [[ ${#discovered_configs[@]} -eq 0 ]]; then
        log_error "No config files discovered"
        exit 1
    fi

    log_info "DEBUG: Discovered ${#discovered_configs[@]} configs: ${discovered_configs[*]}"

    local valid_configs=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && valid_configs+=("$line")
    done < <(validate_configs "${discovered_configs[@]}")

    log_info "DEBUG: Validated ${#valid_configs[@]} configs: ${valid_configs[*]}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would process ${#valid_configs[@]} config files"
        for config in "${valid_configs[@]}"; do
            echo "  - $config"
        done
        exit 0
    fi

    # Setup batch run directory
    setup_batch_run

    # Add configs to summary file
    for config in "${valid_configs[@]}"; do
        echo "  - $config" >> "$SUMMARY_FILE"
    done

    log_info "Starting batch run with ${#valid_configs[@]} config files"
    log_info "Batch directory: $BATCH_RUN_DIR"

    # Choose pipeline type
    if [[ "$USE_STAGED_PIPELINE" == "true" ]]; then
        log_info "Using staged pipeline with local processing"
        run_staged_pipeline "${valid_configs[@]}"
    else
        log_info "Using original pipeline"
        # Run configs using original method
        if [[ $PARALLEL_JOBS -gt 1 ]]; then
            log_info "Running with $PARALLEL_JOBS parallel jobs"
            run_configs_parallel "${valid_configs[@]}"
        else
            run_configs_sequential "${valid_configs[@]}"
        fi
    fi

    # Generate final summary
    generate_summary "${valid_configs[@]}"
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_success "All configs completed successfully!"
    else
        log_error "Some configs failed or timed out"
    fi

    exit $exit_code
}

# Run main function with all arguments
main "$@"