#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$PROJECT_DIR/EvoTreeNAD"

# ======== Default Arguments ==========
CUDA_DEVICES=${1:-0}
RUN_NAME=${2:-default_run}
CONFIG_DIR=${3:-$SCRIPT_DIR/fds1}
CONFIG_PATH=${4:-$CONFIG_DIR/run_config_osscode_gpt41idea.json}
BATCH_ITER_NUM=${5:-50}
LOG_DIR=${6:-$CONFIG_DIR}


# ======== Ensure log directory exists ==========
mkdir -p "$LOG_DIR"

# ======== Timestamp & Log File ==========
timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${RUN_NAME}_${CUDA_DEVICES}_BatchIter${BATCH_ITER_NUM}_${timestamp}.log"

# ======== Metadata Header ==========
{
    echo "Run Name     : $RUN_NAME"
    echo "CUDA Devices : $CUDA_DEVICES"
    echo "Config Dir   : $CONFIG_DIR"
    echo "Config Path  : $CONFIG_PATH"
    echo "Log Dir      : $LOG_DIR"
    echo "Batch Iterations   : $BATCH_ITER_NUM"
    echo "Start Time   : $(date +"%Y-%m-%d %H:%M:%S")"
    echo "============================="
} > "$LOG_FILE"

# ======== Run the Python Script ==========
(
    python3 -u "$CODE_DIR/main_run.py" \
        --cuda "$CUDA_DEVICES" \
        --run_name "$RUN_NAME" \
        --config_dir "$CONFIG_DIR" \
        --customized_run_config_path "$CONFIG_PATH" \
        --batch_iter_num "$BATCH_ITER_NUM" \

    echo "============================="
    echo "End Time     : $(date +"%Y-%m-%d %H:%M:%S")"
) >> "$LOG_FILE" 2>&1 &

# ======== PID Info ==========
pid=$!
echo "Script running in background (PID: $pid)"
echo "Log file: $LOG_FILE"
echo "Watch with: tail -f \"$LOG_FILE\""

