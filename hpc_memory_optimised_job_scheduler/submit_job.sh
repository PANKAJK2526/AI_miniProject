#!/bin/bash

# --- BATCH JOB PARAMETERS ---
# JOB_NAME="matrix_mul_test"
# GPUS=1
# ----------------------------

JOB_ID=$RANDOM
LOG_FILE="job_${JOB_ID}.log"
MONITOR_FILE="monitor_${JOB_ID}.csv"

echo "=========================================="
echo "   HPC JOB SUBMISSION SYSTEM (MOCK)       "
echo "=========================================="
echo "Job ID:      $JOB_ID"
echo "Output Log:  $LOG_FILE"
echo "Monitor Log: $MONITOR_FILE"
echo "Starting Time: $(date)"
echo "------------------------------------------"

# 1. Start Background Monitoring (Daemon)
# We use nvidia-smi query-gpu to log utilization every 100ms
echo "Timestamp, GPU_Util(%), Memory_Util(%)" > $MONITOR_FILE
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
           --format=csv,noheader,nounits -lms 100 >> $MONITOR_FILE &

MONITOR_PID=$!
echo "[SYSTEM] Monitoring Daemon started (PID $MONITOR_PID)"

# 2. Run the Main Application
# We redirect stdout/stderr to the log file (Batch mode)
echo "[SYSTEM] Launching Application..."
start_time=$(date +%s%N)

./hpc_pinned_stress >> $LOG_FILE 2>&1

end_time=$(date +%s%N)
duration=$(( ($end_time - $start_time) / 1000000 ))

# 3. Cleanup
echo "[SYSTEM] Application finished in ${duration} ms."
kill $MONITOR_PID
echo "[SYSTEM] Monitoring Daemon stopped."

echo "=========================================="
echo "JOB COMPLETED."
