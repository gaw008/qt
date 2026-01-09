#!/bin/bash
# ============================================
# Cold Data Archival Job Script
# Run via cron: 0 8 * * * /root/quant_system_full/scripts/run_archival.sh
# (8 UTC = 3 AM EST)
# ============================================

SCRIPT_DIR="/root/quant_system_full"
LOG_FILE="/var/log/quant_archival.log"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting archival job" >> "$LOG_FILE"

# Activate virtual environment
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Virtual environment not found" >> "$LOG_FILE"
    exit 1
fi

# Change to project directory
cd "$SCRIPT_DIR" || exit 1

# Run archival job
python -m dashboard.backend.archival_job >> "$LOG_FILE" 2>&1
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Archival job completed successfully" >> "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Archival job failed with exit code $RESULT" >> "$LOG_FILE"
fi

# Rotate log if too large (>10MB)
LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
if [ "$LOG_SIZE" -gt 10485760 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Log rotated" > "$LOG_FILE"
fi

exit $RESULT
