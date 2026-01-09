# Monitoring Tools Automation Guide

Comprehensive guide for automating the strategy monitoring tools with scheduled tasks.

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Tools](#monitoring-tools)
3. [Recommended Schedule](#recommended-schedule)
4. [Windows Setup (Task Scheduler)](#windows-setup-task-scheduler)
5. [Linux/Mac Setup (Cron)](#linuxmac-setup-cron)
6. [Manual Testing](#manual-testing)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The monitoring system consists of 4 automated tools that track the performance of improved selection strategies V2:

1. **compare_strategies.py** - Compares original vs improved strategy selections
2. **performance_tracker.py** - Records daily performance metrics to CSV
3. **weekly_report_generator.py** - Generates comprehensive weekly performance reports
4. **strategy_monitor_dashboard.py** - Real-time Streamlit dashboard (manual/on-demand)

---

## Monitoring Tools

### 1. Strategy Comparer (compare_strategies.py)

**Purpose**: Compare selection results between strategies
**Frequency**: On-demand or daily after selection completes
**Output**: `comparison_report.json`

```bash
python compare_strategies.py --output comparison_report.json
```

### 2. Performance Tracker (performance_tracker.py)

**Purpose**: Record daily performance metrics to time-series CSV
**Frequency**: Daily at market close (16:30 EST)
**Output**: `performance_tracking.csv`

```bash
python performance_tracker.py --output performance_tracking.csv
```

### 3. Weekly Report Generator (weekly_report_generator.py)

**Purpose**: Generate comprehensive weekly performance reports
**Frequency**: Weekly (Sunday evening 18:00)
**Output**: `reports/weekly_report_YYYY-MM-DD.{md,json}`

```bash
python weekly_report_generator.py --output reports/
```

### 4. Strategy Monitor Dashboard (strategy_monitor_dashboard.py)

**Purpose**: Real-time performance monitoring dashboard
**Frequency**: Manual/on-demand
**Access**: http://localhost:8503

```bash
streamlit run strategy_monitor_dashboard.py --server.port 8503
```

---

## Recommended Schedule

| Tool | Frequency | Time (EST) | Priority |
|------|-----------|------------|----------|
| Performance Tracker | Daily | 16:30 | High |
| Weekly Report | Weekly (Sunday) | 18:00 | Medium |
| Strategy Comparer | On-demand | After selections | Low |
| Dashboard | Manual | As needed | Low |

### Schedule Rationale

1. **Performance Tracker (16:30 EST)**: Runs after market close (16:00 EST) to capture final portfolio state
2. **Weekly Report (Sunday 18:00)**: Runs after trading week ends for full weekly analysis
3. **Strategy Comparer**: Run manually when comparing strategy outputs
4. **Dashboard**: Keep running during market hours for real-time monitoring

---

## Windows Setup (Task Scheduler)

### Prerequisites

1. Python installed and in PATH
2. Virtual environment activated (if using)
3. All dependencies installed

### Step-by-Step Setup

#### 1. Open Task Scheduler

```
Start Menu → Search "Task Scheduler" → Open
```

#### 2. Create Daily Performance Tracker Task

**Action → Create Basic Task:**

- **Name**: Strategy Performance Tracker
- **Description**: Daily recording of performance metrics
- **Trigger**: Daily at 4:30 PM
- **Action**: Start a program
- **Program/script**: `C:\quant_system_v2\.venv\Scripts\python.exe`
- **Arguments**: `performance_tracker.py --output performance_tracking.csv`
- **Start in**: `C:\quant_system_v2\quant_system_full`

**Advanced Settings:**
- Run whether user is logged on or not
- Run with highest privileges (if needed)
- Wake the computer to run this task (optional)

#### 3. Create Weekly Report Task

**Action → Create Basic Task:**

- **Name**: Weekly Strategy Report
- **Description**: Weekly performance report generation
- **Trigger**: Weekly on Sunday at 6:00 PM
- **Action**: Start a program
- **Program/script**: `C:\quant_system_v2\.venv\Scripts\python.exe`
- **Arguments**: `weekly_report_generator.py --output reports/`
- **Start in**: `C:\quant_system_v2\quant_system_full`

**Advanced Settings:**
- Run whether user is logged on or not
- Run with highest privileges (if needed)

### PowerShell Script Alternative

Create `run_monitoring.ps1`:

```powershell
# Performance Tracker - Daily at 16:30
$action = New-ScheduledTaskAction -Execute "C:\quant_system_v2\.venv\Scripts\python.exe" `
    -Argument "performance_tracker.py --output performance_tracking.csv" `
    -WorkingDirectory "C:\quant_system_v2\quant_system_full"

$trigger = New-ScheduledTaskTrigger -Daily -At 16:30

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName "StrategyPerformanceTracker" `
    -Action $action -Trigger $trigger -Principal $principal

# Weekly Report - Sunday at 18:00
$action = New-ScheduledTaskAction -Execute "C:\quant_system_v2\.venv\Scripts\python.exe" `
    -Argument "weekly_report_generator.py --output reports/" `
    -WorkingDirectory "C:\quant_system_v2\quant_system_full"

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 18:00

Register-ScheduledTask -TaskName "WeeklyStrategyReport" `
    -Action $action -Trigger $trigger -Principal $principal

Write-Host "Scheduled tasks created successfully!"
```

Run with: `powershell -ExecutionPolicy Bypass -File run_monitoring.ps1`

---

## Linux/Mac Setup (Cron)

### Prerequisites

1. Python installed
2. Virtual environment (if using)
3. All dependencies installed

### Cron Configuration

Edit crontab:

```bash
crontab -e
```

Add the following entries:

```bash
# Daily Performance Tracker - 4:30 PM EST (adjust for your timezone)
30 16 * * * cd /path/to/quant_system_v2/quant_system_full && /path/to/.venv/bin/python performance_tracker.py --output performance_tracking.csv >> logs/performance_tracker.log 2>&1

# Weekly Report - Sunday 6:00 PM EST
0 18 * * 0 cd /path/to/quant_system_v2/quant_system_full && /path/to/.venv/bin/python weekly_report_generator.py --output reports/ >> logs/weekly_report.log 2>&1
```

### Timezone Considerations

If your server is not in EST timezone, adjust the hours accordingly:

- **EST to UTC**: Add 5 hours (EST is UTC-5)
- **EST to PST**: Subtract 3 hours (PST is UTC-8)

Example for UTC timezone:

```bash
# 4:30 PM EST = 9:30 PM UTC
30 21 * * * cd /path/to/quant_system_v2/quant_system_full && python performance_tracker.py >> logs/performance_tracker.log 2>&1

# 6:00 PM EST Sunday = 11:00 PM UTC Sunday
0 23 * * 0 cd /path/to/quant_system_v2/quant_system_full && python weekly_report_generator.py >> logs/weekly_report.log 2>&1
```

### Verify Cron Jobs

```bash
# List all cron jobs
crontab -l

# Check cron logs
grep CRON /var/log/syslog  # Debian/Ubuntu
grep CRON /var/log/cron     # RedHat/CentOS
```

### Shell Script Alternative

Create `run_monitoring.sh`:

```bash
#!/bin/bash

# Configuration
PROJECT_DIR="/path/to/quant_system_v2/quant_system_full"
PYTHON_BIN="/path/to/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Function to run with logging
run_with_log() {
    local script=$1
    local logfile=$2
    cd "$PROJECT_DIR"
    echo "[$(date)] Starting $script" >> "$logfile"
    $PYTHON_BIN "$script" >> "$logfile" 2>&1
    echo "[$(date)] Completed $script with exit code $?" >> "$logfile"
}

# Daily Performance Tracker
if [ "$(date +%H:%M)" == "16:30" ]; then
    run_with_log "performance_tracker.py" "$LOG_DIR/performance_tracker.log"
fi

# Weekly Report (Sunday)
if [ "$(date +%u)" == "7" ] && [ "$(date +%H:%M)" == "18:00" ]; then
    run_with_log "weekly_report_generator.py" "$LOG_DIR/weekly_report.log"
fi
```

Make executable and add to cron:

```bash
chmod +x run_monitoring.sh

# Add to cron (every minute, script checks time internally)
* * * * * /path/to/run_monitoring.sh
```

---

## Manual Testing

### Test Individual Tools

Before setting up automation, test each tool manually:

```bash
# 1. Test Performance Tracker
python performance_tracker.py --output test_performance.csv
# Check: test_performance.csv created with data

# 2. Test Strategy Comparer
python compare_strategies.py --output test_comparison.json
# Check: test_comparison.json created with analysis

# 3. Test Weekly Report Generator
python weekly_report_generator.py --output test_reports/ --days 7
# Check: test_reports/weekly_report_YYYY-MM-DD.{md,json} created

# 4. Test Dashboard
streamlit run strategy_monitor_dashboard.py --server.port 8503
# Check: Dashboard loads at http://localhost:8503
```

### Clean Up Test Files

```bash
# Remove test files
rm test_performance.csv
rm test_comparison.json
rm -rf test_reports/
```

### Verify Dependencies

```bash
# Check all required packages installed
python -c "import streamlit, pandas, csv, json; print('All dependencies OK')"
```

---

## Troubleshooting

### Common Issues

#### 1. "Module not found" Error

**Problem**: Python can't find required modules

**Solution**:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# Install missing dependencies
pip install streamlit pandas
```

#### 2. "File not found" Error

**Problem**: Script can't find status.json or CSV files

**Solution**:
- Verify working directory is correct
- Check file paths in scripts
- Ensure trading system is running and generating status.json

#### 3. Task Doesn't Run on Schedule

**Windows**:
```
- Check Task Scheduler History
- Verify task is enabled
- Check user permissions
- Review task conditions (e.g., "Start only if computer is idle")
```

**Linux/Mac**:
```bash
# Check cron service status
sudo systemctl status cron  # systemd
sudo service cron status    # init.d

# Check cron logs
grep CRON /var/log/syslog

# Verify cron syntax
crontab -l
```

#### 4. Permission Denied

**Solution**:
```bash
# Linux/Mac: Make scripts executable
chmod +x *.py

# Windows: Run Task Scheduler as Administrator
# Or adjust script permissions
```

#### 5. Dashboard Won't Start

**Problem**: Streamlit dashboard fails to start

**Solution**:
```bash
# Check if port 8503 is already in use
netstat -an | grep 8503  # Linux/Mac
netstat -an | findstr 8503  # Windows

# Try different port
streamlit run strategy_monitor_dashboard.py --server.port 8504
```

### Log Files

Check log files for detailed error information:

```
logs/performance_tracker.log
logs/weekly_report.log
backtest_comparison.log
```

### Support

If issues persist:

1. Check Python version: `python --version` (3.11+ recommended)
2. Verify all files exist and are readable
3. Review error messages in logs
4. Test scripts manually before automation
5. Ensure trading system is running and accessible

---

## Verification Checklist

After setup, verify everything works:

- [ ] Performance Tracker runs daily and updates CSV
- [ ] Weekly Report generates every Sunday
- [ ] Dashboard accessible and displays current data
- [ ] All output files created in correct locations
- [ ] No errors in log files
- [ ] Automated tasks run even when not logged in (optional)
- [ ] Email notifications working (if configured)

---

## Advanced Configuration

### Email Notifications (Optional)

Add email notifications when reports are generated:

```python
# Add to weekly_report_generator.py
import smtplib
from email.mime.text import MIMEText

def send_email_notification(report_path):
    msg = MIMEText(f"Weekly report generated: {report_path}")
    msg['Subject'] = 'Weekly Strategy Report Generated'
    msg['From'] = 'your_email@example.com'
    msg['To'] = 'recipient@example.com'

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your_email@example.com', 'your_password')
        server.send_message(msg)
```

### Integration with Existing System

To integrate with the main trading system:

1. Add monitoring calls to `bot/runner.py` or main loop
2. Use task intervals from `.env` configuration
3. Coordinate with existing selection and trading schedules

---

## Next Steps

After automation is configured:

1. **Monitor for 1-2 weeks** to ensure tasks run reliably
2. **Review weekly reports** for performance insights
3. **Adjust schedules** if needed based on your workflow
4. **Set up alerts** for critical events (optional)
5. **Compare strategies** after 4-8 weeks of data collection

**Observation Period**: Allow 4-8 weeks for meaningful strategy comparison data.
