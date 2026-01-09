# ============================================
# Quant System Archive Sync Script
# Syncs Parquet archives from Vultr to Windows local storage
# Schedule via Windows Task Scheduler (daily at 4:00 AM)
# ============================================

$ErrorActionPreference = "Stop"

# Configuration
$remoteHost = "root@209.222.10.82"
$remotePath = "/root/quant_system_full/archives/"
$localPath = "C:\quant_system_v2\archives\"
$logFile = "C:\quant_system_v2\archive_sync.log"
$sshKeyPath = "C:\Users\wgy20\.ssh\id_rsa"  # Update with your SSH key path

# Logging function
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "$timestamp - $Message"
    Add-Content -Path $logFile -Value $logEntry
    Write-Host $logEntry
}

Write-Log "Starting archive sync..."

try {
    # Ensure local directory exists
    if (-not (Test-Path $localPath)) {
        New-Item -ItemType Directory -Force -Path $localPath | Out-Null
        Write-Log "Created local archive directory: $localPath"
    }

    # Check if SSH key exists
    if (-not (Test-Path $sshKeyPath)) {
        Write-Log "WARNING: SSH key not found at $sshKeyPath"
        Write-Log "Attempting sync without explicit key (using ssh-agent or default key)"
        $sshKeyOption = ""
    } else {
        $sshKeyOption = "-i `"$sshKeyPath`""
    }

    # Sync using scp (requires OpenSSH installed on Windows)
    # -r: recursive
    # -p: preserve timestamps
    Write-Log "Syncing from ${remoteHost}:${remotePath} to ${localPath}"

    # Check if scp is available
    $scpPath = Get-Command scp -ErrorAction SilentlyContinue
    if (-not $scpPath) {
        throw "SCP not found. Please install OpenSSH Client (Settings > Apps > Optional Features > OpenSSH Client)"
    }

    # Execute scp
    if ($sshKeyOption) {
        $scpCommand = "scp $sshKeyOption -r -p `"${remoteHost}:${remotePath}*`" `"$localPath`""
    } else {
        $scpCommand = "scp -r -p `"${remoteHost}:${remotePath}*`" `"$localPath`""
    }

    Write-Log "Executing: $scpCommand"
    Invoke-Expression $scpCommand

    # Count synced files
    $fileCount = (Get-ChildItem -Path $localPath -Filter "*.parquet" -Recurse | Measure-Object).Count
    $jsonCount = (Get-ChildItem -Path $localPath -Filter "*.json" -Recurse | Measure-Object).Count

    Write-Log "Sync completed successfully"
    Write-Log "Archive files: $fileCount Parquet, $jsonCount JSON"

    # Calculate total size
    $totalSize = (Get-ChildItem -Path $localPath -Recurse | Measure-Object -Property Length -Sum).Sum
    $totalSizeMB = [math]::Round($totalSize / 1MB, 2)
    Write-Log "Total archive size: $totalSizeMB MB"

} catch {
    $errorMessage = $_.Exception.Message
    Write-Log "ERROR: Sync failed - $errorMessage"
    exit 1
}

Write-Log "Archive sync completed"
exit 0
