# MacBook Sync - Sensitive Files Packaging Script
# Run this script as Administrator for 7-Zip installation

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Quant System - Secrets Packaging Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check/Install 7-Zip
$7zipPath = "C:\Program Files\7-Zip\7z.exe"
if (!(Test-Path $7zipPath)) {
    Write-Host "[1/4] 7-Zip not found. Installing..." -ForegroundColor Yellow
    $installerPath = "$env:TEMP\7z-installer.exe"

    # Download 7-Zip
    Write-Host "  Downloading 7-Zip..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://www.7-zip.org/a/7z2408-x64.exe" -OutFile $installerPath

    # Install 7-Zip (requires admin for silent install)
    Write-Host "  Installing 7-Zip..." -ForegroundColor Gray
    Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -Verb RunAs

    # Clean up installer
    Remove-Item $installerPath -Force

    if (!(Test-Path $7zipPath)) {
        Write-Host "  ERROR: 7-Zip installation failed. Please install manually." -ForegroundColor Red
        Write-Host "  Download from: https://www.7-zip.org/download.html" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  7-Zip installed successfully!" -ForegroundColor Green
} else {
    Write-Host "[1/4] 7-Zip found." -ForegroundColor Green
}

# Step 2: Create temp directory and copy files
Write-Host "[2/4] Collecting sensitive files..." -ForegroundColor Yellow
$tempDir = "$env:TEMP\quant_secrets_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$filesToCopy = @(
    @{Source="C:\quant_system_v2\quant_system_full\.env"; Dest=".env"},
    @{Source="C:\quant_system_v2\quant_system_full\private_key.pem"; Dest="private_key.pem"},
    @{Source="C:\quant_system_v2\quant_system_full\dashboard\backend\.env"; Dest="backend.env"}
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file.Source) {
        Copy-Item $file.Source "$tempDir\$($file.Dest)"
        Write-Host "  Copied: $($file.Dest)" -ForegroundColor Gray
    } else {
        Write-Host "  WARNING: Not found: $($file.Source)" -ForegroundColor Yellow
    }
}

# Copy props directory
$propsSource = "C:\quant_system_v2\quant_system_full\props"
if (Test-Path $propsSource) {
    Copy-Item $propsSource "$tempDir\props" -Recurse
    Write-Host "  Copied: props\" -ForegroundColor Gray
}

# Step 3: Get password from user
Write-Host ""
Write-Host "[3/4] Setting encryption password..." -ForegroundColor Yellow
$password = Read-Host "Enter encryption password (remember this for MacBook)"

if ([string]::IsNullOrWhiteSpace($password)) {
    Write-Host "  ERROR: Password cannot be empty!" -ForegroundColor Red
    Remove-Item -Recurse -Force $tempDir
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 4: Create encrypted archive
Write-Host "[4/4] Creating encrypted archive..." -ForegroundColor Yellow
$outputPath = "$env:USERPROFILE\Desktop\quant_secrets.7z"

# Remove existing archive if present
if (Test-Path $outputPath) {
    Remove-Item $outputPath -Force
}

# Create encrypted archive with 7-Zip
# -p: password, -mhe=on: encrypt header (hide filenames)
$7zipArgs = @("a", "-p$password", "-mhe=on", $outputPath, "$tempDir\*")
$process = Start-Process -FilePath $7zipPath -ArgumentList $7zipArgs -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  SUCCESS!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Archive created: $outputPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Transfer quant_secrets.7z to your MacBook" -ForegroundColor White
    Write-Host "  2. On MacBook, install p7zip: brew install p7zip" -ForegroundColor White
    Write-Host "  3. Extract: 7z x quant_secrets.7z" -ForegroundColor White
    Write-Host "  4. Enter the password you just set" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "  ERROR: Failed to create archive (exit code: $($process.ExitCode))" -ForegroundColor Red
}

# Clean up temp directory
Remove-Item -Recurse -Force $tempDir

Write-Host ""
Read-Host "Press Enter to close"
