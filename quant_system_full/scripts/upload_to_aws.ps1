# PowerShell script to upload project to AWS EC2
# Usage: .\upload_to_aws.ps1 -KeyPath "path\to\key.pem" -EC2IP "your-ec2-ip"

param(
    [Parameter(Mandatory=$true)]
    [string]$KeyPath,

    [Parameter(Mandatory=$true)]
    [string]$EC2IP,

    [string]$User = "ubuntu"
)

$ProjectPath = "C:\quant_system_v2\quant_system_full"
$RemotePath = "/home/$User/"

Write-Host "=========================================="
Write-Host "Uploading Quant Trading System to AWS"
Write-Host "=========================================="
Write-Host "EC2 IP: $EC2IP"
Write-Host "Key: $KeyPath"
Write-Host ""

# Files/folders to exclude
$ExcludeList = @(
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
    "*.pyc",
    "*.log",
    "logs/*",
    "data_cache/*.db",
    ".env.local"
)

# Create exclude file
$ExcludeFile = "$env:TEMP\rsync_exclude.txt"
$ExcludeList | Out-File -FilePath $ExcludeFile -Encoding UTF8

Write-Host "Step 1: Uploading main project files..."

# Use SCP to upload (simpler than rsync on Windows)
# First, create a zip file excluding unnecessary files
$ZipPath = "$env:TEMP\quant_system.zip"

Write-Host "Creating archive..."
if (Test-Path $ZipPath) { Remove-Item $ZipPath }

# Use PowerShell to create zip excluding certain folders
$SourcePath = $ProjectPath
$ExcludeDirs = @(".venv", "node_modules", "__pycache__", ".git", "logs")

$FilesToZip = Get-ChildItem -Path $SourcePath -Recurse | Where-Object {
    $exclude = $false
    foreach ($dir in $ExcludeDirs) {
        if ($_.FullName -like "*\$dir\*" -or $_.FullName -like "*\$dir") {
            $exclude = $true
            break
        }
    }
    -not $exclude
}

Write-Host "Compressing files... (this may take a few minutes)"
Compress-Archive -Path "$ProjectPath\*" -DestinationPath $ZipPath -Force

Write-Host "Step 2: Uploading to EC2..."
scp -i $KeyPath $ZipPath "${User}@${EC2IP}:/home/$User/quant_system.zip"

Write-Host "Step 3: Extracting on EC2..."
ssh -i $KeyPath "${User}@${EC2IP}" "cd /home/$User && rm -rf quant_system_full && unzip -o quant_system.zip -d quant_system_full && rm quant_system.zip"

Write-Host "Step 4: Uploading private key..."
$PrivateKeyPath = "$ProjectPath\private_key.pem"
if (Test-Path $PrivateKeyPath) {
    scp -i $KeyPath $PrivateKeyPath "${User}@${EC2IP}:/home/$User/quant_system_full/"
    Write-Host "Private key uploaded."
} else {
    Write-Host "WARNING: private_key.pem not found. Upload manually."
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Upload Complete!"
Write-Host "=========================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. SSH to EC2: ssh -i `"$KeyPath`" $User@$EC2IP"
Write-Host "2. Run setup: cd quant_system_full && chmod +x scripts/*.sh && ./scripts/aws_setup.sh"
Write-Host "3. Edit .env: nano .env"
Write-Host "4. Start services: ./scripts/start_all_aws.sh"
Write-Host ""

# Cleanup
Remove-Item $ZipPath -ErrorAction SilentlyContinue
Remove-Item $ExcludeFile -ErrorAction SilentlyContinue
