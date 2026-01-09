#!/bin/bash
# AWS EC2 Setup Script for Quant Trading System
# Run this script on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "=========================================="
echo "Quant Trading System - AWS Setup"
echo "=========================================="

# Update system
echo "[1/8] Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
echo "[2/8] Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Node.js 20
echo "[3/8] Installing Node.js 20..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install additional tools
echo "[4/8] Installing additional tools..."
sudo apt install -y git htop

# Install cloudflared
echo "[5/8] Installing cloudflared..."
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb
rm /tmp/cloudflared.deb

# Install PM2
echo "[6/8] Installing PM2..."
sudo npm install -g pm2 serve

# Setup project directory
echo "[7/8] Setting up project directory..."
cd ~
if [ ! -d "quant_system_full" ]; then
    echo "Please upload your project to ~/quant_system_full first!"
    exit 1
fi

cd quant_system_full

# Create Python virtual environment
echo "[8/8] Setting up Python environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r bot/requirements.txt
pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git

# Install Node dependencies and build
cd UI
npm install
npm run build
cd ..

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Upload private_key.pem"
echo "3. Run: ./scripts/start_all.sh"
echo "4. Configure Cloudflare Tunnel"
echo ""
