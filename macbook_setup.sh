#!/bin/bash
# MacBook Setup Script for Quant Trading System
# Run this script on your MacBook after transferring quant_secrets.7z

set -e

echo "========================================"
echo "  Quant System - MacBook Setup Script"
echo "========================================"
echo ""

# Configuration
PROJECT_DIR="$HOME/Projects/quant_system_v2"
SECRETS_FILE="$HOME/Downloads/quant_secrets.7z"

# Step 1: Check prerequisites
echo "[1/7] Checking prerequisites..."
command -v git >/dev/null 2>&1 || { echo "ERROR: git not installed. Run: xcode-select --install"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not installed."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "ERROR: npm not installed. Install Node.js first."; exit 1; }
echo "  Git, Python3, npm - OK"

# Step 2: Install p7zip if needed
echo "[2/7] Checking 7zip..."
if ! command -v 7z &> /dev/null; then
    echo "  Installing p7zip via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "  ERROR: Homebrew not installed. Please install first:"
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    brew install p7zip
fi
echo "  7zip - OK"

# Step 3: Clone repository
echo "[3/7] Setting up project directory..."
mkdir -p "$HOME/Projects"
if [ -d "$PROJECT_DIR" ]; then
    echo "  Project directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "  Cloning repository..."
    git clone https://github.com/gaw008/qt.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
echo "  Repository ready at $PROJECT_DIR"

# Step 4: Extract secrets
echo "[4/7] Extracting sensitive files..."
if [ ! -f "$SECRETS_FILE" ]; then
    echo "  ERROR: $SECRETS_FILE not found!"
    echo "  Please copy quant_secrets.7z to ~/Downloads/ first"
    exit 1
fi

cd "$HOME/Downloads"
7z x -y quant_secrets.7z -oquant_secrets_extracted

# Move files to correct locations
cd "$PROJECT_DIR/quant_system_full"
cp "$HOME/Downloads/quant_secrets_extracted/.env" . 2>/dev/null || true
cp "$HOME/Downloads/quant_secrets_extracted/private_key.pem" . 2>/dev/null || true
cp "$HOME/Downloads/quant_secrets_extracted/backend.env" dashboard/backend/.env 2>/dev/null || true
mkdir -p props
cp -r "$HOME/Downloads/quant_secrets_extracted/props/"* props/ 2>/dev/null || true

# Set correct permissions
chmod 600 private_key.pem 2>/dev/null || true

# Clean up extracted files
rm -rf "$HOME/Downloads/quant_secrets_extracted"
rm "$SECRETS_FILE"

echo "  Secrets installed successfully"

# Step 5: Fix paths in .env
echo "[5/7] Adjusting paths for MacBook..."
if [ -f ".env" ]; then
    # Replace Windows path with Mac path
    sed -i '' "s|C:/quant_system_v2/quant_system_full|$PROJECT_DIR/quant_system_full|g" .env
    sed -i '' "s|C:\\\\quant_system_v2\\\\quant_system_full|$PROJECT_DIR/quant_system_full|g" .env
    echo "  Paths updated in .env"
fi

# Step 6: Setup Python environment
echo "[6/7] Setting up Python environment..."
cd "$PROJECT_DIR/quant_system_full"
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r bot/requirements.txt -q
pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git -q

echo "  Python environment ready"

# Step 7: Setup Node.js environment
echo "[7/7] Setting up Node.js environment..."
cd "$PROJECT_DIR/quant_system_full/UI"
npm install -q
npm run build

echo ""
echo "========================================"
echo "  SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Project location: $PROJECT_DIR"
echo ""
echo "To activate Python environment:"
echo "  cd $PROJECT_DIR/quant_system_full"
echo "  source .venv/bin/activate"
echo ""
echo "To start the system:"
echo "  python start_all.py"
echo ""
echo "To access VULTR server:"
echo "  ssh root@209.222.10.82"
echo "  Password: mA!4iv4v8QWgiy(V"
echo ""
echo "Cloud service URLs:"
echo "  VULTR: https://my.vultr.com/"
echo "  Supabase: https://supabase.com/dashboard/project/txceqncllasfjbarufzi"
echo "  Trading UI: https://trade.wgyjdaiassistant.cc"
echo ""
