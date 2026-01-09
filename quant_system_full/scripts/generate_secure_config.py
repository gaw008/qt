#!/usr/bin/env python3
"""
Critical Security Configuration Generator
Generates cryptographically secure configuration for live trading deployment
"""

import os
import sys
import secrets
import string
from cryptography.fernet import Fernet
from pathlib import Path

def generate_strong_token(length=32):
    """Generate cryptographically strong token"""
    return secrets.token_urlsafe(length)

def generate_encryption_key():
    """Generate encryption key for sensitive data"""
    return Fernet.generate_key()

def encrypt_sensitive_data(data, key):
    """Encrypt sensitive configuration data"""
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def create_secure_env():
    """Create secure .env configuration"""

    # Generate secure tokens
    admin_token = generate_strong_token(32)
    encryption_key = generate_encryption_key()

    # Read current .env for reference
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            current_env = f.read()
    else:
        current_env = ""

    # Create secure configuration
    secure_config = f"""# === SECURITY NOTICE ===
# This configuration has been hardened for live trading deployment
# Generated on: {os.popen('date').read().strip()}
#
# CRITICAL: Keep this file secure and never commit to version control
# CRITICAL: Use HTTPS only in production
# CRITICAL: Rotate admin token regularly

# === Unicode Encoding Configuration ===
PYTHONIOENCODING=utf-8

# === AI API - ENCRYPTED ===
# NOTICE: API keys should be stored in secure key management system
GEMINI_API_KEY_ENCRYPTED=encrypted_in_production

# === Secure Authentication ===
# CRITICAL: Strong admin token - rotate every 30 days
ADMIN_TOKEN={admin_token}

# === Encryption Configuration ===
# CRITICAL: Store this key separately from application code
ENCRYPTION_KEY={encryption_key.decode()}

# === AI Training Configuration ===
ENABLE_DAILY_AI_TRAINING=true
AI_TRAINING_FREQUENCY=daily
AI_MODEL_TYPE=lightgbm
AI_TARGET_METRIC=sharpe_ratio
AI_DATA_SOURCE=yahoo_api
AI_SELECTION_WEIGHT=0.4
AI_TRADING_WEIGHT=0.6
AI_MIN_TRAINING_DAYS=30
AI_AUTO_RETRAIN_THRESHOLD=0.3

# === Server Configuration - HTTPS ONLY ===
# CRITICAL: Use HTTPS in production
API_PORT=8443
API_HOST=0.0.0.0
FRONTEND_PORT=3443
WEBSOCKET_PORT=8444
STREAMLIT_PORT=8502

# TLS/SSL Configuration
USE_TLS=true
TLS_CERT_PATH=/etc/ssl/certs/quant_trading.crt
TLS_KEY_PATH=/etc/ssl/private/quant_trading.key

# CORS configuration - restrict to known origins
CORS_ORIGINS=https://localhost:3443,https://trading.yourdomain.com

# === Bot / TradeUP SDK ===
TIGER_ID=20550012
ACCOUNT=41169270
# CRITICAL: Private key path must be secure
PRIVATE_KEY_PATH=/secure/keys/private_key.pem
SECRET_KEY=
TIMEZONE=US/Eastern
LANG=en_US
# CRITICAL: Set to false only after all security measures implemented
DRY_RUN=true

# === Data Source Configuration ===
DATA_SOURCE=auto
YAHOO_API_TIMEOUT=15.0
YAHOO_API_RETRIES=3
YAHOO_API_RETRY_DELAY=2.0
USE_MCP_TOOLS=true

# === Market Configuration ===
PRIMARY_MARKET=US
SECONDARY_MARKETS=
MARKET_DATA_TIMEZONE=US/Eastern

# === Selection Strategy Configuration ===
DEFAULT_SELECTION_STRATEGY=value_momentum
STREAMING_MODE=true
STOCK_UNIVERSE_FILE=all_stock_symbols.csv
SELECTION_UNIVERSE_SIZE=500
SELECTION_RESULT_SIZE=20
SELECTION_MIN_SCORE=0.0

# === ML + Traditional Hybrid Configuration ===
USE_HYBRID_SIGNALS=true
ML_SIGNAL_WEIGHT=0.7
ML_CONFIDENCE_THRESHOLD=0.6
USE_GPU_ACCELERATION=true
GPU_MODELS_DIR=gpu_models

# === Task Scheduling Configuration ===
SELECTION_TASK_INTERVAL=3600
TRADING_TASK_INTERVAL=15
MONITORING_TASK_INTERVAL=60
RECOVERY_TASK_INTERVAL=300

# === Stock Universe Filtering ===
EXCLUDE_SECTORS=
INCLUDE_SECTORS=
MIN_MARKET_CAP=100000000
MAX_MARKET_CAP=5000000000000
MIN_DAILY_VOLUME=50000
MIN_STOCK_PRICE=1.0
MAX_STOCK_PRICE=2000.0

# === Performance Configuration ===
BATCH_SIZE=20
MAX_CONCURRENT_REQUESTS=5
REQUEST_DELAY=2.0
MAX_RETRIES=5

# === Fallback Strategy Configuration ===
USE_FALLBACK_STOCKS=true
FALLBACK_STRATEGY=selection_error
MAX_FALLBACK_STOCKS=15
MIN_FALLBACK_SCORE=75.0
LOG_FALLBACK_USAGE=true
TIMEOUT_PER_BATCH=300

# === API Configuration ===
API_BASE=https://localhost:8443

# === Security Headers ===
ENABLE_SECURITY_HEADERS=true
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60

# === Logging Configuration ===
LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=true
LOG_SENSITIVE_DATA=false

# === Backup Configuration ===
ENABLE_ENCRYPTED_BACKUPS=true
BACKUP_RETENTION_DAYS=30
BACKUP_ENCRYPTION_KEY_ID=backup_key_001

# === Emergency Controls ===
EMERGENCY_STOP_TOKEN={generate_strong_token(16)}
KILL_SWITCH_ENABLED=true
"""

    # Write secure configuration
    secure_env_path = env_path.parent / '.env.secure'
    with open(secure_env_path, 'w') as f:
        f.write(secure_config)

    print(f"Secure configuration generated: {secure_env_path}")
    print(f"Admin token: {admin_token}")
    print(f"Emergency stop token: {generate_strong_token(16)}")

    return admin_token, encryption_key.decode()

def secure_private_key():
    """Secure private key file permissions"""

    private_key_path = Path(__file__).parent.parent / 'private_key.pem'

    if not private_key_path.exists():
        print(f"WARNING: Private key not found at {private_key_path}")
        return False

    # Windows-specific permission commands
    if os.name == 'nt':
        commands = [
            f'icacls "{private_key_path}" /remove "NT AUTHORITY\\Authenticated Users"',
            f'icacls "{private_key_path}" /remove "BUILTIN\\Users"',
            f'icacls "{private_key_path}" /grant "{os.getenv("USERNAME")}:F"',
            f'icacls "{private_key_path}" /inheritance:r'
        ]

        for cmd in commands:
            print(f"Executing: {cmd}")
            result = os.system(cmd)
            if result != 0:
                print(f"WARNING: Command failed with code {result}")

    # Unix-like systems
    else:
        os.chmod(private_key_path, 0o600)
        print(f"Set permissions 600 on {private_key_path}")

    return True

def create_tls_config():
    """Create TLS configuration script"""

    tls_script = '''#!/bin/bash
# TLS Certificate Generation for Quantitative Trading System
# Run this script to generate self-signed certificates for development
# For production, use certificates from a trusted CA

CERT_DIR="/etc/ssl/certs"
KEY_DIR="/etc/ssl/private"
DOMAIN="localhost"

# Create directories if they don't exist
sudo mkdir -p $CERT_DIR
sudo mkdir -p $KEY_DIR

# Generate private key
sudo openssl genrsa -out $KEY_DIR/quant_trading.key 2048

# Generate certificate signing request
sudo openssl req -new -key $KEY_DIR/quant_trading.key -out quant_trading.csr -subj "/C=US/ST=NY/L=NYC/O=QuantTrading/CN=$DOMAIN"

# Generate self-signed certificate
sudo openssl x509 -req -days 365 -in quant_trading.csr -signkey $KEY_DIR/quant_trading.key -out $CERT_DIR/quant_trading.crt

# Set secure permissions
sudo chmod 600 $KEY_DIR/quant_trading.key
sudo chmod 644 $CERT_DIR/quant_trading.crt

# Clean up
rm quant_trading.csr

echo "TLS certificates generated successfully"
echo "Certificate: $CERT_DIR/quant_trading.crt"
echo "Private key: $KEY_DIR/quant_trading.key"
'''

    tls_script_path = Path(__file__).parent / 'generate_tls_certs.sh'
    with open(tls_script_path, 'w') as f:
        f.write(tls_script)

    os.chmod(tls_script_path, 0o755)
    print(f"TLS certificate generation script created: {tls_script_path}")

def main():
    """Main security configuration function"""

    print("=== CRITICAL SECURITY CONFIGURATION ===")
    print("Implementing security fixes for live trading deployment...")

    try:
        # Step 1: Generate secure configuration
        admin_token, encryption_key = create_secure_env()

        # Step 2: Secure private key
        secure_private_key()

        # Step 3: Create TLS configuration
        create_tls_config()

        print("\n=== SECURITY IMPLEMENTATION COMPLETE ===")
        print("CRITICAL NEXT STEPS:")
        print("1. Replace current .env with .env.secure")
        print("2. Run generate_tls_certs.sh to create SSL certificates")
        print("3. Update firewall rules to allow only HTTPS traffic")
        print("4. Store encryption keys in secure key management system")
        print("5. Test all security controls before enabling live trading")
        print("\nWARNING: Do not enable DRY_RUN=false until all security measures are verified")

    except Exception as e:
        print(f"ERROR: Security configuration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()