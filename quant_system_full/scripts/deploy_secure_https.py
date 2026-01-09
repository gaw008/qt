#!/usr/bin/env python3
"""
HTTPS Deployment Script
Configures the system for secure HTTPS-only communications
"""

import os
import sys
import subprocess
from pathlib import Path

class HTTPSDeployment:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def create_self_signed_certificate(self):
        """Create self-signed SSL certificate for development/testing"""
        print("Creating self-signed SSL certificate...")

        # Create SSL directory
        ssl_dir = self.project_root / 'ssl'
        ssl_dir.mkdir(exist_ok=True)

        cert_path = ssl_dir / 'cert.pem'
        key_path = ssl_dir / 'key.pem'

        # Generate private key and certificate
        openssl_cmd = [
            'openssl', 'req', '-new', '-newkey', 'rsa:2048', '-days', '365',
            '-nodes', '-x509', '-keyout', str(key_path), '-out', str(cert_path),
            '-subj', '/CN=localhost'
        ]

        try:
            subprocess.run(openssl_cmd, check=True, capture_output=True)
            print(f"✓ SSL certificate created: {cert_path}")
            print(f"✓ SSL private key created: {key_path}")

            # Set secure permissions on key file
            if os.name != 'nt':
                os.chmod(key_path, 0o600)

            return cert_path, key_path

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create SSL certificate: {e}")
            return None, None
        except FileNotFoundError:
            print("❌ OpenSSL not found. Please install OpenSSL to create certificates.")
            return None, None

    def update_fastapi_for_https(self):
        """Update FastAPI backend to support HTTPS"""
        app_path = self.project_root / 'dashboard' / 'backend' / 'app.py'

        https_update = '''
# Add HTTPS support
import ssl

# Add after existing imports
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add after app initialization
if os.getenv('USE_TLS', 'false').lower() == 'true':
    # Force HTTPS in production
    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    return response

# Update the main startup section
if __name__ == "__main__":
    import uvicorn

    # Use secure configuration
    host = "0.0.0.0"
    use_tls = os.getenv('USE_TLS', 'false').lower() == 'true'

    if use_tls:
        port = int(os.getenv('API_PORT', 8443))
        ssl_keyfile = os.getenv('TLS_KEY_PATH', 'ssl/key.pem')
        ssl_certfile = os.getenv('TLS_CERT_PATH', 'ssl/cert.pem')

        print(f"Starting HTTPS API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        port = int(os.getenv('API_PORT', 8000))
        print(f"Starting HTTP API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=True)
'''

        print("HTTPS configuration update prepared.")
        print("To apply, manually add the HTTPS middleware and security headers to app.py")
        print("This ensures the API server can run with SSL/TLS encryption.")

    def create_nginx_ssl_config(self):
        """Create nginx configuration with SSL"""
        nginx_ssl_config = '''
# Quantitative Trading System - SSL Configuration
server {
    listen 80;
    server_name localhost;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name localhost;

    ssl_certificate ssl/cert.pem;
    ssl_certificate_key ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
'''

        nginx_config_path = self.project_root / 'nginx_ssl.conf'
        with open(nginx_config_path, 'w') as f:
            f.write(nginx_ssl_config)

        print(f"✓ Nginx SSL configuration created: {nginx_config_path}")

    def deploy_https(self):
        """Deploy HTTPS configuration"""
        print("=== HTTPS DEPLOYMENT ===")
        print("Configuring secure HTTPS communications...")

        # Step 1: Create SSL certificates
        cert_path, key_path = self.create_self_signed_certificate()
        if not cert_path:
            print("❌ SSL certificate creation failed")
            return False

        # Step 2: Update FastAPI configuration
        self.update_fastapi_for_https()

        # Step 3: Create nginx SSL configuration
        self.create_nginx_ssl_config()

        print("\n✓ HTTPS deployment completed")
        print("Next steps:")
        print("1. Apply the FastAPI HTTPS updates to dashboard/backend/app.py")
        print("2. Start the API server with USE_TLS=true")
        print("3. Configure nginx with the SSL configuration")
        print("4. Test HTTPS connectivity")

        return True

def main():
    """Main HTTPS deployment function"""
    deployment = HTTPSDeployment()
    deployment.deploy_https()

if __name__ == "__main__":
    main()