#!/usr/bin/env python3
"""
Start Backend Server Simple
Start the backend server on HTTP port 8000 for testing real Tiger data
"""

import os
import sys
from pathlib import Path

# Add backend path
sys.path.append(str(Path(__file__).parent / 'dashboard' / 'backend'))

# Override environment variables for simple HTTP testing
os.environ['API_PORT'] = '8000'
os.environ['API_HOST'] = '0.0.0.0'
os.environ['USE_TLS'] = 'false'

def start_backend_simple():
    """Start backend server with simple HTTP configuration"""
    print("Starting Backend Server (HTTP mode for testing)")
    print("=" * 50)

    try:
        import uvicorn
        from app import app

        print("Starting server on http://localhost:8000")
        print("Tiger API integration: ENABLED")
        print("Real data endpoints: /api/positions, /api/portfolio/summary, /api/orders")
        print("\nPress Ctrl+C to stop the server")

        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )

    except Exception as e:
        print(f"Failed to start backend server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_backend_simple()