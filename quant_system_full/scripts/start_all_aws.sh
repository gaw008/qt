#!/bin/bash
# Start all services using PM2
# Run this script after setup is complete

set -e

PROJECT_DIR="/home/ubuntu/quant_system_full"
cd $PROJECT_DIR

echo "Starting Quant Trading System..."

# Activate virtual environment
source .venv/bin/activate

# Stop any existing processes
pm2 delete all 2>/dev/null || true

# Start Backend API
echo "Starting API Backend..."
pm2 start "cd $PROJECT_DIR && source .venv/bin/activate && uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000" --name api

# Start React Frontend (serve built files)
echo "Starting React Frontend..."
pm2 start "serve -s $PROJECT_DIR/UI/dist -l 5173" --name frontend

# Start Streamlit Dashboard
echo "Starting Streamlit Dashboard..."
pm2 start "cd $PROJECT_DIR && source .venv/bin/activate && streamlit run dashboard/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true" --name streamlit

# Start Trading Bot
echo "Starting Trading Bot..."
pm2 start "cd $PROJECT_DIR && source .venv/bin/activate && python start_bot.py" --name bot

# Save PM2 configuration
pm2 save

# Show status
echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
pm2 status

echo ""
echo "Useful commands:"
echo "  pm2 status      - Check status"
echo "  pm2 logs        - View logs"
echo "  pm2 restart all - Restart all"
echo "  pm2 stop all    - Stop all"
echo ""
