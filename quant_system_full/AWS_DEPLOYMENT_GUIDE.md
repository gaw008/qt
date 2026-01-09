# AWS Deployment Guide - Quant Trading System

## Architecture Overview

```
Internet
    |
Cloudflare (DNS + SSL)
    |
    v
AWS EC2 (us-east-1)
    |
    +-- FastAPI Backend (port 8000)
    +-- React Frontend (port 5173)
    +-- Streamlit Dashboard (port 8501)
    |
    v
Tiger Brokers API (Low Latency)
```

---

## Recommended AWS Configuration

### EC2 Instance

| Item | Recommended | Minimum |
|------|-------------|---------|
| Region | us-east-1 (N. Virginia) | us-east-1 |
| Instance Type | t3.medium | t3.small |
| vCPU | 2 | 2 |
| Memory | 4 GB | 2 GB |
| Storage | 50 GB SSD | 30 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Estimated Monthly Cost

| Component | t3.small | t3.medium |
|-----------|----------|-----------|
| EC2 Instance | $15 | $30 |
| Storage (50GB) | $5 | $5 |
| Data Transfer | ~$5 | ~$5 |
| **Total** | **~$25/month** | **~$40/month** |

---

## Step-by-Step Deployment

### Step 1: Create AWS Account and EC2 Instance

1. Go to https://aws.amazon.com and create account
2. Navigate to EC2 Dashboard
3. Click "Launch Instance"
4. Configure:
   - Name: `quant-trading-system`
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: `t3.small` or `t3.medium`
   - Key pair: Create new key pair (download .pem file)
   - Storage: 50 GB gp3
   - Security Group: See Step 2

### Step 2: Configure Security Group

Allow inbound traffic:

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | Your IP | Remote access |
| HTTP | 80 | 0.0.0.0/0 | Cloudflare |
| HTTPS | 443 | 0.0.0.0/0 | Cloudflare |
| Custom TCP | 8000 | 0.0.0.0/0 | API Backend |
| Custom TCP | 5173 | 0.0.0.0/0 | React Frontend |
| Custom TCP | 8501 | 0.0.0.0/0 | Streamlit |

### Step 3: Connect to EC2

```bash
# Windows (PowerShell)
ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>

# Or use PuTTY with converted .ppk key
```

### Step 4: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Git
sudo apt install -y git

# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# Install PM2 for process management
sudo npm install -g pm2
```

### Step 5: Upload Code

Option A: Git Clone (if using private repo)
```bash
git clone https://github.com/your-repo/quant_system_full.git
cd quant_system_full
```

Option B: Upload via SCP (from local Windows)
```powershell
# From Windows PowerShell
scp -i "your-key.pem" -r C:\quant_system_v2\quant_system_full ubuntu@<EC2-IP>:~/
```

### Step 6: Configure Environment

```bash
cd ~/quant_system_full

# Create Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r bot/requirements.txt
pip install git+https://github.com/tigerbrokers/openapi-python-sdk.git

# Install Node dependencies
cd UI
npm install
npm run build
cd ..
```

### Step 7: Configure .env

```bash
# Copy and edit .env
nano .env
```

Update these values:
```bash
# Server Configuration
API_HOST=0.0.0.0
CORS_ORIGINS=http://localhost:3000,https://trade.wgyjdaiassistant.cc,https://api.wgyjdaiassistant.cc

# Access Password
ACCESS_PASSWORD=WGYJD0508

# Tiger API (keep existing values)
TIGER_ID=20550012
ACCOUNT=41169270
PRIVATE_KEY_PATH=/home/ubuntu/quant_system_full/private_key.pem
```

### Step 8: Upload Private Key

```powershell
# From Windows PowerShell
scp -i "your-key.pem" C:\quant_system_v2\quant_system_full\private_key.pem ubuntu@<EC2-IP>:~/quant_system_full/
```

### Step 9: Create Startup Scripts

Create `start_services.sh`:
```bash
#!/bin/bash
cd /home/ubuntu/quant_system_full

# Activate virtual environment
source .venv/bin/activate

# Start Backend API
pm2 start "uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000" --name api

# Start React Frontend (serve built files)
pm2 start "npx serve -s UI/dist -l 5173" --name frontend

# Start Streamlit Dashboard
pm2 start "streamlit run dashboard/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0" --name streamlit

# Start Trading Bot
pm2 start "python start_bot.py" --name bot

# Save PM2 configuration
pm2 save
```

Make executable and run:
```bash
chmod +x start_services.sh
./start_services.sh
```

### Step 10: Configure Cloudflare Tunnel on AWS

```bash
# Login to Cloudflare
cloudflared tunnel login

# Use existing tunnel or create new
cloudflared tunnel create quant-trading-aws

# Create config
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Config content:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/ubuntu/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: api.wgyjdaiassistant.cc
    service: http://localhost:8000
  - hostname: trade.wgyjdaiassistant.cc
    service: http://localhost:5173
  - hostname: dash.wgyjdaiassistant.cc
    service: http://localhost:8501
  - service: http_status:404
```

Configure DNS and start:
```bash
cloudflared tunnel route dns quant-trading-aws api.wgyjdaiassistant.cc
cloudflared tunnel route dns quant-trading-aws trade.wgyjdaiassistant.cc
cloudflared tunnel route dns quant-trading-aws dash.wgyjdaiassistant.cc

# Install as service
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### Step 11: Configure Auto-Start on Reboot

```bash
# PM2 startup configuration
pm2 startup
# Run the command it outputs

# Save current process list
pm2 save
```

---

## Management Commands

### Check Status
```bash
pm2 status
pm2 logs
pm2 logs api --lines 100
```

### Restart Services
```bash
pm2 restart all
pm2 restart api
pm2 restart bot
```

### Stop Services
```bash
pm2 stop all
```

### View Logs
```bash
pm2 logs --lines 200
tail -f ~/.pm2/logs/api-out.log
```

### Update Code
```bash
cd ~/quant_system_full
git pull  # If using git
pm2 restart all
```

---

## Monitoring

### System Resources
```bash
htop
free -h
df -h
```

### PM2 Monitoring
```bash
pm2 monit
```

---

## Troubleshooting

### Port Already in Use
```bash
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Permission Issues
```bash
sudo chown -R ubuntu:ubuntu ~/quant_system_full
```

### Python Package Issues
```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r bot/requirements.txt
```

---

## Security Recommendations

1. **Restrict SSH Access**: Only allow your IP in security group
2. **Enable AWS CloudWatch**: Monitor instance metrics
3. **Regular Updates**: `sudo apt update && sudo apt upgrade`
4. **Backup**: Create AMI snapshots regularly
5. **Elastic IP**: Assign static IP to avoid IP changes on restart

---

## Cost Optimization

1. **Reserved Instance**: Save 30-40% with 1-year commitment
2. **Spot Instance**: Save up to 90% (but may be interrupted)
3. **Right-sizing**: Start with t3.small, upgrade if needed
4. **Stop when not trading**: EC2 only charges when running

---

## Quick Reference

| Service | Local Port | External URL |
|---------|------------|--------------|
| API Backend | 8000 | https://api.wgyjdaiassistant.cc |
| React Frontend | 5173 | https://trade.wgyjdaiassistant.cc |
| Streamlit | 8501 | https://dash.wgyjdaiassistant.cc |

**Password**: WGYJD0508
