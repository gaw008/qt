# Vultr Cloud Deployment Documentation

## Overview

Quant Trading System deployed to Vultr VPS with Cloudflare Tunnel for secure external access.

**Deployment Date**: 2026-01-03

---

## CRITICAL: Post-Deployment Checklist

After uploading files to Vultr, you MUST fix the following paths:

### Fix PRIVATE_KEY_PATH (REQUIRED)

The local .env files use Windows paths. On Vultr, you MUST change them to Linux paths:

```bash
# SSH to Vultr
ssh root@209.222.10.82

# Fix Main .env
sed -i 's|PRIVATE_KEY_PATH=C:/quant_system_v2/quant_system_full/private_key.pem|PRIVATE_KEY_PATH=/root/quant_system_full/private_key.pem|g' /root/quant_system_full/.env

# Fix Backend .env
sed -i 's|PRIVATE_KEY_PATH=C:/quant_system_v2/quant_system_full/private_key.pem|PRIVATE_KEY_PATH=/root/quant_system_full/private_key.pem|g' /root/quant_system_full/dashboard/backend/.env

# Restart services
pm2 restart api runner --update-env
```

**Symptoms if not fixed:**
- Portfolio value shows 0 or "-"
- Daily P&L shows 0 or "-"
- Active positions shows 0 or "-"
- API logs show: `ERROR: Private key not found at: C:/quant_system_v2/...`

---

## Server Information

| Item | Value |
|------|-------|
| Provider | Vultr |
| IP Address | 209.222.10.82 |
| Username | root |
| Password | mA!4iv4v8QWgiy(V |
| OS | Ubuntu |
| Cost | ~$24/month |

---

## Domain Configuration

| Subdomain | Service | Local Port |
|-----------|---------|------------|
| trade.wgyjdaiassistant.cc | React Frontend | 5173 |
| api.wgyjdaiassistant.cc | FastAPI Backend | 8000 |

**Access Password**: `WGYJD0508`

---

## Cloudflare Tunnel

| Item | Value |
|------|-------|
| Tunnel Name | quant-trading-vultr |
| Tunnel ID | 58f9a5c9-3b07-4f31-88d6-bbb8cbe69f47 |
| Config File | /root/.cloudflared/config.yml |
| Credentials | /root/.cloudflared/58f9a5c9-3b07-4f31-88d6-bbb8cbe69f47.json |

### Tunnel Config (/root/.cloudflared/config.yml)
```yaml
tunnel: 58f9a5c9-3b07-4f31-88d6-bbb8cbe69f47
credentials-file: /root/.cloudflared/58f9a5c9-3b07-4f31-88d6-bbb8cbe69f47.json

ingress:
  - hostname: api.wgyjdaiassistant.cc
    service: http://localhost:8000
    originRequest:
      noTLSVerify: true
  - hostname: trade.wgyjdaiassistant.cc
    service: http://localhost:5173
  - service: http_status:404
```

---

## File Locations on Vultr

| Path | Description |
|------|-------------|
| /root/quant_system_full/ | Main project directory |
| /root/quant_system_full/.env | Environment variables |
| /root/quant_system_full/private_key.pem | Tiger API private key |
| /root/quant_system_full/props/tiger_openapi_config.properties | Tiger API config |
| /root/quant_system_full/dashboard/backend/app.py | FastAPI backend |
| /root/quant_system_full/dashboard/backend/.env | Backend-specific env |
| /root/quant_system_full/dashboard/worker/runner.py | Trading bot runner |
| /root/quant_system_full/UI/dist/ | Built frontend |
| /root/quant_system_full/start_api.sh | API startup script |
| /root/quant_system_full/start_runner.sh | Runner startup script |

---

## PM2 Services

| Name | Script | Description |
|------|--------|-------------|
| api | start_api.sh | FastAPI backend on port 8000 |
| frontend | npm run preview | Vite preview server on port 5173 |
| runner | start_runner.sh | Trading bot runner |

### Common PM2 Commands
```bash
# List all services
pm2 list

# Restart a service
pm2 restart api
pm2 restart frontend
pm2 restart runner
pm2 restart all

# View logs
pm2 logs api --lines 50
pm2 logs runner --lines 50
pm2 logs --lines 50  # All services

# Restart with env update
pm2 restart api --update-env
```

---

## Key Configuration Files

### /root/quant_system_full/.env
```bash
TIGER_ID=20550012
ACCOUNT=41169270
PRIVATE_KEY_PATH=/root/quant_system_full/private_key.pem
ACCESS_PASSWORD=WGYJD0508
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://trade.wgyjdaiassistant.cc,https://api.wgyjdaiassistant.cc,*
DATA_SOURCE=auto
AI_DATA_SOURCE=yahoo_api
```

### /root/quant_system_full/dashboard/backend/.env
Important: This file must have Linux paths, not Windows paths:
```bash
PRIVATE_KEY_PATH=/root/quant_system_full/private_key.pem
```

---

## API Endpoints Added

### Runner Logs Endpoint
```
GET /api/runner/logs?lines=100
Authorization: Session <token>

Response:
{
  "success": true,
  "data": {
    "lines": ["log line 1", "log line 2", ...],
    "total": 100
  }
}
```

---

## Frontend Components Added

### ConsoleLogs Component
Location: `/root/quant_system_full/UI/src/components/ConsoleLogs.tsx`

Features:
- Displays runner.py console output in real-time
- Auto-refresh every 5 seconds
- Pause/Resume functionality
- Color-coded log levels (ERROR=red, WARNING=yellow, INFO=blue, SUCCESS=green)

Added to: `Dashboard.tsx` (bottom of page)

---

## Issues Encountered & Solutions

### 1. CORS Blocking Login
**Problem**: OPTIONS preflight requests returning 400 "Disallowed CORS origin"
**Cause**: `app.py` CORS config used hardcoded localhost origins, didn't include production domains
**Solution**: Modified CORS middleware to always include production domains:
```python
production_origins = ["https://trade.wgyjdaiassistant.cc", "https://api.wgyjdaiassistant.cc", "*"]
all_origins = list(set(cors_origins + production_origins))
```

### 2. Private Key Path Error
**Problem**: `ERROR: Private key not found at: C:/quant_system_v2/quant_system_full/private_key.pem`
**Cause**: `/root/quant_system_full/dashboard/backend/.env` had Windows path
**Solution**: Changed to Linux path: `/root/quant_system_full/private_key.pem`

### 3. Tiger Props File Missing
**Problem**: `Props file not found: /root/quant_system_full/props/tiger_openapi_config.properties`
**Solution**: Created the props directory and config file

### 4. Session Token Invalid After Restart (FIXED 2026-01-04)
**Problem**: 403 Forbidden after API restart, "Login failed" with correct password
**Cause**: In-memory session tokens cleared on restart
**Solution**: Added file-based session persistence in `app.py`:
- Sessions now saved to `/root/quant_system_full/dashboard/state/sessions.json`
- Sessions survive API restarts
- Added 403/401 auto-logout in frontend (`api.ts`) with redirect to login
- Fixed `ProtectedRoute.tsx` catch block to properly handle verification failures

### 5. Frontend Build Error - ScrollArea
**Problem**: `Could not load scroll-area` component
**Solution**: Removed ScrollArea dependency from ConsoleLogs component, used plain div with overflow-auto

### 6. Frontend Calling Wrong API URL (FIXED 2026-01-04)
**Problem**: "Unexpected token '<'" error - frontend calling `trade.wgyjdaiassistant.cc` instead of `api.wgyjdaiassistant.cc`
**Cause**: Browser cached old JS file, `.env.production` wasn't being read properly
**Solution**:
- Added build version constant to `api.ts` for cache busting
- Verified `.env.production` contains `VITE_API_BASE_URL=https://api.wgyjdaiassistant.cc`
- Rebuilt frontend with `npx vite build --force` to generate new hash
- Cleaned old JS files from dist/assets/ on Vultr
- User must hard refresh (Ctrl+Shift+R) to load new version

---

## Maintenance Commands

### SSH Connection
```bash
ssh root@209.222.10.82
# Password: mA!4iv4v8QWgiy(V
```

### Check Service Status
```bash
pm2 list
systemctl status cloudflared
```

### Rebuild Frontend
```bash
cd /root/quant_system_full/UI
npm run build
pm2 restart frontend
```

### View Logs
```bash
# API logs
pm2 logs api --lines 100

# Runner logs
pm2 logs runner --lines 100

# All logs
pm2 logs --lines 50

# Cloudflare tunnel logs
journalctl -u cloudflared -f
```

### Restart Everything
```bash
pm2 restart all --update-env
systemctl restart cloudflared
```

---

## Quick Troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| Can't access site | `pm2 list` | `pm2 restart all` |
| Login fails | API logs for CORS errors | Check CORS config in app.py |
| No data showing | Re-login (session expired) | Refresh page, login again |
| 403 errors | Session token invalid | Re-login after API restart |
| Runner not running | `pm2 logs runner` | Check for import errors |

---

## Architecture Diagram

```
Internet
    |
    v
Cloudflare (DNS + SSL)
    |
    v
Cloudflare Tunnel (cloudflared service)
    |
    +---> trade.wgyjdaiassistant.cc --> localhost:5173 (React Frontend)
    |
    +---> api.wgyjdaiassistant.cc --> localhost:8000 (FastAPI Backend)

PM2 Process Manager
    |
    +---> frontend (Vite preview :5173)
    +---> api (Uvicorn :8000)
    +---> runner (Python trading bot)
```

---

## Data Flow

1. User accesses https://trade.wgyjdaiassistant.cc
2. Cloudflare Tunnel routes to localhost:5173
3. React frontend loads, shows login page
4. User enters password (WGYJD0508)
5. POST /api/auth/login -> returns session_token
6. Frontend stores token in localStorage
7. All subsequent API calls include `Authorization: Session <token>`
8. Dashboard displays portfolio, positions, orders, alerts
9. ConsoleLogs component fetches /api/runner/logs every 5 seconds

---

## Supabase Integration (Added 2026-01-04)

### Supabase Project

| Item | Value |
|------|-------|
| Project URL | https://txceqncllasfjbarufzi.supabase.co |
| Dashboard | https://supabase.com/dashboard/project/txceqncllasfjbarufzi |
| Region | Default |

### Database Schema (16 Tables)

**Core Tables (Hot Data):**
| Table | Retention | Description |
|-------|-----------|-------------|
| runs | 30 days | Task execution tracking |
| orders | 90 days | Order history |
| fills | 90 days | Fill records |
| positions | 7 days | Position snapshots |
| metrics_snapshots | 7 days | Real-time metrics |
| selection_results | 30 days | Stock selection results |
| strategy_configs | Permanent | Strategy configurations |
| archive_log | Permanent | Archival audit trail |

**Analysis Tables (Extended Retention):**
| Table | Retention | Description |
|-------|-----------|-------------|
| trade_signals | 90 days | All trading signals for hit rate analysis |
| execution_analysis | 90 days | Slippage & market impact tracking |
| daily_performance | 365 days | Daily P&L and metrics |
| market_regimes | 365 days | Market regime detection |
| ai_training_history | Permanent | AI model training records |
| strategy_performance | 365 days | Per-strategy performance |
| compliance_events | Permanent | Compliance violations |
| factor_crowding_history | 90 days | Factor crowding data |

### New Files Added

| Path | Description |
|------|-------------|
| /root/quant_system_full/dashboard/backend/supabase_client.py | Supabase Python client |
| /root/quant_system_full/dashboard/backend/supabase_schema.sql | Database schema SQL |
| /root/quant_system_full/dashboard/backend/archival_job.py | Cold data archival job |
| /root/quant_system_full/scripts/run_archival.sh | Archival cron script |

### Environment Variables Added

```bash
# Added to /root/quant_system_full/.env and /root/quant_system_full/dashboard/backend/.env
SUPABASE_URL=https://txceqncllasfjbarufzi.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGci...  # Service role key (keep secret)
```

### New API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/history/orders | GET | Order history (days, limit, symbol) |
| /api/history/positions | GET | Position snapshots |
| /api/history/performance | GET | Daily performance history |
| /api/history/runs | GET | Task run history |
| /api/history/signals | GET | Trade signal history |
| /api/history/compliance | GET | Compliance events |
| /api/history/metrics | GET | Metrics snapshots |
| /api/supabase/status | GET | Supabase connection status |

### Archival Cron Job

```bash
# Cron entry (runs daily at 8:00 UTC / 3:00 AM EST)
0 8 * * * /root/quant_system_full/scripts/run_archival.sh

# View cron
crontab -l

# Check archival logs
tail -100 /var/log/quant_archival.log
```

### Data Flow with Supabase

```
Runner Tasks (trading, selection, monitoring, etc.)
    |
    v
supabase_client.py --> Supabase Cloud (Hot Data)
                            |
                            | Daily Archival (run_archival.sh)
                            v
                       /root/quant_system_full/archives/ (Parquet files)
                            |
                            | Windows Sync (sync_archives.ps1)
                            v
                       C:\quant_system_v2\archives\ (Cold Data - Permanent)
```

### Runner Auto-Write Matrix

| Task | Tables Written |
|------|----------------|
| real_trading_task | runs, orders, positions, trade_signals |
| stock_selection_task | runs, selection_results, trade_signals |
| real_time_monitoring_task | runs, metrics_snapshots |
| factor_crowding_monitoring_task | runs, factor_crowding_history |
| compliance_monitoring_task | runs, compliance_events |

### Supabase Troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| No data in Supabase | `pm2 logs runner` for errors | Check env vars loaded |
| Connection refused | Supabase dashboard status | Check SUPABASE_URL |
| Table not found | SQL Editor | Run supabase_schema.sql |
| Archival not running | `crontab -l` | Add cron entry |

### Manual Supabase Commands

```bash
# Test Supabase connection
cd /root/quant_system_full/dashboard/backend
source /root/quant_system_full/.venv/bin/activate
export $(grep SUPABASE /root/quant_system_full/.env | xargs)
python3 -c "from supabase_client import supabase_client; print(supabase_client.get_recent_runs(1))"

# Run archival manually
python -m dashboard.backend.archival_job

# Dry run archival (see what would be archived)
python -m dashboard.backend.archival_job --dry-run
```

---

## Deployment Workflow (Update & Deploy New Content)

This section documents the complete workflow for updating code locally and deploying to Vultr.

### Quick Reference Commands

```bash
# 1. SSH to Vultr
ssh root@209.222.10.82

# 2. Navigate to project
cd /root/quant_system_full

# 3. Pull changes or sync files
# Option A: Git pull (if using git)
git pull origin main

# Option B: SCP from local Windows (run from Windows PowerShell)
scp -r C:\quant_system_v2\quant_system_full\* root@209.222.10.82:/root/quant_system_full/

# 4. Restart services
pm2 restart all --update-env
```

### Detailed Deployment Steps

#### Step 1: Make Changes Locally (Windows)

Edit files in `C:\quant_system_v2\quant_system_full\` as needed.

#### Step 2: Transfer Files to Vultr

**Option A: SCP Specific Files (Recommended)**
```powershell
# From Windows PowerShell - Transfer specific changed files
scp C:\quant_system_v2\quant_system_full\dashboard\backend\app.py root@209.222.10.82:/root/quant_system_full/dashboard/backend/
scp C:\quant_system_v2\quant_system_full\UI\src\lib\api.ts root@209.222.10.82:/root/quant_system_full/UI/src/lib/
```

**Option B: SCP Entire Directory**
```powershell
# Transfer entire project (excludes node_modules, .venv, etc. via .scpignore)
scp -r C:\quant_system_v2\quant_system_full\* root@209.222.10.82:/root/quant_system_full/
```

**Option C: Git (if configured)**
```bash
# On Vultr server
cd /root/quant_system_full
git pull origin main
```

#### Step 3: Backend Changes (Python/FastAPI)

If you modified Python files (e.g., `app.py`, `runner.py`):

```bash
# SSH to Vultr
ssh root@209.222.10.82

# Restart API service
pm2 restart api --update-env

# Restart runner service
pm2 restart runner --update-env

# Check logs for errors
pm2 logs api --lines 50
pm2 logs runner --lines 50
```

#### Step 4: Frontend Changes (React/TypeScript)

If you modified frontend files (e.g., `api.ts`, components):

**Method A: Build Locally and Upload (Recommended)**
```powershell
# On Windows - Build locally
cd C:\quant_system_v2\quant_system_full\UI
npm run build

# Upload built dist folder
scp -r C:\quant_system_v2\quant_system_full\UI\dist\* root@209.222.10.82:/root/quant_system_full/UI/dist/
```

Then on Vultr:
```bash
# Restart frontend service
pm2 restart frontend
```

**Method B: Build on Vultr Server**
```bash
# SSH to Vultr
ssh root@209.222.10.82

# Navigate to UI directory
cd /root/quant_system_full/UI

# Clear old build and rebuild
rm -rf dist/*
npm run build

# Restart frontend
pm2 restart frontend
```

### Cache Busting (Important!)

When frontend changes don't appear, browser/Cloudflare may be caching old JS files.

#### Update Build Version (api.ts)
Before building, update the BUILD_VERSION in `UI/src/lib/api.ts`:
```typescript
const BUILD_VERSION = '2026-01-04-v1'  // Change this date/version
```

#### Force Clean Build
```bash
# On Vultr
cd /root/quant_system_full/UI

# Clean and rebuild with force flag
rm -rf dist/* node_modules/.vite
npx vite build --force

pm2 restart frontend
```

#### Clear Cloudflare Cache
1. Login to Cloudflare Dashboard
2. Go to wgyjdaiassistant.cc domain
3. Caching > Configuration > Purge Everything

#### User Browser Cache
Inform users to hard refresh:
- **Windows**: `Ctrl + Shift + R` or `Ctrl + F5`
- **Mac**: `Cmd + Shift + R`
- Or open Developer Tools (F12) > Network tab > check "Disable cache"

### Environment Variables

#### Frontend (.env.production)
Location: `UI/.env.production`
```bash
VITE_API_BASE_URL=https://api.wgyjdaiassistant.cc
VITE_WS_URL=wss://api.wgyjdaiassistant.cc/ws
```

Important: Vite reads `.env.production` during build time, not runtime. Changes require rebuild.

#### Backend (.env)
Location: `/root/quant_system_full/.env` and `/root/quant_system_full/dashboard/backend/.env`

After changing .env, restart with:
```bash
pm2 restart api --update-env
pm2 restart runner --update-env
```

### Complete Deployment Checklist

```
[ ] 1. Make changes locally
[ ] 2. Test locally if possible
[ ] 3. Update BUILD_VERSION in api.ts (for frontend changes)
[ ] 4. Build frontend locally: npm run build
[ ] 5. SCP changed files to Vultr
[ ] 6. SSH to Vultr
[ ] 7. For backend: pm2 restart api runner --update-env
[ ] 8. For frontend: pm2 restart frontend
[ ] 9. Check logs: pm2 logs --lines 50
[ ] 10. Test in browser with hard refresh (Ctrl+Shift+R)
[ ] 11. If still caching issues: Purge Cloudflare cache
```

### Common Deployment Scenarios

#### Scenario 1: Fix API Bug (Python)
```bash
# Local: Edit app.py
# Transfer:
scp C:\quant_system_v2\quant_system_full\dashboard\backend\app.py root@209.222.10.82:/root/quant_system_full/dashboard/backend/

# Vultr:
ssh root@209.222.10.82
pm2 restart api --update-env
pm2 logs api --lines 30
```

#### Scenario 2: Update Frontend Component
```bash
# Local: Edit component, update BUILD_VERSION, build
cd C:\quant_system_v2\quant_system_full\UI
npm run build

# Transfer built files:
scp -r C:\quant_system_v2\quant_system_full\UI\dist\* root@209.222.10.82:/root/quant_system_full/UI/dist/

# Vultr:
ssh root@209.222.10.82
pm2 restart frontend

# Browser: Ctrl+Shift+R to hard refresh
```

#### Scenario 3: Full System Update
```bash
# Transfer everything:
scp -r C:\quant_system_v2\quant_system_full\dashboard root@209.222.10.82:/root/quant_system_full/
scp -r C:\quant_system_v2\quant_system_full\UI\dist root@209.222.10.82:/root/quant_system_full/UI/

# Vultr:
ssh root@209.222.10.82
pm2 restart all --update-env
pm2 logs --lines 50
```

#### Scenario 4: Emergency Rollback
```bash
# If deployment broke something, restart all services:
ssh root@209.222.10.82
pm2 restart all --update-env

# Check all services running:
pm2 list

# View all logs:
pm2 logs --lines 100
```

### Troubleshooting Deployment Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Frontend shows old content | Browser/Cloudflare cache | Hard refresh + Purge Cloudflare |
| "Unexpected token '<'" | Wrong API URL in JS | Rebuild with correct .env.production |
| 403 after deploy | Session invalidated | Re-login (sessions persist in file now) |
| API not responding | Service crashed | `pm2 restart api`, check `pm2 logs api` |
| Build fails on Vultr | Low memory | Build locally, upload dist folder |

### Useful Deployment Commands Reference

```bash
# Check what's running
pm2 list

# View real-time logs
pm2 logs --lines 0

# Restart single service
pm2 restart api
pm2 restart frontend
pm2 restart runner

# Restart all services
pm2 restart all --update-env

# Check disk space
df -h

# Check memory usage
free -h

# Check Cloudflare tunnel
systemctl status cloudflared
journalctl -u cloudflared -f

# List files in UI dist
ls -la /root/quant_system_full/UI/dist/assets/

# Check which port frontend is using
netstat -tlnp | grep 5173
```

---

## Windows Local Configuration

### Archive Sync Script

Location: `C:\quant_system_v2\sync_archives.ps1`

Schedule via Windows Task Scheduler:
1. Open Task Scheduler
2. Create Basic Task: "Quant Archive Sync"
3. Trigger: Daily at 4:00 AM
4. Action: Start Program
   - Program: `powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\quant_system_v2\sync_archives.ps1"`

### Manual Archive Sync

```powershell
# From Windows PowerShell
scp -r root@209.222.10.82:/root/quant_system_full/archives/* C:\quant_system_v2\archives\
```
