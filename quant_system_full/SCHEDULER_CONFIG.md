# Market-Aware Scheduler Configuration Guide

## Environment Variables

The scheduler system supports configuration through environment variables:

### Core Scheduler Settings
```bash
# Market type (US or CN)
PRIMARY_MARKET=US

# Task execution intervals (seconds)
SELECTION_INTERVAL=3600      # Stock selection tasks (default: 1 hour)
TRADING_INTERVAL=30          # Trading monitoring tasks (default: 30 seconds)  
MONITORING_INTERVAL=120      # Market monitoring tasks (default: 2 minutes)

# Performance settings
MAX_CONCURRENT_TASKS=3       # Maximum concurrent tasks (default: 3)
```

### Data Source Configuration
```bash
# Data source priority
DATA_SOURCE=auto             # Options: auto, yahoo_api, yahoo_mcp, tiger

# Yahoo Finance API settings
YAHOO_API_TIMEOUT=30         # Request timeout (seconds)
YAHOO_API_RETRIES=3          # Retry attempts
YAHOO_API_RETRY_DELAY=1      # Delay between retries (seconds)

# MCP integration
USE_MCP_TOOLS=true           # Enable Yahoo Finance MCP tools
YAHOO_MCP_PROXY_URL=         # Optional proxy for MCP requests
```

### Market Data Settings
```bash
# Trading mode
DRY_RUN=true                 # Use placeholder data instead of live feeds

# Tiger Brokers configuration (for live trading)
TIGER_ID=your_tiger_id
ACCOUNT=your_account_number
PRIVATE_KEY_PATH=private_key.pem
```

### State Management
```bash
# State directory override
STATE_DIR=dashboard/state    # Default state directory

# Dashboard authentication
ADMIN_TOKEN=your_admin_token # Required for API access
```

## Market Schedule Support

### US Market Hours (Eastern Time)
- **Pre-market**: 4:00 AM - 9:30 AM
- **Regular**: 9:30 AM - 4:00 PM  
- **After-hours**: 4:00 PM - 8:00 PM
- **Closed**: 8:00 PM - 4:00 AM (next day), Weekends

### Chinese Market Hours (Shanghai Time)
- **Pre-market**: 8:00 AM - 9:30 AM
- **Regular**: 9:30 AM - 3:00 PM (with 11:30 AM - 1:00 PM break)
- **After-hours**: 3:00 PM - 5:00 PM
- **Closed**: 5:00 PM - 8:00 AM (next day), Weekends

## Task Execution Matrix

| Market Phase  | Selection Tasks | Trading Tasks | Monitoring Tasks |
|---------------|-----------------|---------------|------------------|
| Closed        | ✓ Yes          | ✗ No          | ✓ Yes            |
| Pre-market    | ✗ No           | ✓ Yes         | ✓ Yes            |
| Regular       | ✗ No           | ✓ Yes         | ✓ Yes            |
| After-hours   | ✗ No           | ✓ Yes         | ✓ Yes            |

## Selection Strategy Configuration

### Universe Definition
The scheduler uses a curated universe of liquid stocks across sectors:
- Technology: AAPL, MSFT, GOOGL, AMZN, NVDA, META, etc.
- Healthcare: UNH, JNJ, PFE, ABBV, MRK, TMO, etc.
- Financial: BRK-B, JPM, BAC, WFC, GS, MS, etc.
- Consumer: WMT, HD, PG, KO, PEP, NKE, etc.
- Industrial/Energy: CAT, BA, GE, XOM, CVX, etc.

### Selection Criteria
Default criteria for stock selection:
```python
SelectionCriteria(
    max_stocks=10,              # Maximum stocks to select
    min_market_cap=1e9,         # $1B minimum market cap
    max_market_cap=1e12,        # $1T maximum market cap
    min_volume=100000,          # Minimum daily volume
    min_price=5.0,              # Minimum stock price
    max_price=500.0,            # Maximum stock price
    min_score_threshold=50.0,   # Minimum strategy score
    exclude_sectors=[],         # Sectors to exclude
    include_sectors=[],         # Sectors to include (if specified)
    exclude_symbols=[]          # Specific symbols to exclude
)
```

## Monitoring and Health Checks

### Task Health Metrics
- **Healthy Tasks**: Tasks with no errors and recent execution
- **Error Tasks**: Tasks with execution failures
- **Task Statistics**: Run counts and execution times
- **Error Tracking**: Per-task error counts and timestamps

### System Status Updates
Status information is written to `dashboard/state/status.json`:
```json
{
  "market_status": {...},
  "task_statistics": {...},
  "task_health": {...},
  "selection_results": {...},
  "scheduler_uptime": "Running"
}
```

### Logging
- Market phase transitions are logged
- Task execution times are tracked
- Errors are logged with details
- Selection results are summarized

## Testing Configuration

For testing the scheduler system:
```bash
# Set shorter intervals for testing
export SELECTION_INTERVAL=60
export TRADING_INTERVAL=10  
export MONITORING_INTERVAL=30
export DRY_RUN=true

# Run test suite
python test_scheduler.py
```

## Kill Switch Integration

The scheduler integrates with the kill switch mechanism:
- Checks `dashboard/state/kill.flag` every cycle
- Pauses task execution when kill switch is active
- Continues monitoring for kill switch removal
- Logs kill switch status changes

## Production Deployment

Recommended production settings:
```bash
PRIMARY_MARKET=US
SELECTION_INTERVAL=3600    # 1 hour
TRADING_INTERVAL=30        # 30 seconds
MONITORING_INTERVAL=300    # 5 minutes
MAX_CONCURRENT_TASKS=5
DRY_RUN=false
USE_MCP_TOOLS=true
DATA_SOURCE=auto
```

For high-frequency scenarios:
```bash
TRADING_INTERVAL=5         # 5 seconds
MONITORING_INTERVAL=60     # 1 minute
MAX_CONCURRENT_TASKS=10
```