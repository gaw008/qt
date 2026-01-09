# Quant System API Documentation

## Overview

The Quant System API provides comprehensive access to the quantitative trading system's functionality, including market data, trading operations, risk management, and system monitoring. The API is built with FastAPI and includes automatic interactive documentation.

**Base URL:** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs`
**OpenAPI Schema:** `http://localhost:8000/openapi.json`

## Authentication

All API endpoints (except `/health`) require Bearer token authentication.

```
Authorization: Bearer <your_token>
```

Set the `ADMIN_TOKEN` environment variable or use the default "changeme" for development.

## API Endpoints

### Core System Endpoints

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "time": 1693478400
}
```

#### GET /status
Get current bot status and system information.

**Response:**
```json
{
  "bot": "running",
  "heartbeat": 1693478400,
  "pnl": 1250.50,
  "positions": [],
  "last_signal": "BUY AAPL",
  "paused": false,
  "reason": null
}
```

#### POST /kill
Pause/kill the trading bot.

**Request Body:**
```json
{
  "reason": "manual stop"
}
```

#### POST /resume
Resume the trading bot.

**Request Body:**
```json
{
  "note": "resuming operations"
}
```

#### GET /logs
Get recent log entries.

**Parameters:**
- `n` (int): Number of log lines to retrieve (1-2000, default 200)

**Response:**
```json
{
  "lines": [
    "2023-08-30 10:30:00 - Bot started",
    "2023-08-30 10:35:00 - Analysis complete"
  ]
}
```

### Market Data API

#### GET /api/markets/assets
Get list of available assets with basic information.

**Parameters:**
- `limit` (int): Number of assets to return (1-10000, default 100)
- `offset` (int): Number of assets to skip (default 0)
- `asset_type` (str): Filter by asset type (stock, etf, reit, adr, futures)

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "asset_type": "stock",
    "price": 175.43,
    "volume": 58420000,
    "market_cap": 2890000000000,
    "currency": "USD",
    "exchange": "NASDAQ",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "change_percent": 1.25,
    "change_amount": 2.17
  }
]
```

#### GET /api/markets/heatmap
Get market heatmap data for visualization.

**Parameters:**
- `limit` (int): Number of assets to include (1-1000, default 100)
- `sector` (str): Filter by sector

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "market_cap": 2890000000000,
    "price_change_percent": 1.25,
    "volume_change_percent": 15.3,
    "performance_1d": 1.25,
    "performance_1w": -2.1,
    "performance_1m": 8.7
  }
]
```

#### POST /api/markets/filter
Filter assets based on specified criteria.

**Request Body:**
```json
{
  "asset_types": ["stock", "etf"],
  "sectors": ["Technology", "Healthcare"],
  "min_market_cap": 1000000000,
  "max_market_cap": 1000000000000,
  "min_volume": 1000000,
  "min_price": 10.0,
  "max_price": 500.0,
  "performance_range": {
    "min_1d": -5.0,
    "max_1d": 5.0
  }
}
```

**Parameters:**
- `limit` (int): Maximum results to return (1-1000, default 100)

### Trading API

#### GET /api/positions
Get all current trading positions.

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "quantity": 100,
    "entry_price": 170.50,
    "current_price": 175.43,
    "position_type": "LONG",
    "entry_time": "2023-08-30T10:30:00",
    "market_value": 17543.0,
    "unrealized_pnl": 493.0,
    "realized_pnl": 0.0,
    "total_pnl": 493.0,
    "stop_loss_price": 162.0,
    "take_profit_price": 185.0,
    "sector": "Technology"
  }
]
```

#### GET /api/positions/{symbol}
Get position information for a specific symbol.

**Path Parameters:**
- `symbol` (str): Stock symbol (e.g., "AAPL")

#### GET /api/orders
Get order history and status.

**Parameters:**
- `status` (str): Filter by order status (PENDING, FILLED, CANCELLED, REJECTED)
- `symbol` (str): Filter by symbol
- `limit` (int): Maximum results (1-1000, default 100)

**Response:**
```json
[
  {
    "order_id": "12345678",
    "symbol": "AAPL",
    "order_type": "LIMIT",
    "side": "BUY",
    "quantity": 100,
    "filled_quantity": 100,
    "remaining_quantity": 0,
    "price": 170.50,
    "filled_price": 170.45,
    "status": "FILLED",
    "created_time": "2023-08-30T10:30:00",
    "updated_time": "2023-08-30T10:31:00"
  }
]
```

#### POST /api/orders
Create a new trading order.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "order_type": "LIMIT",
  "side": "BUY",
  "quantity": 100,
  "price": 170.00,
  "stop_price": null,
  "time_in_force": "DAY"
}
```

**Response:** Returns the created Order object.

#### GET /api/orders/{order_id}
Get specific order by ID.

**Path Parameters:**
- `order_id` (str): Unique order identifier

#### POST /api/orders/{order_id}/cancel
Cancel a pending order.

**Path Parameters:**
- `order_id` (str): Unique order identifier

**Response:**
```json
{
  "success": true,
  "message": "Order 12345678 cancelled successfully"
}
```

### Risk Management API

#### GET /api/risk/var
Get Value at Risk (VaR) metrics for the portfolio.

**Response:**
```json
{
  "var_1d_95": 5000.0,
  "var_1d_99": 7500.0,
  "var_5d_95": 11180.0,
  "var_5d_99": 16770.0,
  "expected_shortfall_95": 6180.0,
  "expected_shortfall_99": 7980.0,
  "portfolio_value": 100000.0,
  "daily_volatility": 0.0095
}
```

#### GET /api/risk/exposure
Get portfolio exposure by sector.

**Response:**
```json
[
  {
    "sector": "Technology",
    "exposure_amount": 45000.0,
    "exposure_percent": 45.0,
    "position_count": 8,
    "avg_return": 2.5,
    "risk_contribution": 0.45
  }
]
```

#### GET /api/risk/drawdown
Get maximum drawdown and drawdown metrics.

**Response:**
```json
{
  "max_drawdown_percent": -8.5,
  "max_drawdown_amount": -8500.0,
  "current_drawdown_percent": -2.1,
  "current_drawdown_amount": -2100.0,
  "days_in_drawdown": 12,
  "time_to_recovery_estimate": 18,
  "peak_value": 108500.0,
  "current_value": 100000.0,
  "drawdown_periods": 5,
  "avg_recovery_time": 24.5
}
```

#### GET /api/risk/metrics
Get comprehensive risk metrics for the portfolio.

**Response:**
```json
{
  "var_1d": 5000.0,
  "var_5d": 11180.0,
  "expected_shortfall": 6180.0,
  "beta": 1.1,
  "max_drawdown": -8.5,
  "sharpe_ratio": 0.85,
  "sortino_ratio": 1.15
}
```

### System Performance API

#### GET /api/system/performance
Get system performance metrics.

**Response:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 62.8,
  "gpu_usage": 35.1,
  "disk_usage": 78.5,
  "network_io": {
    "bytes_sent": 1024000,
    "bytes_recv": 2048000,
    "packets_sent": 1500,
    "packets_recv": 2200
  },
  "process_count": 287,
  "uptime": 86400.0,
  "load_average": [0.5, 0.7, 0.8]
}
```

#### GET /api/system/health
Extended system health check with comprehensive metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-08-30T15:30:00",
  "uptime": 86400.0,
  "bot_status": {
    "bot": "running",
    "paused": false
  },
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "disk_usage": 78.5,
    "gpu_usage": 35.1,
    "process_count": 287,
    "load_average": [0.5, 0.7, 0.8]
  },
  "network": {
    "io_counters": {
      "bytes_sent": 1024000,
      "bytes_recv": 2048000,
      "packets_sent": 1500,
      "packets_recv": 2200
    }
  },
  "warnings": [],
  "errors": [],
  "recommendations": [
    "Monitor CPU usage trends",
    null,
    null
  ]
}
```

## Data Models

### AssetInfo
Basic asset information structure.

**Fields:**
- `symbol` (str): Asset symbol
- `name` (str): Full asset name
- `asset_type` (str): Type of asset (stock, etf, reit, adr, futures)
- `price` (float): Current price
- `volume` (int): Trading volume
- `market_cap` (float): Market capitalization
- `currency` (str): Trading currency
- `exchange` (str): Exchange name
- `sector` (str): Business sector
- `industry` (str): Business industry
- `change_percent` (float): Price change percentage
- `change_amount` (float): Absolute price change

### Position
Trading position information.

**Fields:**
- `symbol` (str): Asset symbol
- `quantity` (int): Position size
- `entry_price` (float): Entry price
- `current_price` (float): Current market price
- `position_type` (str): Position type (LONG, SHORT, NEUTRAL)
- `entry_time` (str): Entry timestamp
- `market_value` (float): Current market value
- `unrealized_pnl` (float): Unrealized profit/loss
- `realized_pnl` (float): Realized profit/loss
- `total_pnl` (float): Total profit/loss
- `stop_loss_price` (float): Stop loss price
- `take_profit_price` (float): Take profit price
- `sector` (str): Asset sector

### Order
Order information structure.

**Fields:**
- `order_id` (str): Unique order identifier
- `symbol` (str): Asset symbol
- `order_type` (str): Order type (MARKET, LIMIT, STOP)
- `side` (str): Order side (BUY, SELL)
- `quantity` (int): Order quantity
- `filled_quantity` (int): Filled quantity
- `remaining_quantity` (int): Remaining quantity
- `price` (float, optional): Order price
- `filled_price` (float, optional): Filled price
- `status` (str): Order status (PENDING, FILLED, CANCELLED, REJECTED)
- `created_time` (str): Creation timestamp
- `updated_time` (str, optional): Last update timestamp

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing or invalid token)
- `403` - Forbidden (access denied)
- `404` - Not Found (resource not found)
- `500` - Internal Server Error

Error responses include detailed error messages:

```json
{
  "detail": "Position for INVALID not found"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production deployments.

## Integration Notes

### Data Sources
The API integrates with:
- Multi-asset data manager for market data
- Portfolio management system for positions
- Risk management modules for risk calculations
- System monitoring tools for performance metrics
- Tiger API for real-time data (when available)

### State Management
- Orders are stored in `dashboard/state/orders.json`
- Portfolio data is stored in `dashboard/state/portfolio.json`
- System status is managed through `state_manager.py`

### Mock Data
When real data sources are unavailable, the API returns realistic mock data to ensure frontend functionality during development and testing.

## Usage Examples

### Python Client Example
```python
import requests

# Set up authentication
headers = {"Authorization": "Bearer your_token_here"}

# Get market assets
response = requests.get("http://localhost:8000/api/markets/assets", headers=headers)
assets = response.json()

# Create an order
order_data = {
    "symbol": "AAPL",
    "order_type": "MARKET",
    "side": "BUY",
    "quantity": 10
}
response = requests.post("http://localhost:8000/api/orders", json=order_data, headers=headers)
order = response.json()

# Get risk metrics
response = requests.get("http://localhost:8000/api/risk/metrics", headers=headers)
risk_metrics = response.json()
```

### JavaScript/React Example
```javascript
const API_BASE = 'http://localhost:8000';
const headers = {
  'Authorization': 'Bearer your_token_here',
  'Content-Type': 'application/json'
};

// Get positions
const getPositions = async () => {
  const response = await fetch(`${API_BASE}/api/positions`, { headers });
  return response.json();
};

// Create order
const createOrder = async (orderData) => {
  const response = await fetch(`${API_BASE}/api/orders`, {
    method: 'POST',
    headers,
    body: JSON.stringify(orderData)
  });
  return response.json();
};
```

## Development and Testing

### Starting the API Server
```bash
cd dashboard/backend
python app.py
```

The server will start on `http://localhost:8000` with auto-reload enabled for development.

### Interactive Documentation
Visit `http://localhost:8000/docs` for automatic interactive API documentation powered by Swagger UI.

### Testing Endpoints
Use the interactive documentation or tools like curl, Postman, or Python requests to test API endpoints.

Example curl command:
```bash
curl -H "Authorization: Bearer changeme" http://localhost:8000/api/markets/assets?limit=5
```

## Security Considerations

1. **Authentication**: Change the default ADMIN_TOKEN in production
2. **HTTPS**: Use HTTPS in production environments
3. **CORS**: Configure CORS settings for frontend integration
4. **Input Validation**: All endpoints include input validation via Pydantic models
5. **Error Handling**: Sensitive information is not exposed in error messages

## Future Enhancements

1. **WebSocket Support**: Real-time data streaming
2. **Advanced Authentication**: JWT tokens, OAuth integration
3. **Rate Limiting**: Implement request rate limiting
4. **Caching**: Redis-based caching for improved performance
5. **Logging**: Structured logging with correlation IDs
6. **Metrics**: Prometheus metrics integration
7. **Database Integration**: Replace file-based state with database