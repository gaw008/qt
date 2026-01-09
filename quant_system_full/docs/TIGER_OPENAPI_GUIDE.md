# Tiger OpenAPI Usage Guide

This document provides a comprehensive guide for using the Tiger OpenAPI SDK in the quant trading system.

## Table of Contents

1. [Configuration](#configuration)
2. [Account Operations](#account-operations)
3. [Position Management](#position-management)
4. [Market Data](#market-data)
5. [Contract Information](#contract-information)
6. [Error Handling](#error-handling)

---

## Configuration

### Required Files

| File | Location (Windows) | Location (Vultr) |
|------|-------------------|------------------|
| Private Key | `private_key.pem` | `/root/quant_system_full/private_key.pem` |
| Properties | `props/tiger_openapi_config.properties` | `/root/quant_system_full/props/tiger_openapi_config.properties` |

### Environment Variables

```bash
TIGER_ID=20550012
ACCOUNT=41169270
# Windows
PRIVATE_KEY_PATH=C:/quant_system_v2/quant_system_full/private_key.pem
# Linux (Vultr)
PRIVATE_KEY_PATH=/root/quant_system_full/private_key.pem
```

### Properties File Format

```properties
private_key_pk1=<RSA_PRIVATE_KEY_CONTENT>
tiger_id=20550012
account=41169270
license=TBUS
env=PROD
sandbox_debug=false
```

### Client Initialization

```python
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
from tigeropen.quote.quote_client import QuoteClient

# Initialize config
props_path = '/path/to/props/'  # Directory containing tiger_openapi_config.properties
config = TigerOpenClientConfig(props_path=props_path)

# Create clients
trade_client = TradeClient(config)
quote_client = QuoteClient(config)
```

---

## Account Operations

### Get Portfolio Summary (Assets)

```python
# Get account assets
account = '41169270'
assets = trade_client.get_prime_assets(account=account, base_currency='USD')

# Access securities segment
segment = assets.segments['S']  # 'S' = Securities, 'C' = Commodities

# Key fields
net_liquidation = segment.net_liquidation  # Total portfolio value
buying_power = segment.buying_power        # Available buying power
cash_balance = segment.cash_balance        # Cash balance (negative = margin used)
unrealized_pl = segment.unrealized_pl      # Unrealized profit/loss
realized_pl = segment.realized_pl          # Realized profit/loss
available_funds = segment.available_funds  # Available funds for trading
```

### Asset Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `net_liquidation` | float | Total portfolio value |
| `buying_power` | float | Available buying power for trading |
| `cash_balance` | float | Cash balance (negative indicates margin usage) |
| `unrealized_pl` | float | Unrealized profit/loss on open positions |
| `realized_pl` | float | Realized profit/loss from closed positions |
| `available_funds` | float | Funds available for new orders |
| `init_margin` | float | Initial margin requirement |
| `maintain_margin` | float | Maintenance margin requirement |
| `gross_position_value` | float | Total value of all positions |

---

## Position Management

### Get All Positions

```python
account = '41169270'
positions = trade_client.get_positions(account=account)

for pos in positions:
    symbol = pos.contract.symbol          # Stock symbol
    quantity = pos.quantity               # Number of shares
    avg_cost = pos.average_cost           # Average cost per share
    market_price = pos.market_price       # Current market price
    market_value = pos.market_value       # Current market value
    unrealized_pnl = pos.unrealized_pnl   # Unrealized P&L
    unrealized_pnl_percent = pos.unrealized_pnl_percent  # Unrealized P&L %

    print(f'{symbol}: {quantity} shares @ ${avg_cost:.2f}, P&L: ${unrealized_pnl:.2f}')
```

### Position Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `contract.symbol` | str | Stock ticker symbol |
| `quantity` | int | Number of shares held |
| `average_cost` | float | Average purchase price per share |
| `market_price` | float | Current market price |
| `market_value` | float | Current total market value |
| `unrealized_pnl` | float | Unrealized profit/loss in dollars |
| `unrealized_pnl_percent` | float | Unrealized P&L as percentage |
| `realized_pnl` | float | Realized P&L for this position |

---

## Market Data

### Get Real-time Stock Quotes

```python
from tigeropen.quote.quote_client import QuoteClient

quote_client = QuoteClient(config)

# Get quotes for multiple stocks (max 50 per request)
symbols = ['AAPL', 'MSFT', 'GOOGL']
briefs = quote_client.get_stock_briefs(symbols, include_hour_trading=False)

# Returns pandas DataFrame with columns:
# symbol, ask_price, bid_price, latest_price, volume, open, high, low, latest_time, status
for _, row in briefs.iterrows():
    print(f"{row['symbol']}: ${row['latest_price']:.2f} (Vol: {row['volume']:,})")
```

### Get Historical Bars (K-Lines)

```python
from tigeropen.common.consts import BarPeriod

# Get daily bars
bars = quote_client.get_bars(
    symbols=['AAPL'],
    period=BarPeriod.DAY,
    begin_time=-1,  # From earliest available
    end_time=-1,    # To latest available
    limit=251       # Number of bars (max 1200)
)

# Available periods:
# BarPeriod.DAY, WEEK, MONTH, YEAR
# BarPeriod.ONE_MINUTE, FIVE_MINUTES, FIFTEEN_MINUTES, THIRTY_MINUTES, SIXTY_MINUTES

# Returns DataFrame with: symbol, time, open, high, low, close, volume, amount
```

### Get Order Book (Depth Quote)

```python
from tigeropen.common.consts import Market

# Get bid/ask depth
depth = quote_client.get_depth_quote(symbols=['AAPL'], market=Market.US)

# Returns dict with 'asks' and 'bids' - each is list of (price, volume, order_count)
for symbol, data in depth.items():
    print(f"{symbol} Asks: {data['asks'][:3]}")  # Top 3 ask levels
    print(f"{symbol} Bids: {data['bids'][:3]}")  # Top 3 bid levels
```

### Get Trade Ticks

```python
# Get recent trades
ticks = quote_client.get_trade_ticks(
    symbols=['AAPL'],
    begin_index=-1,
    end_index=-1,
    limit=200  # Max 200 per request
)

# Returns DataFrame with: index, price, volume, direction (+/-)
```

### Rate Limits

| API | Limit | Notes |
|-----|-------|-------|
| Stock quotes | 50 symbols/request | A-shares: 30 max |
| K-line bars | 1200 records/request | |
| Minute K-lines | 1 month history | |
| 15/30/60-min K-lines | 1 year history | |

---

## Contract Information

### Get Stock Contract

```python
from tigeropen.common.consts import SecurityType

# Get single contract
contract = trade_client.get_contract('AAPL', sec_type=SecurityType.STK)

print(f"Symbol: {contract.symbol}")
print(f"Name: {contract.name}")
print(f"Currency: {contract.currency}")
print(f"Shortable: {contract.shortable}")
print(f"Marginable: {contract.marginable}")
print(f"Min Tick: {contract.min_tick}")
```

### Get Multiple Contracts

```python
# Get contracts for multiple symbols (max 50)
contracts = trade_client.get_contracts('AAPL')

for contract in contracts:
    print(f"{contract.symbol}: {contract.name}")
```

### Get Options/Derivatives

```python
# Get options for a specific expiry
options = trade_client.get_derivative_contracts(
    symbol='AAPL',
    sec_type=SecurityType.OPT,
    expiry='20260116',  # Format: yyyyMMdd
    lang='en_US'
)
```

### Contract Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `identifier` | str | Unique contract ID |
| `symbol` | str | Stock ticker |
| `sec_type` | str | Security type (STK, OPT, FUT, WAR) |
| `name` | str | Contract name |
| `currency` | str | Trading currency (USD, HKD, etc.) |
| `shortable` | bool | Can be shorted |
| `marginable` | bool | Can use margin |
| `min_tick` | float | Minimum price increment |
| `status` | int | 0=not tradeable, 1=tradeable |
| `expiry` | str | Expiration date (options/futures) |
| `strike` | float | Strike price (options) |
| `put_call` | str | CALL or PUT (options) |
| `multiplier` | float | Contract multiplier |

---

## Order Operations

### Place Market Order

```python
from tigeropen.trade.domain.order import Order
from tigeropen.common.consts import OrderType, OrderStatus

# Create order
order = Order(
    account=account,
    contract=contract,
    action='BUY',  # 'BUY' or 'SELL'
    order_type='MKT',
    quantity=100
)

# Place order
order_id = trade_client.place_order(order)
print(f"Order placed: {order_id}")
```

### Place Limit Order

```python
order = Order(
    account=account,
    contract=contract,
    action='BUY',
    order_type='LMT',
    quantity=100,
    limit_price=150.00
)

order_id = trade_client.place_order(order)
```

### Get Orders

```python
# Get all orders
orders = trade_client.get_orders(account=account)

for order in orders:
    print(f"{order.id}: {order.action} {order.quantity} {order.contract.symbol}")
    print(f"  Status: {order.status}")
    print(f"  Filled: {order.filled_quantity} @ ${order.avg_fill_price}")
```

### Cancel Order

```python
# Cancel by order ID
trade_client.cancel_order(id=order_id)
```

---

## Error Handling

### Common Error Patterns

```python
try:
    assets = trade_client.get_prime_assets(account=account)
except Exception as e:
    if 'signature' in str(e).lower():
        print("Private key error - check PRIVATE_KEY_PATH")
    elif 'account' in str(e).lower():
        print("Account error - check ACCOUNT environment variable")
    elif 'connection' in str(e).lower():
        print("Network error - check internet connection")
    else:
        print(f"Tiger API error: {e}")
```

### SDK Import Check

```python
try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient
    TIGER_SDK_AVAILABLE = True
except ImportError:
    TIGER_SDK_AVAILABLE = False
    print("Tiger SDK not installed. Run: pip install tigeropen")
```

### Private Key Validation

```python
import os

private_key_path = os.getenv('PRIVATE_KEY_PATH')
if not private_key_path or not os.path.exists(private_key_path):
    print(f"ERROR: Private key not found at: {private_key_path}")
else:
    print(f"Private key found at: {private_key_path}")
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Portfolio shows 0 | Wrong PRIVATE_KEY_PATH | Fix path in .env files |
| "Private key not found" | Windows path on Linux | Change to Linux path |
| "Signature error" | Invalid private key | Regenerate key from Tiger |
| "Account not found" | Wrong ACCOUNT | Verify account number |
| Empty positions | No holdings | Normal if no positions |
| Rate limit errors | Too many requests | Add delays between calls |

### Debug Connection

```python
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Tiger API Debug ===")
print(f"TIGER_ID: {os.getenv('TIGER_ID')}")
print(f"ACCOUNT: {os.getenv('ACCOUNT')}")
print(f"PRIVATE_KEY_PATH: {os.getenv('PRIVATE_KEY_PATH')}")

key_path = os.getenv('PRIVATE_KEY_PATH')
print(f"Key exists: {os.path.exists(key_path) if key_path else False}")

try:
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient

    config = TigerOpenClientConfig(props_path='props')
    client = TradeClient(config)

    assets = client.get_prime_assets(account=os.getenv('ACCOUNT'))
    print(f"Connection SUCCESS - Net Liquidation: ${assets.segments['S'].net_liquidation:,.2f}")
except Exception as e:
    print(f"Connection FAILED: {e}")
```

---

## API Reference Links

- [Tiger OpenAPI Documentation](https://docs.itigerup.com/)
- [Stock Quotes API](https://docs.itigerup.com/docs/quote-stock)
- [Account API](https://docs.itigerup.com/docs/accounts)
- [Contract API](https://docs.itigerup.com/docs/get-contract)
- [Rate Limits](https://docs.itigerup.com/docs/ratelimit)
- [Python SDK GitHub](https://github.com/tigerbrokers/openapi-python-sdk)
