---
title: 获取订单信息
---
### get_orders 获取订单列表

`
TradeClient.get_orders(account=None, sec_type=None, market=Market.ALL, symbol=None, start_time=None, end_time=None, limit=100, is_brief=False, states=None)
`

**说明**

获取账户的订单历史，包括所有状态、各个证券类型的订单，可以传参数进行筛选

**参数**

参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
sec_type| SecurityType |证券类型， 可以使用 tigeropen.common.consts.SecurityType 下的常量
market|Market|所属市场，可以使用 tigeropen.common.consts.Market 下的常量 
symbol|str|证券代码
start_time|str或int|起始时间。毫秒单位时间戳或日期字符串，如 1643346000000 或 '2019-01-01' 或 '2019-01-01 12:00:00
end_time|str或int|截至时间。毫秒单位时间戳或日期字符串，如 1653346000000 或 '2019-11-01' 或 '2019-11-01 15:00:00
limit|int|每次获取订单的数量
is_brief|bool|布尔型，是否返回精简的订单数据
status|OrderStatus|订单状态，可以使用 tigeropen.common.consts.OrderStatus 的枚举
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置

**返回**

`list`

列表中的每个元素都是一个 Order 对象（tigeropen.trade.domain.order.Order），具体字段含义详见 [Order 对象](/zh/python/appendix1/object.html#order-订单)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

orders = trade_client.get_orders(sec_type=SecurityType.STK, market=Market.ALL)
```

**返回示例**

```python
[Order({'account': '1', 'id': 162998104807903232, 'order_id': 341, 'parent_id': 0, 'order_time': 1557972846184, 'reason': '136:Order is already being cancelled.', 'trade_time': 1557975394512, 'action': 'BUY', 'quantity': 2, 'filled': 0, 'avg_fill_price': 0, 'commission': 0, 'realized_pnl': 0, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': 'DAY', 'outside_rth': True, 'contract': SPY, 'status': 'CANCELLED', 'remaining': 2}),
Order({'account': '1', 'id': 162998998620389376, 'order_id': 344, 'parent_id': 0, 'order_time': 1557973698590, 'reason': '136:Order is already being cancelled.', 'trade_time': 1557973773622, 'action': 'BUY', 'quantity': 1, 'filled': 0, 'avg_fill_price': 0, 'commission': 0, 'realized_pnl': 0, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': 'DAY', 'outside_rth': True, 'contract': SPY, 'status': 'CANCELLED', 'remaining': 1}),
Order({'account': '1', 'id': 152239266327625728, 'order_id': 230, 'parent_id': 0, 'order_time': 1547712418243, 'reason': '201:Order rejected - Reason: YOUR ORDER IS NOT ACCEPTED. IN ORDER TO OBTAIN THE DESIRED POSITION YOUR EQUITY WITH LOAN VALUE [1247.90 USD] MUST EXCEED THE INITIAL MARGIN [4989.99 USD]', 'trade_time': 1547712418275, 'action': 'BUY', 'quantity': 100, 'filled': 0, 'avg_fill_price': 0, 'commission': 0, 'realized_pnl': 0, 'trail_stop_price': None, 'limit_price': 5, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': 'DAY', 'outside_rth': True, 'contract': AAPL, 'status': 'REJECTED', 'remaining': 100})]
```
---


### get_order 获取指定订单

`
TradeClient.get_order(account=None, id=None, order_id=None, is_brief=False)
`

**说明**

通过id获取指定的订单

**参数**

参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
id|int|在提交订单后返回的全局订单id
order_id|int|本地订单id
is_brief|bool|是否返回精简的订单数据
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置，个人开发者无需关注

**返回**

`Order`对象

具体字段含义详见 [Order 对象](/zh/python/appendix1/object.html#order-订单)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

order = trade_client.get_order(id=1230001200123)
```

**返回示例**

```python
{'account': '1', 'id': 162998104807903232, 'order_id': 341, 'parent_id': 0, 'order_time': 1557972846184, 'reason': '136:Order is already being cancelled.', 'trade_time': 1557975394512, 'action': 'BUY', 'quantity': 2, 'filled': 0, 'avg_fill_price': 0, 'commission': 0, 'realized_pnl': 0, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': 'DAY', 'outside_rth': True, 'contract': SPY, 'status': 'CANCELLED', 'remaining': 2}
```


---


### get\_open\_orders 获取待成交的订单列表

`
TradeClient.get_open_orders(account=None, sec_type=None, market=Market.ALL, symbol=None, start_time=None, end_time=None)
`

**说明**

获取待成交的订单列表

**参数**

参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
sec_type|SecurityType|证券类型，可以使用 tigeropen.common.consts.SecurityType 下的常量
market|Market|所属市场，可以使用 tigeropen.common.consts.Market 下的常量，如 Market.US
symbol|str|证券代码
start_time|str或int|开始时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07'
end_time|str或int|截至时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
secret_key|str|机构交易员密钥，机构用户专有，个人开发者无需关注，需要在client_config中配置

**返回**

`list`

列表中的每个元素是一个 Order 对象（tigeropen.trade.domain.order.Order）, 具体字段含义详见[Order对象](/zh/python/appendix1/object.html#order-订单)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

open_orders = trade_client.get_open_orders(sec_type=SecurityType.STK, market=Market.ALL)
```

**返回示例**

同 get_orders 

---

### get\_cancelled\_orders 获取已撤销的订单列表

`
TradeClient.get_cancelled_orders(account=None, sec_type=None, market=Market.ALL, symbol=None, start_time=None, end_time=None)
`

**说明**

获取已撤销的订单列表。包括主动撤销、系统撤销、已失效的订单等。

**参数**
参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
sec_type|SecurityType|证券类型，可以使用 tigeropen.common.consts.SecurityType 下的常量
market|Market|所属市场，可以使用 tigeropen.common.consts.Market 下的常量，如 Market.US
symbol|str|证券代码
start_time|str或int|开始时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07'
end_time|str或int|截至时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
secret_key|str|机构交易员密钥，机构用户专有，个人开发者无需关注，需要在client_config中配置

**返回**

`list`

list 中的每个元素都是一个 Order 对象，具体字段含义详见[Order对象](/zh/python/appendix1/object.html#order-订单)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

cancelled_orders = trade_client.get_cancelled_orders(sec_type=SecurityType.STK, market=Market.ALL)
```

**返回示例**

同 get_orders

---

### get\_filled\_orders 获取已成交的订单列表

`
TradeClient.get_filled_orders(account=None, sec_type=None, market=Market.ALL, symbol=None, start_time=None, end_time=None)
`

**说明**
获取已成交订单列表

**参数**
参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
sec_type|SecurityType|证券类型，可以使用 tigeropen.common.consts.SecurityType 下的常量
market|Market|所属市场，可以使用 tigeropen.common.consts.Market 下的常量，如 Market.US
symbol|str|证券代码
start_time|str或int|开始时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07'
end_time|str或int|截至时间。毫秒级别的时间戳，或日期时间字符串，如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
secret_key|str|机构交易员密钥，机构用户专有，个人开发者无需关注，需要在client_config中配置

**返回**
`list`

list 中的每个元素都是一个 Order 对象，具体字段含义详见[Order对象](/zh/python/appendix1/object.html#order-订单)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

filled_orders = trade_client.get_filled_orders(sec_type=SecurityType.STK, market=Market.ALL)
```

**返回示例**

同 get_orders

---

### get\_transactions 获取订单成交记录  
**说明**  
获取已成交订单的详细成交记录(仅适用于综合账户)。

**参数**  
参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
order_id|int|订单id
symbol|str|标的代码。使用symbol查询时sec_type为必传
sec_type| SecurityType |证券类型， 可以使用 tigeropen.common.consts.SecurityType 下的常量
market|Market|所属市场，可以使用 tigeropen.common.consts.Market 下的常量 
start_time|str或int|起始时间。毫秒单位时间戳或日期字符串，如 1643346000000 或 '2019-01-01' 或 '2019-01-01 12:00:00
end_time|str或int|截至时间。毫秒单位时间戳或日期字符串，如 1653346000000 或 '2019-11-01' 或 '2019-11-01 15:00:00
limit|int|每次获取记录的数量
expiry|str|过期日(适用于期权)。 形式 'yyyyMMdd', 比如 '220121'|
strike|float|行权价(适用于期权)。如 100.5
put_call|str|看涨或看跌(适用于期权)。'PUT' 或 'CALL'

**返回**
`list`
列表中每个元素为 `Transaction` 对象，具体字段含义见 [Transaction](/zh/python/appendix1/object.html#transaction-成交记录)

**示例**
```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

filled_orders = trade_client.get_transactions(symbol='AAPL', sec_type=SecurityType.STK)
```

**返回示例**
```
Transaction({'account': 111111, 'order_id': 20947299719447552, 'contract': AAPL/STK/USD, 'id': 20947300069016576, 'action': 'BUY', 'filled_quantity': 1, 'filled_price': 132.25, 'filled_amount': 132.25, 'transacted_at': '2020-12-23 17:06:54'}), 

Transaction({'account': 111111, 'order_id': 19837920138101760, 'contract': AAPL/STK/USD, 'id': 19837920740508672, 'action': 'BUY', 'filled_quantity': 1, 'filled_price': 116.21, 'filled_amount': 116.21, 'transacted_at': '2020-09-16 18:02:00'})]
        
```