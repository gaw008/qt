---
title: 取消或修改订单
---
### cancel_order 取消订单

`
TradeClient.cancel_order(account=None, id=None, order_id=None)
`

**说明** 

撤销已下的订单

**参数** 

参数名|类型|描述
----|----|----
account|str|账户id，若不填则使用 client_config 中的默认 account
id|int|全局订单id，取消订单操作建议使用id字段，在 place_order 之后，可以通过 Order.id 获取
order_id|int|本地订单order_id， 可以通过 Order.order_id 获取
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置

**返回**

`bool`

True 表示取消成功 ， False 表示取消失败

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)

is_cancelled = trade_client.cancel_order(id=1230001200123)
```

**返回示例**

```python
True
```

### modify_order 修改订单

`
TradeClient.modify_order(order, quantity,  limit_price, aux_price, trail_stop_price, trailing_percent, time_in_force, outside_rth)
`

**说明** 

修改已下订单，具体可修改的字段见下表

**参数**

参数名|类型|描述
----|----|----
order|Order|要修改的 Order 对象(tigeropen.trade.domain.order.Order)
quantity|int|修改后的下单股数
limit_price|float|修改后的限价，当 order_type 为LMT,STP,STP_LMT时该参数必需
aux_price|float|对于限价止损单，表示触发价， 对于移动止损单，表示价差。当 order_type 为STP,STP_LMT时该参数必需，
trail_stop_price|float|当 order_type 为 TRAIL 时必须，为触发止损单的价格
trailing_percent|float|跟踪止损单-百分比 ，当 order_type 为 TRAIL时,aux_price和trailing_percent两者互斥
time_in_force|str|订单有效期，只能是 'DAY'（当日有效）和'GTC'（取消前有效），默认为'DAY'
outside_rth|bool|是否允许盘前盘后交易(美股专属)
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置

**返回**

`bool`

True 表示修改成功 ， False 表示修改失败

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)



contract = stock_contract(symbol='AAPL', currency='USD')
order = limit_order(account=client_config.account, contract=contract, action='BUY', limit_price=100.0, quantity=1)
trade_client.place_order(order)

is_modified = trade_client.modify_order(order, quantity=2, limit_price=105.0)
```

**返回示例**

```python
True
```