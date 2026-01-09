---
title: 下单交易
---
### place_oder 下单
`
TradeClient.place_order():
`

**说明**

交易下单接口。关于如何选择标的、订单类型、方向数量等，请见下方说明。**请在运行程序前结合本文档的[概述](/zh/python/overview/introduction.html)部分及[FAQ-交易-支持的订单列表](/zh/python/FAQ/trade.html#支持交易的订单类型)部分，检查您的账户是否支持所请求的订单，并检查交易规则是否允许在程序运行时段对特定标的下单**。若下单失败，可首先阅读文档[FAQ-交易](/zh/python/FAQ/trade.html#下单失败排查方法)部分排查

下单成功后，参数里的order对象的订单id即被填充（`order.id`），可用于后续的查询或撤单。  

对于附加订单，仅主订单为限价单时支持。

**参数**

`Order`对象 ([`tigeropen.trade.domain.order.Order`](/zh/python/appendix1/object.html#order-订单))

可用 `tigeropen.common.util.order_utils` 下的工具函数，如 limit_order()，market_order(), ，根据您需要的具体订单类型和参数，在本地生成订单对象。创建方法详见[Order对象-构建方法](/zh/python/appendix1/object.html#构建方法)部分
或者用 `TradeClient.create_order()` 向服务端请求订单号，然后生成订单对象(不推荐)  


订单常用属性: 

参数名|类型|描述
----|----|----
action|str|买卖方向，'BUY'表示买入，'SELL'表示卖出
order_type|str|订单类型，'MKT' 市价单 / 'LMT' 限价单 / 'STP' 止损单 / 'STP_LMT' 止损限价单 / 'TRAIL' 跟踪止损单
time_in_force|str|订单有效期，'DAY'当日有效，'GTC'取消前有效
limit_price|float|限价的价格
quantity|int|下单数量，必须为大于0的整数。如果不是美股，quantity需要为股票每手股数的整数倍，可以用 [TradeClient.get_trade_metas](/zh/python/operation/quotation/stock.html#get-trade-metas-获取股票交易需要的信息) 查询每手股数

**返回** 

如果下单成功则返回`True`，失败抛出异常。
若成功下单，则参数Order对象的id会被填充为实际的订单号

**示例1**
```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)

# 生成股票合约
contract = stock_contract(symbol='AAPL', currency='USD')
# 生成订单对象
order = limit_order(account=client_config.account, contract=contract, action='BUY', limit_price=0.1, quantity=1)
# 修改订单有效期为 GTC
# order.time_in_force = 'GTC' 
# 下单
result = trade_client.place_order(order)

print(result)
>>> True
print(order)
>>> Order({'account': '111111', 'id': 2498911111111111111, 'order_id': None, 'parent_id': None, 'order_time': None, 'reason': None, 'trade_time': None, 'action': 'BUY', 'quantity': 1, 'filled': 0, 'avg_fill_price': 0, 'commission': None, 'realized_pnl': None, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': None, 'outside_rth': None, 'order_legs': None, 'algo_params': None, 'secret_key': None, 'contract': AAPL/STK/USD, 'status': 'NEW', 'remaining': 1})

# 若下单成功，则 order.id 为订单的id，此后可用该id查询订单或撤单
my_order = trade_client.get_order(id=order.id)
trade_client.cancel_order(id=order.id)
# 或操作order对象进行改单
trade_client.modify_order(order, limit_price=190.5)

```

**示例2**  
下单港股时，下单数量需要符合股票的每手股数。比如 `00700` 的每手股数为 100, 则下单股数只能为100， 200， 500等每手股数的整数倍。
可提前用 `QuoteClient.get_trade_metas` 获取股票的每手股数。 

```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)
quote_client = QuoteClient(client_config)

symbol = '00700'

# 获取每手股数
metas = quote_client.get_trade_metas([symbol])
lot_size = int(metas['lot_size'].iloc[0])

# 生成股票合约
contract = stock_contract(symbol=symbol, currency='HKD')
# 生成订单对象
order = limit_order(account=client_config.account, contract=contract, action='BUY', limit_price=400.0, quantity=2 * log_size)
# 下单
result = trade_client.place_order(order)
```

**示例3**
```python
from tigeropen.common.util.contract_utils import future_contract
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)

# 生成期货合约
contract = future_contract(symbol='CL', currency='USD')
# 生成订单对象
order = limit_order(account=client_config.account, contract=contract, action='BUY', limit_price=0.1, quantity=1)
# 下单
trade_client.place_order(order)

```

**示例4**
```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)

# 获取股票合约（不推荐）
contract = trade_client.get_contract('AAPL', sec_type=SecurityType.STK)
# 获取订单对象（不推荐）
order = trade_client.create_order(account=client_config.account, contract=contract, action='SELL', order_type='LMT', quantity=1, limit_price=200.0)
# 下单
trade_client.place_order(order)

```

**示例5**  
附加订单，算法订单
```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.common.util.order_utils import limit_order, limit_order_with_legs, order_leg, algo_order_params, \
    algo_order
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)

# 限价单 + 附加订单 (仅主订单为限价单时支持附加订单)
stop_loss_order_leg = order_leg('LOSS', 8.0, time_in_force='GTC')  # 附加止损
profit_taker_order_leg = order_leg('PROFIT', 12.0, time_in_force='GTC')  # 附加止盈
main_order = limit_order_with_legs(account, contract, 'BUY', 100, limit_price=10.0,
order_legs=[profit_taker_order_leg, stop_loss_order_leg])
trade_client.place_order(main_order)
# 查询主订单所关联的附加订单
order_legs = trade_client.get_open_orders(account, parent_id=main_order.id)
print(order_legs)
    
# 算法订单
contract = stock_contract(symbol='AAPL', currency='USD')
params = algo_order_params(start_time='2022-01-19 23:00:00', end_time='2022-11-19 23:50:00', no_take_liq=True,
                           allow_past_end_time=True, participation_rate=0.1)
order = algo_order(account, contract, 'BUY', 1000, 'VWAP', algo_params=params, limit_price=100.0)
trade_client.place_order(order)
```


### create_order 请求创建订单
`
TradeClient.create_order():
`

**说明**

请求订单号，创建订单对象。不推荐使用，建议使用 `tigeropen.common.util.order_utils` 下的工具函数本地创建订单，如 `limit_order`, `market_order`.


**参数**  

参数名|类型|描述  
----|----|----
account|str|账户id
contract|Contract|合约对象
action|str|买卖方向, 'BUY':买入, 'SELL':卖出
order_type|str|订单类型, 'MKT' 市价单 / 'LMT' 限价单 / 'STP' 止损单 / 'STP_LMT' 止损限价单 / 'TRAIL' 跟踪止损单
quantity|int|下单数量, 为大于0的整数
limit_price|float|限价单价格
aux_price|float|在止损单表示止损价格; 在跟踪止损单表示价差
trail_stop_price|float|跟踪止损单--触发止损单的价格
trailing_percent|float|跟踪止损单--百分比
percent_offset||
time_in_force|str|订单有效期， 'DAY'（当日有效）和'GTC'（取消前有效)
outside_rth|bool|是否允许盘前盘后交易(美股专属)
order_legs||附加订单列表
algo_params||算法订单参数

**示例**
```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')
trade_client = TradeClient(client_config)


contract = stock_contract(symbol='AAPL', currency='USD')
order = openapi_client.create_order(account, contract, 'BUY', 'LMT', 100, limit_price=5.0)

trade_client.place_order(order)

```
---