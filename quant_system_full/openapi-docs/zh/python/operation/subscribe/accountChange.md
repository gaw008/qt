---
title: 账户变动
---
**以下全部为异步API，需要指定一个方法响应返回的结果**


### 资产变化的订阅与取消 
**订阅方法**  

```PushClient.subscribe_asset(account=None) ```

**取消方法**

```PushClient.unsubscribe_asset()```

**参数**

参数名|类型|描述
----|----|----
account|str|需要订阅的 account id，不传则订阅所有关联的 account


**返回**  

需要使用 `PushClient.asset_changed` 响应返回结果。返回结果第一个参数为account，第二个参数为各字段元组构成的list。  
字段含义可对照参考获取资产接口
[get_prime_assets](/zh/python/operation/trade/accountInfo.html#get-prime-assets-获取综合-模拟账户资产信息)，
[get_assets](/zh/python/operation/trade/accountInfo.html#get-assets-获取账户资产信息)

详细字段解释参考对象:  
[PortfolioAccount 综合/模拟资产](/zh/python/appendix1/object.html#portfolioaccount-资产-综合-模拟账户)   
[Segment 综合/模拟分品种资产](/zh/python/appendix1/object.html#segment-分品种资产-综合-模拟账户)

[PortfolioAccount 环球资产](/zh/python/appendix1/object.html#portfolioaccount-资产-环球账户)  
[SecuritySegment 环球股票资产](/zh/python/appendix1/object.html#securitysegment-股票资产-环球账户)  
[CommoditySegment 环球期货资产](/zh/python/appendix1/object.html#commoditysegment-期货资产-环球账户)

字段|类型|描述
----|----|----
segment|str|按交易品种划分的分类。S表示股票，C表示期货，summary表示环球账户的汇总资产信息
cash|float|现金额。当前所有币种的现金余额之和
available_funds|float|可用资金，隔夜剩余流动性
excess_liquidity|float|当前剩余流动性
equity_with_loan|float|含贷款价值总权益。等于总资产 - 美股期权
net_liquidation|float|总资产(净清算值)。总资产就是我们账户的净清算现金余额和证券总市值之和
gross_position_value|float|证券总价值
initial_margin_requirement|float|初始保证金
maintenance_margin_requirement|float|维持保证金
buying_power|float|购买力。仅适用于股票品种，即segment为S时有意义

**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

# 定义回调方法
def on_asset_changed(account, items):
    """可进行自定义处理，此处仅打印"""
    print(f'asset change. account:{account}, items:{items}')

# 绑定回调方法
push_client.asset_changed = on_asset_changed
# 连接
push_client.connect(client_config.tiger_id, client_config.private_key)

# 订阅 
push_client.subscribe_asset(account=client_config.account)
#取消订阅资产变化
PushClient.unsubscribe_asset()
```

**回调数据示例**  
```
account:123456, 
items:[('cash', 679760.8392063834), ('gross_position_value', 202318.91581133104), ('equity_with_loan', 881789.7550177145),
 ('net_liquidation', 882079.7550177145), ('initial_margin_requirement', 108369.06714828224), ('buying_power', 3093682.751477729),
  ('excess_liquidity', 793970.6843320723), ('available_funds', 773420.6878694323), ('maintenance_margin_requirement', 87819.07068564222),
   ('segment', 'S')]
   
account:111111, 
items:[('cash', 99997.18), ('gross_position_value', 15620.0), ('equity_with_loan', 99572.18), ('net_liquidation', 99997.18),
 ('initial_margin_requirement', 467.5), ('buying_power', 0.0), ('excess_liquidity', 99572.18), ('available_funds', 99529.68),
  ('maintenance_margin_requirement', 425.0), ('segment', 'C')]
```

---

### 持仓变化的订阅与取消
**订阅方法**

`PushClient.subscribe_position(account=None)`

**取消方法**

`PushClient.unsubscribe_position()`

**参数**

参数名|类型|描述
----|----|----
account|str|需要订阅的 account id，不传则订阅所有关联的 account


**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

# 定义回调方法
def on_position_changed(account, items):
    """
    :param account:
    :param items:
    :return:
    items 数据示例:
        [('symbol', 'ABCD'), ('market_price', 3.68525), ('market_value', 0.0), ('sec_type', 'STK'),
        ('segment', 'summary'), ('currency', 'USD'), ('quantity', 0.0), ('average_cost', 3.884548)]
    """
    print(account, items)
    
# 绑定回调方法
push_client.position_changed = on_position_changed
# 连接
push_client.connect(client_config.tiger_id, client_config.private_key)

# 订阅持仓的变化
PushClient.subscribe_position(account=client_config.account)
# 取消订阅持仓的变化
PushClient.unsubscribe_position()
```


**返回**  
需要使用 `PushClient.position_changed` 响应返回结果，第一个参数为account，第二个参数为各字段元组构成的list。
列表项中的各字段可对照参考 [get_position](/zh/python/operation/trade/accountInfo.html#get-positions-获取持仓数据) 中的 [Position](/zh/python/appendix1/object.html#position-持仓) 对象。


字段|类型|描述
----|----|----
segment|str|按交易品种划分的分类。S表示股票，C表示期货
symbol|str|持仓标的代码，如 'AAPL', '00700', 'ES', 'CN'
identifier|str|标的标识符。股票的identifier与symbol相同。期货的会带有合约月份，如 'CN2201'
currency|str|币种。USD美元，HKD港币
sec_type|str|交易品种，标的类型。STK表示股票，FUT表示期货
market_price|float|标的当前价格
market_value|float|持仓市值
quantity|int|持仓数量
average_cost|float|持仓均价
unrealized_pnl|float|持仓盈亏


**回调数据示例**
```python
# 股票持仓变化推送
account:1111111, 
items:[('symbol', '09626'), ('currency', 'HKD'), ('sec_type', 'STK'), ('market_price', 305.0), ('quantity', 20), ('average_cost', 0.0), ('market_value', 6100.0), ('identifier', '09626'), ('unrealized_pnl', 6100.0), ('segment', 'S')]

# 期货持仓变化推送
account:1111111, 
items:[('symbol', 'CN'), ('currency', 'USD'), ('sec_type', 'FUT'), ('market_price', 15620.0), ('quantity', 1), ('average_cost', 0.0), ('market_value', 15620.0), ('identifier', 'CN2201'), ('unrealized_pnl', 15620.0), ('segment', 'C')]
```

---

### 订单变化的订阅和取消 
**订阅方法**

`PushClient.subscribe_order(account=client_config.account)`

**取消方法**

`PushClient.unsubscribe_order()`

**参数**

参数名|类型|描述
----|----|----
account|str|需要订阅的 account id，不传则订阅所有关联的 account

**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

# 定义回调方法
def on_order_changed(account, items):
    """
    :param account:
    :param items:
    :return:
    items 数据示例:
        [('order_type', 'LMT'), ('symbol', 'ABCD'), ('order_id', 1000101463), ('sec_type', 'STK'), ('filled', 100),
        ('quantity', 100), ('segment', 'summary'), ('action', 'BUY'), ('currency', 'USD'), ('id', 173612806463631360),
        ('order_time', 1568095814556), ('time_in_force', 'DAY'), ('identifier', 'ABCD'), ('limit_price', 113.7),
        ('outside_rth', True), ('avg_fill_price', 113.7), ('trade_time', 1568095815418),
        ('status', <OrderStatus.FILLED: 'Filled'>)]
    """
    print(account, items)
    
# 绑定回调方法
push_client.order_changed = on_order_changed
# 连接
push_client.connect(client_config.tiger_id, client_config.private_key)

# 订阅订单变化
PushClient.subscribe_order(account=client_config.account)
# 取消订阅订单变化
PushClient.unsubscribe_order()
```


**返回**  
需要使用 `PushClient.order_changed` 响应返回结果，第一个参数为account，第二个参数为各字段元组构成的list。  
返回结果中的字段含义可对照参考 [get_order](/zh/python/operation/trade/orderInfo.html#get-order-获取指定订单) 中的 [Order](/zh/python/appendix1/object.html#order-订单) 对象

字段|类型|描述
----|----|----
segment|str|按交易品种划分的分类。S表示股票，C表示期货
id|int|订单号
symbol|str|持仓标的代码，如 'AAPL', '00700', 'ES', 'CN'
identifier|str|标的标识符。股票的identifier与symbol相同。期货的会带有合约月份，如 'CN2201'
currency|str|币种。USD美元，HKD港币
sec_type|str|交易品种，标的类型。STK表示股票，FUT表示期货
action|str|买卖方向。BUY表示买入，SELL表示卖出。
order_type|str|订单类型。'MKT'市价单/'LMT'限价单/'STP'止损单/'STP_LMT'止损限价单/'TRAIL'跟踪止损单
quantity|int|下单数量
limit_price|float|限价单价格
filled|int|成交数量
avg_fill_price|float|成交均价
realized_pnl|float|已实现盈亏
status|tigeropen.common.consts.OrderStatus|[订单状态](/zh/python/appendix2/#订单状态)
outside_rth|bool|是否允许盘前盘后交易，仅适用于美股
order_time|int|下单时间


**回调数据示例**
```python
# 股票订单推送示例
account:111111, 
items: [('id', 25224910928347136), ('symbol', '09626'), ('currency', 'HKD'), ('sec_type', 'STK'), ('action', 'BUY'), ('quantity', 20), ('filled', 20), ('order_type', 'LMT'), ('avg_fill_price', 305.0), ('status', <OrderStatus.FILLED: 'Filled'>),  ('realized_pnl', 0.0), ('replace_status', 'NONE'), ('outside_rth', False), ('limit_price', 305.0), ('order_time', 1641349997000), ('identifier', '09626'), ('segment', 'S')]```
# 期货订单推送示例
account:111111, 
items: [('id', 25224890075841536), ('symbol', 'CN'), ('currency', 'USD'), ('sec_type', 'FUT'), ('action', 'BUY'), ('quantity', 1), ('filled', 1), ('order_type', 'LMT'), ('avg_fill_price', 15620.0), ('status', <OrderStatus.FILLED: 'Filled'>), ('realized_pnl', 0.0), ('replace_status', 'NONE'), ('outside_rth', False), ('limit_price', 15638.0), ('order_time', 1641349838000), ('identifier', 'CN2201'), ('segment', 'C')]
---