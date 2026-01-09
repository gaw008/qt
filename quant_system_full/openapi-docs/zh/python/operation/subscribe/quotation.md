---
title: 行情订阅
---
### 订阅行情

`PushClient.subscribe_quote(symbols, quote_key_type=QuoteKeyType.TRADE, focus_keys=None)`

**取消方法**

`PushClient.unsubscribe_quote(symbols)`

**说明**

股票行情的订阅与取消接口，返回的数据为实时更新，即每次价格或挂单数据更新就会有数据推送

本接口为异步返回， 需要使用`PushClient.on_quote_changed`响应返回结果

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，如 ['AAPL', 'BABA']，英文代码应使用大写
quote_key_type|QuoteKeyType|tigeropen.common.consts.quote_keys.QuoteKeyType, 订阅的行情key类型
focus_keys|list\<str...\>|订阅的行情key的列表，如指定了quote_key_type参数，此参数可忽略（建议）

其中`tigeropen.common.consts.quote_keys.QuoteKeyType`属性如下：
属性|说明
----|----
QuoteKeyType.TRADE|成交数据，包含的字段： open, latest_price, high, low, prev_close, volume, timestamp
QuoteKeyType.QUOTE|报价数据，包含的字段: ask_price, ask_size, bid_price, bid_size, timestamp
QuoteKeyType.TIMELINE|分时数据，包含的字段: p: 最新价； a:当日截至当前的平均价; t:所属分钟的时间戳， v: 成交量。
QuoteKeyType.ALL|会同时推送上面几类数据

**取消方法参数**
参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，如 ['AAPL', 'BABA']

**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

# 初始化PushClient
protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

#定义回调方法
def on_quote_changed(symbol, items, hour_trading):
  """
  行情推送回调
  :param symbol: 订阅的证券代码
  :param items: list，每个元素是一个tuple，对应订阅的字段名称和值
  :param hour_trading: 是否为盘前盘后的交易
  :return:
  """
  print(f'quote change: {symbol}, {items}, {hour_trading}')

#绑定回调方法
push_client.position_changed = on_position_changed   
  
# 建立连接        
push_client.connect(client_config.tiger_id, client_config.private_key)

# 订阅指定标的行情. 可指定关注的key, 不指定则默认 QuoteKeyType.Trade. 参见 tigeropen.common.consts.quote_keys.QuoteKeyType
push_client.subscribe_quote(['AAPL', 'BABA'], quote_key_type=QuoteKeyType.ALL)


# 取消订阅
push_client.unsubscribe_quote(['AAPL', 'BABA'])
# 断开连接
push_client.disconnect()
```

**回调数据**

需使用回调函数`on_quote_changed(symbol, items, hour_trading)` 接收回调数据，其中:
- symbol 为证券代码，`str`类型
- items为返回行情数据的列表，`list`类型，每个元素为一个`tuple`，对应订阅的字段名称和值
- hour_trading表示是否盘前盘后交易，`bool`

**回调数据示例**

items 数据示例：
```python
[('latest_price', 339.8), ('ask_size', 42500), ('ask_price', 340.0), ('bid_size', 1400), ('bid_price', 339.8),
  ('high', 345.0), ('prev_close', 342.4), ('low', 339.2), ('open', 344.0), ('volume', 7361440),
  ('minute', {'p': 339.8, 'a': 341.084, 't': 1568098440000, 'v': 7000, 'h': 340.0, 'l': 339.8}),
  ('timestamp', '1568098469463')]
```

### 订阅深度行情

`Push_client.subscribe_depth_quote(symbols)`

**取消方法**

`PushClient.unsubscribe_depth_quote(symbols)`

**说明**

订阅深度行情，返回最高40档的买卖盘挂单数据, 返回的数据为实时更新，即挂单数据更新就会有数据推送

**参数**
参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，如 ['AAPL', 'BABA']

**取消方法参数**
参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，如 ['AAPL', 'BABA']

**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

# 初始化PushClient
protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

#定义回调方法
def on_quote_changed(symbol, items, hour_trading):
  """
  行情推送回调
  :param symbol: 订阅的证券代码
  :param items: list，每个元素是一个tuple，对应订阅的字段名称和值
  :param hour_trading: 是否为盘前盘后的交易
  :return:
  """
  print(f'quote change: {symbol}, {items}, {hour_trading}')

#绑定回调方法
push_client.position_changed = on_position_changed   
  
# 建立连接        
push_client.connect(client_config.tiger_id, client_config.private_key)

# 订阅深度行情
push_client.subscribe_depth_quote(['AMD', 'MSFT'])


# 取消订阅
push_client.unsubscribe_depth_quote(['AMD', 'MSFT'])
# 断开连接
push_client.disconnect()
```

**回调数据**

需订阅回调函数on_quote_changed(symbol, items, hour_trading) 接收回调数据，其中:
- symbol 为证券代码，`str`类型
- items为返回行情数据的列表，`list`类型，由 3 个等长`list`构成，分别代表：price(价格)，volume(数量)，count(单量)
- hour_trading表示是否盘前盘后交易，`bool`

注意：此接口最多只会推送买卖前 40 档数据

**回调数据示例**

深度行情 items 数据示例，相邻档位价格可能一样， 其中 count 是可选的
```python
[('bid_depth',
    '[[127.87,127.86,127.86,127.86,127.85,127.85,127.85,127.84,127.84,127.84,127.84,127.83,127.83, 127.83,127.83,127.83,127.82,127.81,127.8,127.8,127.8,127.8,127.8,127.8,127.8,127.79,127.79,127.78, 127.78, 127.75,127.68,127.6,127.6,127.55,127.5,127.5,127.5,127.5,127.29,127.28],
      [69,2,5,20,1,1,1,18,1,70,80,40,2,330,330,1,40,80,20,10,131,2,30,50,300,1,38,1,1,15,6,20,1,3,100,15,25,30,49,43]
      ]'),
('ask_depth',
    '[[127.91,127.94,127.95,127.95,127.95,127.95,127.95,127.96,127.98,127.98,127.98,127.98,127.99,127.99, 128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0, 128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0,128.0],
      [822,4,98,50,5,5,500,642,300,40,1,36,19,1,1,1,1,50,1,1,50,1,100,10,1,1,10,1,1,1,1,5,1,8,1,1,120,70,1,4]
    ]'),
('timestamp', 1621933454191)]
```

### 查询已订阅的标的列表

`PushClient.query_subscribed_quote()`

**说明**

查询已订阅的标的列表

本接口为异步返回， 需要使用`PushClient.subscribed_symbols`响应返回结果

**参数**

无

**回调数据**

需要使用`PushClient.subscribed_symbols`响应返回结果

返回结果为`tuple`类型， 共四个元素：
- 第一个元素是订阅合约的列表，`list`类型
- 第二个元素是每个合约订阅的 focus_key 信息，`dict`类型
- 第三个元素是当前tigerid可以订阅的合约数量，`int`类型
- 第四个元素是目前已订阅的合约数量，`int`类型

**回调数据示例**

```python
  (['00968', 'ES1906', '00700', '01810'], 
  {'00968': ['volume', 'bidPrice'], 
  '00700': ['volume', 'bidPrice'], 
  '01810': ['volume', 'bidPrice']
  },
  100, 
  4)
```

### 订阅期权行情
`push_client.subscribe_option(['AAPL 20230120 150.0 CALL', 'SPY 20220930 470.0 PUT'])`

或

`push_client.subscribe_quote(['AAPL 20230120 150.0 CALL', 'SPY 20220930 470.0 PUT'])`

**说明**

期货行情的订阅接口

**参数**
参数名|类型|描述
----|----|----
symbols|list\<str...\>|由期权四要素组成，分别是：标的代码，过期日(YYYYMMDD)，行权价，期权类型

**回调数据**

需通过`push_client.quote_changed`绑定回调方法，回调方法同股票
