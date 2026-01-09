---
title: 基本功能示例
---
老虎Open API SDK提供了便捷地在自己的程序中调用老虎的服务的工具，本章节将对老虎API的核心功能进行一一演示：包括查询行情，订阅行情，以及调用API进行交易

## 查询行情
以下为一个最简单的调用老虎API的示例，演示了如何调用Open API来主动查询股票行情。接下来的例子分别演示了如何调用Open API来进行交易与订阅行情。

除上述基础功能外，Open API还支持查询、交易多个市场的不同标的，以及其他复杂请求。对于其他Open API支持的接口和请求，请在快速入门后阅读文档正文获取列表及使用方法，并参考快速入门以及文档中的例子进行调用

*为方便直接复制运行，以下的说明采用注释的形式*
````python
from tigeropen.common.consts import (Language,        # 语言
                                Market,           # 市场
                                BarPeriod,        # k线周期
                                QuoteRight)       # 复权类型
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.quote.quote_client import QuoteClient

# 查询行情的操作通过QuoteClient对象的成员方法实现，所以调用相关行情接口之前需要先初始化QuoteClient，具体代码如下：
# 首先通过自定义的函数生成配置文件, 函数get_client_config会返回一个包含初始化行情对象所需要的用户信息的ClientConfig对象，用来传入行情对象QuoteClient构造函数中，以进行QuoteClient的初始化。也可选择使用tigeropen.tiger_open_config.get_client_config()函数生成用户配置对象
def get_client_config(sandbox=False):
    """
    https://www.itiger.com/openapi/info 开发者信息获取
    """
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config.private_key = read_private_key('填写私钥PEM文件的路径')
    client_config.tiger_id = '替换为tigerid'
    client_config.account = '替换为账户，建议使用模拟账户'
    client_config.language = Language.zh_CN  #可选，不填默认为英语'
    return client_config

# 调用上方定义的函数生成用户配置ClientConfig对象
client_config = get_client_config()

# 随后传入配置参数对象来初始化QuoteClient
quote_client = QuoteClient(client_config)

# 完成初始化后，就可以调用quote_client方法来使用调用QuoteClient对象的get_stock_brief方法来查询股票行情了，此处以美国股票为例，关于其他支持的市场及标的类型，请参考文档的基本操作部分。
# 对于使用多台设备调用API的用户，需先调用grab_quote_permission进行行情权限的抢占，详情请见基本操作-行情类-通用-grab_quote_permission方法说明
permissions = quote_client.grab_quote_permission() 

#输出list类型的行情权限权限列表
print(permissions)

# 调用API查询股票行情
stock_price = quote_client.get_stock_briefs(['00700'])

# 查询行情函数会返回一个包含当前行情快照的pandas.DataFrame对象，见返回示例。具体字段含义参见get_stock_briefs方法说明
print(stock_price)
````

**返回示例**
```
  symbol  ask_price  ask_size  bid_price  bid_size  pre_close  latest_price  \
0  00700      326.4     15300      326.2     26100     321.80         326.4   

     latest_time    volume    open    high     low  status  
0  1547516984730   2593802  325.00  326.80  323.20  NORMAL 
``` 
    
## 订阅行情

除了选择主动查询的方式（见[快速入门-查询行情](/zh/python/quickStart/basicFunction.html#%E6%9F%A5%E8%AF%A2%E8%A1%8C%E6%83%85)部分），Open API还支持订阅-接受推送的方式来接收行情等信息，具体请见下例。此示例实现了订阅苹果与AMD股票行情，将行情快照输出在console，持续30秒后取消订阅，并且断开与服务器的连接的过程。

需要注意的是，订阅推送相关的请求均为异步处理，故需要用户自定义回调函数，与中间函数进行绑定。某个事件发生，或有最新信息更新被服务器推送时，程序会自动调用用户自定义的回调函数并传入返回接口返回的数据，由用户自定义的回调函数来处理数据。

```python
import time
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import TigerOpenClientConfig

def get_client_config(sandbox=False):
    """
    https://www.itiger.com/openapi/info 开发者信息获取
    """
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config.private_key = read_private_key('填写私钥PEM文件的路径')
    client_config.tiger_id = '替换为tigerid'
    client_config.account = '替换为账户，建议使用模拟账户' 
    return client_config

#首先定义回调函数，本例中为简单起见，仅使用print输出传入的数据
def on_quote_changed(symbol, items, hour_trading):
    """
    行情推送回调，传入的参数（API返回的数据）为：
    symbol: 订阅的证券代码
    items: list，list的每个元素是一个tuple，对应订阅的字段名称和值
    hour_trading: 是否为盘前盘后的交易
    """
    print(symbol, items, hour_trading)

#定义订阅成功与否的回调函数
def subscribe_callback(destination, content):
    """
    订阅成功与否的回调, 传入的参数（API返回的数据）为:
    destination: 订阅的类型. 有 quote, trade/asset, trade/position, trade/order
    content: 回调信息. 如成功 {'code': 0, 'message': 'success'}; 若失败则 code 不为0, message 为错误详情
    """
    print('subscribe:{}, callback content:{}'.format(destination, content))

#定义取消订阅成功事件的回调函数
def unsubscribe_callback(destination, content):
    """
    退订成功与否的回调，传入的参数为:
    destination: 取消订阅的类型. 有 quote, trade/asset, trade/position, trade/order
    content: 回调信息
    """
    print('subscribe:{}, callback content:{}'.format(destination, content))

#定义连接建立事件的回调函数
def connect_callback():
    """连接建立回调"""
    print('connected')

def disconnect_callback():
    """连接断开回调. 此处利用回调进行重连"""
    for t in range(1, 20):
        try:
            print('disconnected, reconnecting')
            push_client.connect(client_config.tiger_id, client_config.private_key)
        except:
            print('connect failed, retry')
            time.sleep(t)
        else:
            print('reconnect success')
            return
    print('reconnect failed, please check your network')

if __name__ == "__main__":
    #首先通过工具函数生成配置文件
    client_config = get_client_config()

    #初始化PushClient
    protocol, host, port = client_config.socket_host_port
    push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))

    #绑定回调函数，推送类方法均为异步响应，需要绑定回调函数处理数据，下面绑定行情变动回调函数
    push_client.quote_changed = on_quote_changed
    # 绑定订阅成功与否的回调
    push_client.subscribe_callback = subscribe_callback
    # 退订成功与否的回调
    push_client.unsubscribe_callback = unsubscribe_callback
    # 断线重连回调
    push_client.disconnect_callback = disconnect_callback

    # 建立连接
    push_client.connect(client_config.tiger_id, client_config.private_key)
    # 订阅行情推送，此处以苹果与AMD为例
    push_client.subscribe_quote(['AAPL', 'AMD'])

    # 等待推送
    time.sleep(30)

    #取消订阅
    push_client.unsubscribe_quote()
    #断开链接
    push_client.disconnect()
````


## 交易下单

交易是Open API的另一个主要功能。此例展示了如何使用Open API对美股老虎证券TIGR下市价单:

````python
from tigeropen.common.consts import (Language,        # 语言
                                    Market,           # 市场
                                    BarPeriod,        # k线周期
                                    QuoteRight)       # 复权类型
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.common.util.signature_utils import read_private_key
from tigeropen.trade.trade_client import TradeClient

# 查询账户与交易相关的操作通过TradeClient对象的成员的方法实现，所以调用相关行情接口之前需要先初始化TradeClient，具体代码如下：
# 首先通过自定义的函数生成配置文件, 函数get_client_config会返回一个包含初始化行情对象所需要的用户信息的ClientConfig对象，用来传入行情对象TradeClient构造函数中，以进行TradeClient的初始化。也可选择使用tigeropen.tiger_open_config.get_client_config()函数生成用户配置对象
def get_client_config(sandbox=False):
    """
    https://www.itiger.com/openapi/info 开发者信息获取
    """
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config = TigerOpenClientConfig(sandbox_debug=sandbox)
    client_config.private_key = read_private_key('填写私钥PEM文件的路径')
    client_config.tiger_id = '替换为tigerid'
    client_config.account = '替换为账户，建议使用模拟账户'
    client_config.language = Language.zh_CN #可选，不填默认为英语
    return client_config

# 调用上方定义的函数生成用户配置ClientConfig对象
client_config = get_client_config()

# 随后传入配置参数对象来初始化TradeClient
trade_client = TradeClient(client_config)

from tigeropen.common.consts import Market, SecurityType, Currency
from tigeropen.common.util.contract_utils import stock_contract

#下单需要先初始化一个contract对象，contract对象中保存着合约信息，详情请见合约对象。创建contract对象的方法请参考文档 基本操作-交易类-获取合约 部分，示例如下：
#方法1: 直接本地构造contract对象。 期货 contract 的构造方法请参考文档 基本操作-交易类-获取合约 部分
contract = stock_contract(symbol='TIGR', currency='USD')

#方法2: 联网方式获取Contract对象，此方法仅针对股票
stock_contract = trade_client.get_contracts(symbol='SPY')[0]

#以下为目前支持的几种订单类型对象
from tigeropen.common.util.order_utils import (market_order,        # 市价单
                                                limit_order,         # 限价单
                                                stop_order,          # 止损单
                                                stop_limit_order,    # 限价止损单
                                                trail_order,         # 移动止损单
                                                order_leg)           # 附加订单

#创建订单对象，订单对象中保存了下单所需的账户、目标合约等信息，详情请见订单对象。这里以限价单为例
stock_order = market_order(account=client_config.account,            # 下单账户，可以使用标准、环球、或模拟账户
                            contract = stock_contract,                # 第1步中获取的合约对象
                            action = 'BUY',
                            quantity = 100)

#提交订单。注意：提交订单前，order对象的id为None, 提交成功后， order对象的id会变为全局订单id
trade_client.place_order(stock_order)

print(stock_order)
````
**返回示例**
```
Order({'account': '164644', 'id': 14275856193552384, 'order_id': None, 'parent_id': None, 'order_time': None, 'reason': None, 'trade_time': None, 'action': 'BUY', 'quantity': 100, 'filled': 0, 'avg_fill_price': 0, 'commission': None, 'realized_pnl': None, 'trail_stop_price': None, 'limit_price': 100, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': None, 'outside_rth': None, 'contract': SPY/STK/USD, 'status': 'NEW', 'remaining': 100})
```

**补充说明**

对于支持的其他类型的订单，请参考文档 [对象-Order对象](/zh/python/appendix1/object.html#order-订单) 中的说明