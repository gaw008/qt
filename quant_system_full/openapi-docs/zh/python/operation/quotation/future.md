---
title: 期货
---
### get\_future\_exchanges 获取期货交易所列表

```python
QuoteClient.get_future_exchanges(sec_type=SecurityType.FUT, lang=None)
```

**说明**

获取期货交易所列表

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参量名|类型|描述
----|----|----
sec_type|SecurityType| 使用 tigeropen.common.consts.SecurityType 下的枚举，默认 SecurityType.FUT，FUT表示期货，FOP表示期货期权。我们目前支持期货期权的交易，但暂不提供行情。

**返回**

`pandas.DataFrame`

结构如下：

COLUMN|类型|描述
----|----|----
code|str|交易所代码
name|str|交易所名称
zone|str|交易所所在时区

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

exchanges = quote_client.get_future_exchanges()
```

**返回示例**
```
      code      name              zone
0      CME  芝加哥商品交易所   America/Chicago
1    NYMEX   纽约商业交易所  America/New_York
2    COMEX   纽约商品交易所  America/New_York
3      SGX    新加坡交易所         Singapore
4     HKEX     香港交易所    Asia/Hong_Kong
5     CBOT  芝加哥期货交易所   America/Chicago
6  CBOEXBT  芝加哥期权交易所   America/Chicago
7   CMEBTC  芝加哥期货交易所   America/Chicago
8      OSE     大阪交易所        Asia/Tokyo
9     CBOE  芝加哥期權交易所   America/Chicago

```

---


### get\_future\_contracts 获取交易所下的可交易合约

```python
QuoteClient.get_future_contracts(exchange, lang=None)
```

**说明**

获取交易所下的可交易合约

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
exchange|str|交易所代码，如 'CBOE'

**返回**

`pandas.DataFrame`

结构如下：

COLUMN|类型|描述
----|----|----
contract_code|str|合约代码，如 VIX2208
type|str|期货合约对应的交易品种，如 CL
symbol|str|对应的ib代码
name|str|期货合约的名称
contract_month|str|合约交割月份，如 202208，表示2022年8月
currency|str|交易的货币
first_notice_date|str|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
last_bidding_close_time|str|竞价截止时间
last_trading_date|str|最后交易日
trade|bool|是否可交易
continuous|bool|是否为连续合约
multiplier|float|合约乘数
min_tick|float|最小变动单位

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

contracts = quote_client.get_future_contracts('CME')
```

**返回示例**

```
  contract_code symbol  type         name contract_month  multiplier exchange  \
0       MXP1912    MXP   MXP       比索1912         201912    500000.0   GLOBEX   
1       MXP1909    MXP   MXP       比索1909         201909    500000.0   GLOBEX   
2      MEUR1909    M6E  MEUR      微欧元1909         201909     12500.0   GLOBEX   
3        ES1909     ES    ES  SP500指数1909         201909        50.0   GLOBEX   
4        ES2003     ES    ES  SP500指数2003         202003        50.0   GLOBEX   

  currency first_notice_date last_bidding_close_time last_trading_date  trade  \
0      USD              None                    None          20191216   True   
1      USD              None                    None          20190916   True   
2      USD              None                    None          20190916   True   
3      USD              None                    None          20190920   True   
4      USD              None                    None          20200320   True   

   continuous  
0       False  
1       False  
2       False  
3       False  
4       False  


```

---

### get\_current\_future\_contract 查询指定品种的当前合约

```python
QuoteClient.get_current_future_contract(future_type, lang=None)
```

**说明**

查询指定品种的当前合约，即合约主连

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
future_type||期货合约对应的交易品种，如 CL

**返回**

`pandas.DataFrame`

结构如下：

COLUMN|类型|描述
----|----|----
contract_code||合约代码，如 CL2112
type|str|期货合约对应的交易品种， 如 CL
name|str|期货合约的名称
contract_month|str|合约交割月分，如 202112，表示2021年12月交割
currency|str|交易的货币
first_notice_date|str|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
last_bidding_close_time|str|竞价截止时间
last_trading_date|str|最后交易日
trade|bool|是否可交易
continuous|bool|是否为连续合约
multiplier|float|合约乘数
min_tick|float|最小变动单位

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

contracts = quote_client.get_current_future_contract('ES')
```

**返回示例**

```
  contract_code type         name contract_month currency first_notice_date  \
0        ES1903   ES  SP500指数1903         201903      USD          20190315   

   last_bidding_close_time last_trading_date  trade  continuous  
0                        0          20190315   True       False  

```

### get\_future\_trading\_times 查询指定合约的交易时间


```python
get_future_trading_times(contract_code, trading_date=None)
```

**说明**

询指定合约的交易时间

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
contract_code|str| 合约代码，如CL1901
trading_date|int|指定交易日的毫秒单位时间戳，如 1643346000000

**返回**

`pandas.DataFrame`

结构如下：

COLUMN|类型|描述
----|----|----
start|int|交易开始时间
end|int|交易结束时间
trading|bool|是否为连续交易
bidding|bool|是否为竞价交易
zone|str|时区


**示例**

```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

times = quote_client.get_future_trading_times('CN1901', trading_date=1545049282852)
# 转换 start、end 的格式
times['zone_start'] = pd.to_datetime(times['start'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(times['zone'][0])
times['zone_end'] = pd.to_datetime(times['end'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(times['zone'][0])

```

**返回示例**
```
           start            end  trading  bidding       zone  
0  1545036600000  1545037200000    False     True  Singapore   
1  1545093900000  1545094800000    False     True  Singapore   
2  1545121800000  1545122100000    False     True  Singapore   
3  1545037200000  1545079500000     True    False  Singapore   
4  1545094800000  1545121800000     True    False  Singapore   

```

<!--## get_future_bars 获取期货K线 

```python
QuoteClient.get_future_bars(symbol, 
                            period=BarPeriod.DAY, 
                            begin_time=-1, 
                            end_time=-1, 
                            limit=1000)
```

**说明**
获取期货K线

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

- identifiers: 期货代码列表
- period: 获取K线的周期， 使用 BarPeriod 下的枚举常量
- begin_time: 开始时间，毫秒时间戳
- end_time: 截至时间， 毫秒时间戳
- limit: 限制条数

**返回**

pandas.DataFrame, 各column 含义如下：

- identifier: 期货合约代码
- time: Bar对应的时间戳, 即Bar的结束时间。Bar的切割方式与交易所一致，以CN1901举例，T日的17:00至T+1日的16:30的数据会被合成一个日级Bar。
- latest_time: Bar 最后的更新时间
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- settlement: 结算价，在未生成结算价时返回0
- volume: 成交量
- open_interest: 未平仓合约数量

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

bars = openapi_client.get_future_bars(['CN1901', 'MXP1903'], 
                                          begin_time=-1, 
                                          end_time=1545105097358)
print(bars.head())

  identifier           time    latest_time     open     high     low    close  \
0     CN1901  1545035400000  1545035700002  11022.5  11097.5  109350  10940.0   
1     CN1901  1544776200000  1544776168000  11182.5  11197.5  110025  11030.0   
2     CN1901  1544689800000  1544689765000  11012.5  11252.5  110125  11212.5   
3     CN1901  1544603400000  1544603321185  10940.0  11070.0  109400  11035.0   
4     CN1901  1544517000000  1544516945000  10895.0  10962.5  108150  10942.5   

   settlement  volume  open_interest  
0     10935.0    3872          15148  
1     11042.5    1379          14895  
2     11202.5    4586          12870  
3     11032.5    3514          12237  
4     10927.5    2378          10575  

```


---

## get_future_trade_ticks 获取期货逐笔成交 

```python
QuoteClient.get_future_trade_ticks(identifiers, 
                                  begin_index=0, 
                                  end_index=30, 
                                  limit=1000)
```

**说明**
获取期货逐笔成交 

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

- identifiers: 证券代码列表
- begin_index: 起始索引
- end_index: 结束索引
- limit: 返回条数限制


**返回**

pandas.DataFrame, 各 column 含义如下

- index: 索引值
- time: 成交时间，精确到毫秒的时间戳
- price: 成交价格
- volume: 成交量

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

ticks = quote_client.get_future_trade_ticks('CN1901')
print(ticks)

   identifier  index           time        price  volume
0      CN1901      0  1547456400000  10607.50000       5
1      CN1901      1  1547456402000  10605.00000       6
2      CN1901      2  1547456407000  10602.50000       4
3      CN1901      3  1547456407000  10605.00000       2
4      CN1901      4  1547456424000  10602.50000       5

```

## get_future_brief 获取期货最新行情 

```python
QuoteClient.get_future_brief(identifiers)
```

**说明**
获取期货最新行情，包括盘口数据，最新成交数据等

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

- identifiers: 期货代码列表，如 ['CL2201']

**返回**

pandas.DataFrame，各 column 含义如下

- identifier: 期货代码
- ask_price: 卖价
- ask_size: 卖量
- bid_price: 买价
- bid_size: 买量
- pre_close: 前收价
- latest_price: 最新价
- latest_size: 最新成交量
- latest_time: 最新价成交时间
- volume:  当日累计成交手数
- open_interest: 未平仓合约数量
- open: 开盘价
- high: 最高价
- low: 最低价
- limit_up: 涨停价
- limit_down: 跌停价

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

briefs = quote_client.get_future_brief(['CN1901', 'MXP1903'])
print(briefs)

  identifier    ask_price  ask_size    bid_price  bid_size pre_close  \
0     CN1901  10677.50000        13  10675.00000       145      None   
1    MXP1903      0.05219         9      0.05218         3      None   

   latest_price  latest_size    latest_time  volume  open_interest  \
0   10677.50000            2  1547519736000  111642         863308   
1       0.05219            4  1547519636000    1706         190126   

          open         high          low    limit_up  limit_down  
0  10607.50000  10707.50000  10572.50000  11670.0000   9550.0000  
1      0.05223      0.05223      0.05216      0.0562      0.0482  

```
--->
