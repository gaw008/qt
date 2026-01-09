---
title: 证券类
---
### get\_market\_status 获取市场状态

`QuoteClient.get_market_status(market=Market.ALL, lang=None)`

**说明**

获取被查询市场的状态（开盘，盘前交易，盘后等），并获取被查询市场的最近开盘时间

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
market|Market|查询的市场，可以使用 tigeropen.common.consts.Market 下提供的枚举常量，如 Market.US，参见枚举参数部分
lang|Language|支持的语言，可以使用 tigeropen.common.consts.Language 下提供的枚举常量，如 Language.zh_CN，参见枚举参数部分

**返回**

`list`

元素为 [MarketStatus](/zh/python/appendix1/object.html#marketstatus-市场状态) 对象，MarketStatus结构如下：

参数名|类型|描述
----|----|----
market|str|市场名称
trading_status|str|交易状态码。未开盘 NOT_YET_OPEN; 盘前交易 PRE_HOUR_TRADING; 交易中 TRADING; 午间休市 MIDDLE_CLOSE; 盘后交易 POST_HOUR_TRADING; 已收盘 CLOSING; 提前休市 EARLY_CLOSED; 休市 MARKET_CLOSED;
status|str|交易状态描述
open_time|datetime|最近的开盘、交易时间，带tzinfo


**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

market_status_list = quote_client.get_market_status(Market.US)
```

**返回示例**

```python
[MarketStatus({'market': 'US', 'status': '盘前交易', 'open_time': datetime.datetime(2019, 1, 7, 9, 30, tzinfo=<DstTzInfo 'US/Eastern' EST-1 day, 19:00:00 STD>), 'trading_status': 'PRE_HOUR_TRADING'})]
```

---

### get_symbols 获取所有证券代码列表

`QuoteClient.get_symbols(market=Market.ALL)`

**说明**

获取所选市场所有证券的代码列表，包含退市和不可交易的部分代码以及指数。

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
market|Market|查询的市场， 可以使用tigeropen.common.consts.Market下提供的枚举常量，如Market.US

**返回**

类型

`list`

元素为市场所有证券的symbol，包含退市和不可交易的部分代码。其中以`.`开头的代码为指数， 如 `.DJI` 表示道琼斯指数。

**示例**
```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

symbols = quote_client.get_symbols(market.US)
print(symbols)
```

**返回示例**
```python
['.DJI', '.IXIC', '.SPX', 'A', 'AA', 'AAA', 'AAAU', 'AAC', 'AAC.U', 'AAC.WS', 'AACG', 'AACI', 'AACIU', 'AACIW', 'AACQW', 'AADI', 'AADR', 'AAIC', 'AAIN', 'AAL', 'AAMC', 'AAME', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL',....,'ZYME', 'ZYNE', 'ZYXI']
```
---

### get\_symbol\_names 获取代码及名称列表

`QuoteClient.get_symbol_names(market=Market.ALL)`

**说明**

获取所选市场所有证券的代码及名称

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
market|Market|查询的市场，可以使用tigeropen.common.consts.Market下提供的枚举常量，如 Market.US

**返回**

`list`

结构如下：

list的每个对象是一个tuple，tuple的第一个元素是symbol，第二个是name。

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

symbol_names = quote_client.get_symbol_names(market=Market.ALL)
```

**返回示例**
```python
[('AAAP', 'Advanced Accelerator Applications SA'), ('AAAU', 'Perth Mint Physical Gold ETF'), ('AABA', 'Altaba'), ('AAC', 'AAC Holdings Inc')]
```

---

### get\_timeline 获取最新一日的分时数据

`QuoteClient.get_timeline(symbols, include_hour_trading=False, begin_time=-1, lang=None)`

**说明**

获取最新一日的分时数据，分时数据与分钟bar类似，每分钟产生一个。只支持查询最新一个交易日的数据，如果需要历史数据，请使用 `get_bars` 接口查询

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，单次上限50只，如 ['AAPL', 'TSLA']
include_hour_trading|bool|是否包含盘前盘后分时数据，可选填，如 True 或 False
begin_time|str|获取分时数据的起始时间, 支持毫秒级别的时间戳，或日期时间字符串。如 1639386000000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
lang|Language|支持的语言，可以使用tigeropen.common.consts.Language下提供的枚举常量

**返回**

`pandas.DataFrame`

结构如下：

COLUMN|类型|描述
----|----|----
symbol|str|证券代码，如 AAPL
time|int|精确到毫秒的时间戳，如 1639386000000
price|float|当前分钟的收盘价
avg_price|float|截至到当前时间的成交量加权均价
pre_close|float|昨日收盘价
volume|int|这一分钟的成交量
trading_session|str|字符串， "pre_market" 表示盘前交易, "regular" 表示盘中交易, "after_hours"表示盘后交易。


**示例**

```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

timeline = quote_client.get_timeline(['01810'], include_hour_trading=False)

# 将 time 转换为对应时区的日期时间
timeline['cn_date'] = pd.to_datetime(timeline['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
timeline['us_date'] = pd.to_datetime(timeline['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```
**返回示例**

```python
symbol           time    price  avg_price  pre_close  volume trading_session
0     01810  1547217000000  23.4700  23.211563       23.4  233000         regular
1     01810  1547217060000  23.6700  23.408620       23.4  339296         regular
2     01810  1547217120000  23.5900  23.423038       23.4   46337         regular
3     01810  1547217180000  23.5000  23.428830       23.4   66697         regular
4     01810  1547217240000  23.5108  23.433360       23.4   46762         regular
```
---

---

### get\_trade\_ticks 获取逐笔成交数据

`
QuoteClient.get_trade_ticks(symbols, 
                            begin_index=0, 
                            end_index=30, 
                            limit=30, 
                            lang=None)
`

**说明**

获取股票的逐笔成交数据

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，最多50个
begin_index|int|开始索引
end_index|int| 结束索引
limit|int|返回条数限制
lang|Language|支持的语言， 可以使用 tigeropen.common.consts.Language 下提供的枚举常量

**返回**

`pandas.DataFrame`

结构如下：

参数名|类型|描述
----|----|----
index|int|索引值       
time|int|毫秒时间戳
price|float|成交价
volume|int|成交量
direction|str|价格变动方向，"-"表示向下变动， "+" 表示向上变动

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

ticks = quote_client.get_trade_ticks(['00700'])
```
**返回示例**

```python
  index           time  price  volume direction
0   17929  1547005726394  323.2     700         +
1   17930  1547005726980  323.0     200         -
2   17931  1547005727554  323.0     100         -
3   17932  1547005727554  323.0     200         -
4   17933  1547005727587  323.0     100         -

```

---

### get\_stock\_briefs 获取股票实时行情

`QuoteClient.get_stock_briefs(symbols, lang=None)`

**说明**

获取股票的实时行情，使用前需购买相应的行情权限，单次请求限制50只股票

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols| list\<str...\> |证券代码的列表，最多50只，如 ['AAPL', 'MSFT']
lang| Language |支持的语言，可以使用tigeropen.common.consts.Language下提供的枚举常量

**返回**

`pandas.DataFrame`

结构如下： 

COLUMN|类型|描述
----|----|----
symbol|str|证券代码
ask_price|float|卖一价
ask_size|int|卖一量
bid_price|float|买一价
bid_size|int|买一量
pre_close|float|前收价
latest_price|float|最新价
latest_time|int|最新成交时间，毫秒单位数字时间戳
volume|int|成交量
open|float|开盘价
high|float|最高价
low|float|最低价
status|str|交易状态

status(交易状态) 取值:
- "UNKNOWN": 未知
- "NORMAL": 正常
- "HALTED": 停牌
- "DELIST": 退市
- "NEW": 新股
- "ALTER": 变更

**示例**

```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

briefs = quote_client.get_stock_briefs(['00700'])

# 将 latest_time 转换为对应时区的日期时间
briefs['cn_date'] = pd.to_datetime(briefs['latest_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
briefs['us_date'] = pd.to_datetime(briefs['latest_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**

```
symbol  ask_price  ask_size  bid_price  bid_size  pre_close  latest_price  \
0  00700      326.4     15300      326.2     26100     321.80         326.4   

  latest_time    volume    open    high     low  status  
0  1547516984730   2593802  325.00  326.80  323.20  NORMAL  
```


---

### get\_stock\_delay\_briefs 获取股票延迟行情
`QuoteClient.get_stock_delay_briefs(symbols, lang=None)`

**说明**

本接口为免费延迟行情接口。不需要购买行情权限，开通开发者账号后可直接请求使用。目前仅支持获取美股延迟行情，相对实时行情延迟15分钟

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码的列表，目前仅支持获取美股延迟行情。 如 ['AAPL', 'MSFT']
lang|Language|支持的语言， 可以使用tigeropen.common.consts.Language下提供的枚举常量， 如 Language.en_US

**返回**

`pandas.DataFrame`

结构如下：

参数名|类型|描述
----|----|----
symbol|str|证券代码
pre_close|float|前收价
time|int|最近成交时间，毫秒为单位的数字时间戳，如 1639429200000
volume|int|成交量
open|float|开盘价
high|float|最高价
low|float|最低价
close|float|收盘价  
halted|float|标的状态 (0: 正常 3: 停牌  4: 退市 7: 新股 8: 变更)

**示例**
```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

briefs = quote_client.get_stock_briefs(['AAPL'])

# 将 time 转换为对应时区的日期时间
briefs['cn_date'] = pd.to_datetime(briefs['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
briefs['us_date'] = pd.to_datetime(briefs['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**
```
symbol  pre_close  halted           time    open   high       low  close  \
0   AAPL     174.33     0.0  1639602000000  175.11  179.5  172.3108  179.3   

    volume                   cn_date                   us_date  
0  131063257 2021-12-16 05:00:00+08:00 2021-12-15 16:00:00-05:00 
```


---
<!--
## get\_stock\_details 获取股票详情
`QuoteClient.get_stock_details(symbols, lang=None)`

**说明**

获取股票详情，包括价格、股本、每股收益、交易状态等基本信息

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码的列表，目前仅支持获取美股延迟行情。 如 ['AAPL', 'MSFT']
lang|Language|支持的语言， 可以使用tigeropen.common.consts.Language下提供的枚举常量， 如 Language.en_US

**返回**

`pandas.DataFrame`

结构如下：

参数名|类型|描述
----|----|----
symbol|str|代码
market|str|市场
sec_type|str|证券类型
exchange|str|交易所
name|str|名称
shortable|int|做空信息
ask_price|float|卖一价
ask_size|int|卖一量
bid_price|float|买一价
bid_size|int|买一量
pre_close|float|前收价
latest_price|float|最新价
adj_pre_close|float|复权后前收价
latest_time|int|最新成交时间，毫秒时间戳
volume|int|成交量
open|float|开盘价
high|float|最高价
low|float|最低价
change|float|涨跌额
amount|int|成交额
amplitude|float|振幅
market_status|str|市场状态 （未开盘，交易中，休市等）
trading_status|int|0: 非交易状态 1: 盘前交易（盘前竞价） 2: 交易中 3: 盘后交易（收市竞价）
float_shares|int|流通股本
shares|int|总股本
eps|float|每股收益
adr_rate|float|ADR的比例数据，非ADR的股票为None
etf|int|非0表示该股票是ETF,1表示不带杠杆的etf,2表示2倍杠杆etf,3表示3倍etf杠杆
listing_date|int|上市日期毫秒时间戳（该市场当地时间零点），该key可能不存在
next_market_status|str|下一交易时段信息
hour_trading|str|盘前盘后信息
stock_split||拆合股
stock_right|str|股权信息
symbol_change|str|股票代码变更
stock_notice|str|公告



**示例**

```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

details = quote_client.get_stock_details(['AAPL'])
print(details)
```
 
**返回示例**
```python
    symbol market exchange sec_type name  shortable  latest_price  pre_close  \
  0   AAPL     US   NASDAQ      STK   苹果        3.0         179.3     174.33   

    adj_pre_close  trading_status market_status      timestamp  \
  0         174.33               0           未开盘  1639602000000   

            latest_time    open   high       low     volume        amount  \
  0  12-15 16:00:00 EST  175.11  179.5  172.3108  131063257  2.305655e+10   

    ask_price  ask_size  bid_price  bid_size  change  amplitude  halted  delay  \
  0        0.0         0        0.0         0    4.97   0.041239     0.0      0   

    float_shares       shares   eps  etf  listing_date  adr_rate  \
  0   16389352394  16406397000  5.61  0.0  345445200000       0.0   

    hour_trading_tag  hour_trading_latest_price  hour_trading_pre_close  \
  0               盘后                     179.72                   179.3   

    hour_trading_latest_time  hour_trading_volume  hour_trading_timestamp  \
  0                19:59 EST              9346453           1639616398088   

    next_market_status_tag  next_market_status_begin_time  \
  0                   盘前交易                  1639645200000   

    stock_split_execute_date stock_split_to_factor stock_split_for_factor  \
  0                     None                  None                   None   

    stock_right_symbol stock_right_rights_symbol stock_right_first_dealing_date  \
  0               None                      None                           None   

    stock_right_last_dealing_date symbol_change_new_symbol  \
  0                          None                     None   

    symbol_change_execute_date stock_notice_title stock_notice_content  \
  0                       None               None                 None   

    stock_notice_type  
  0              None  
```

---
-->

### get\_bars 获取个股K线数据 

`
QuoteClient.get_bars(symbols, 
                    period=BarPeriod.DAY, 
                    begin_time=-1, 
                    end_time=-1, 
                    right=QuoteRight.BR, 
                    limit=251, 
                    lang=NONE)
`

**说明**

获取股票的k线数据， 包括不同的时间周期， 如日线、周线、分钟线等

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，单次上限50只， 如 ['AAPL', 'GOOG']
period|BarPeriod|获取的K线周期。默认 BarPeriod.DAY，可以使用 tigeropen.common.consts.BarPeriod 下提供的枚举常量，如 BarPeriod.DAY。 'day'/'week'/'month'/'year'/'1min'/'5min'/'15min'/'30min'/'60min'
begin_time|int或str|起始时间。支持毫秒级别的时间戳或日期字符串，如 1639371600000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
end_time|int或str|截至时间。支持毫秒级别的时间戳或日期字符串，如 1639371600000 或 '2019-06-07 23:00:00' 或 '2019-06-07' 
right|QuoteRight|复权方式。默认前复权，可以使用 tigeropen.common.consts.QuoteRight 下提供的枚举常量，如 QuoteRight.BR 前复权，QuoteRight.NR 不复权 
limit|int|限制数据的条数。默认 251
lang|Language|支持的语言， 可以使用 tigeropen.common.consts.Language 下提供的枚举常量， 如 Language.zh_CN

**返回**

`pandas.DataFrame`

结构如下：

参数名|类型|描述
----|----|----
time| int |毫秒时间戳，如 1639371600000
open| float |Bar 的开盘价
close| float |Bar 的收盘价
high| float |Bar 的最高价
low| float |Bar 的最低价
volume| float |Bar 的成交量

**示例**

```python
import pandas as pd
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

bars = quote_client.get_bars(['AAPL'])
print(bars.head())

# 转换 time 格式
bars['cn_date'] = pd.to_datetime(bars['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
bars['us_date'] = pd.to_datetime(bars['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**
```python
  symbol           time      open      high       low   close    volume  
0     00700  1515474000000  174.5500  175.0600  173.4100  174.33  21583997   
1     00700  1515560400000  173.1600  174.3000  173.0000  174.29  23959895   
2     00700  1515646800000  174.5900  175.4886  174.4900  175.28  18667729   
3     00700  1515733200000  176.1800  177.3600  175.6500  177.09  25418080   
4     00700  1516078800000  177.9000  179.3900  176.1400  176.19  29565947   
```


---

### get\_depth\_quote 获取深度行情
`
QuoteClient.get_depth_quote(symbols, market)
`

**说明**

获取输入证券代码的买卖N档挂单数据，包括委托价格，数量及订单数，单次请求上限为100只

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**
参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，单次上限100只
market|Market|查询的市场， 可以使用 tigeropen.common.consts.Market 下提供的枚举常量

**返回**

`dict`

数据示例:
```python
  若返回单个 symbol:
  {'symbol': '02833',
    'asks': [(27.4, 300, 2), (27.45, 500, 1), (27.5, 4400, 1), (27.55, 0, 0), (27.6, 5700, 3), (27.65, 0, 0),
            (27.7, 500, 1), (27.75, 0, 0), (27.8, 0, 0), (27.85, 0, 0)],
    'bids': [(27, 4000, 3), (26.95, 200, 1), (26.9, 0, 0), (26.85, 400, 1), (26.8, 0, 0), (26.75, 0, 0),
            (26.7, 0, 0), (26.65, 0, 0), (26.6, 0, 0), (26.55, 0, 0)]
  }

  若返回多个 symbol:
  {'02833':
      {'symbol': '02833',
        'asks': [(27.35, 200, 1), (27.4, 2100, 2), (27.45, 500, 1), (27.5, 4400, 1), (27.55, 0, 0),
                (27.6, 5700, 3), (27.65, 0, 0), (27.7, 500, 1), (27.75, 0, 0), (27.8, 0, 0)],
        'bids': [(27.05, 100, 1), (27, 5000, 4), (26.95, 200, 1), (26.9, 0, 0), (26.85, 400, 1), (26.8, 0, 0),
              (26.75, 0, 0), (26.7, 0, 0), (26.65, 0, 0), (26.6, 0, 0)]
      },
  '02828':
      {'symbol': '02828',
        'asks': [(106.6, 6800, 7), (106.7, 110200, 10), (106.8, 64400, 8), (106.9, 80600, 8), (107, 9440, 16),
              (107.1, 31800, 5), (107.2, 11800, 4), (107.3, 9800, 2), (107.4, 9400, 1), (107.5, 21000, 9)],
        'bids': [(106.5, 62800, 17), (106.4, 68200, 9), (106.3, 78400, 6), (106.2, 52400, 4), (106.1, 3060, 4),
                (106, 33400, 4), (105.9, 29600, 3), (105.8, 9600, 2), (105.7, 15200, 2), (105.6, 0, 0)]}
      }
```

asks 和 bids 对应的列表项数据含义为 (委托价格，委托数量，委托订单数) :
```python
[(ask_price1, ask_volume1, order_count), (ask_price2, ask_volume2, order_count), ...]
[(bid_price1, bid_volume2, order_count), (bid_price2, bid_volume2, order_count), ...]
```

**示例**
```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

depth_quote = quote_client.get_depth_quote(['02833'], Market.HK)
```

**返回示例**
```python
{'symbol': '02833',
'asks': [(27.4, 300, 2), (27.45, 500, 1), (27.5, 4400, 1), (27.55, 0, 0), (27.6, 5700, 3), (27.65, 0, 0),
        (27.7, 500, 1), (27.75, 0, 0), (27.8, 0, 0), (27.85, 0, 0)],
'bids': [(27, 4000, 3), (26.95, 200, 1), (26.9, 0, 0), (26.85, 400, 1), (26.8, 0, 0), (26.75, 0, 0),
        (26.7, 0, 0), (26.65, 0, 0), (26.6, 0, 0), (26.55, 0, 0)]
}
```

---

### get\_short\_interest 获取美股的做空数据 (目前未上线)

`
QuoteClient.get_short_interest(symbols, lang=None)
`

**说明**

获取美股做空数据

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表，单次上限100只
lang| Language |支持的语言， 可以使用 tigeropen.common.consts.Language 下提供的枚举常量

**返回** 

pandas.DataFrame 

结构如下：

COLUMN|类型|描述
----|----|----
symbol|str|证券代码
settlement_date|str|收集信息的时间 
short_interest|int|未平仓做空股数
avg_daily_volume|int|过去一年的日均成交量
days_to_cover|float|回补天数。使用最近一次获取的未平仓做空股数/日均成交量得到
percent_of_float|float|未平仓股数占流通股本的比重

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

short_interest = quote_client.get_short_interest(['GOOG', 'AAPL', '00700'])
```

**返回示例**

```python
symbol settlement_date  short_interest  avg_daily_volume  days_to_cover   percent_of_float  
0    GOOG      2018-12-14         2193320           1894404       1.000000             0.7000  
1    GOOG      2018-11-30         2580444           1846248       1.000000             0.8000  
2    GOOG      2018-11-15         2300074           1677483       1.000000             0.8000  
3    GOOG      2018-10-31         2206410           2371360       1.000000             0.7000  
4    GOOG      2018-10-15         2103149           1821532       1.000000             0.7000  
```

---

### get\_trade\_metas 获取股票交易需要的信息

`
QuoteClient.get_trade_metas(symbols)
`

**说明**

获取股票交易需要的信息，如每手股数，比如港股下单股数必须为每手股数的整数倍

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参数名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表, 上限为50

**返回**

`pandas.DataFrame`

结构如下:

COLUMN|类型|描述
----|----|----
symbol|str|证券代码
lot_size|int|每手股数
min_tick|float|价格最小变动单位
spread_scale|float|报价精度

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

trade_metas = quote_client.get_trade_metas(symbols=['00700', '00336'])
```

**返回示例**

```python
symbol  lot_size  min_tick  spread_scale
0  00700       100      0.20             0
1  00336      1000      0.01             0
```