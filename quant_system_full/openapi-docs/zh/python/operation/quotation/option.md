---
title: 期权
---
### get\_option\_expirations 获取美股期权到期日

`QuoteClient.get_option_expirations(symbols)`

**说明**

获取美股期权到期日

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**

参量名|类型|描述
----|----|----
symbols|list\<str...\>|证券代码列表

**返回**

`pandas.DataFrame`

各 column 的含义如下：

参量名|类型|描述
----|----|----
symbol|str|证券代码
date|str|到日期 YYYY-MM-DD 格式的字符串
timestamp|int|到期日，精确到毫秒的时间戳


**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

expiration = quote_client.get_option_expirations(symbols=['AAPL'])
```

**返回示例**

```python
  symbol        date      timestamp
0    AAPL  2019-01-11  1547182800000
1    AAPL  2019-01-18  1547787600000
2    AAPL  2019-01-25  1548392400000
3    AAPL  2019-02-01  1548997200000
4    AAPL  2019-02-08  1549602000000
```

---

### get\_option\_chain 获取期权链

`QuoteClient.get_option_chain(symbol, expiry, option_filter=None)`

**说明**

获取期权链


**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit.html)

**参数**
参量名|类型|描述
----|----|----
symbol|str|期权对应的正股代码
expiry|str或int|期权到期日，毫秒单位的数字时间戳或日期字符串，如 1705640400000 或 '2024-01-19'
option_filter|tigeropen.quote.domain.filter.OptionFilter|过滤参数，可选

过滤参数:

各筛选指标, 除去 in_the_money 属性外, 其他指标使用时均对应 _min 后缀(表示范围最小值) 或 _max 后缀(表示范围最大值) 的字段名, 如 delta_min, theta_max, 参见代码示例.

OptionFilter可筛选指标如下：

| 参数  | 类型  | 是否必填   | 描述 |  
| :--- | :--- | :--- | :--- |
|implied_volatility|float| No |隐含波动率, 反映市场预期的未来股价波动情况, 隐含波动率越高, 说明预期股价波动越剧烈. |
|in_the_money  |bool|   No |是否价内|
|open_interest |int  |No   |未平仓量, 每个交易日完结时市场参与者手上尚未平仓的合约数. 反映市场的深度和流动性. |
|delta   |float|  No|   delta, 反映正股价格变化对期权价格变化对影响. 股价每变化1元, 期权价格大约变化 delta. 取值 -1.0 ~ 1.0 |
|gamma   |float|  No|   gamma, 反映正股价格变化对于delta的影响. 股价每变化1元, delta变化gamma. | 
|theta   |float|  No|   theta, 反映时间变化对期权价格变化的影响. 时间每减少一天, 期权价格大约变化 theta. |
|vega |float|  No|   vega, 反映波动率对期权价格变化的影响. 波动率每变化1%, 期权价格大约变化 vega. |
|rho  |float|  No|   rho, 反映无风险利率对期权价格变化的影响. 无风险利率每变化1%, 期权价格大约变化 rho. | 

**返回**

`pandas.DataFrame`


字段名|类型|描述
----|----|----
identifier|str|期权代码
symbol|str|期权对应的正股代码
expiry|int|期权到期日，毫秒级别的时间戳
strike|float|行权价
put_call|str|期权的方向
multiplier|float|乘数
ask_price|float|卖价
ask_size|int|卖量
bid_price|float|买价
bid_size|int|买量
pre_close|float|前收价
latest_price|float|最新价
volume|int|成交量
open_interest|int|未平仓数量

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

option_chain = quote_client.get_option_chain(symbol='AAPL', expiry='2019-01-18')
print(option_chain)


# 可定义 OptionFilter 进行过滤
option_filter = OptionFilter(implied_volatility_min=0.5, implied_volatility_max=0.9, delta_min=0, delta_max=1,
                          open_interest_min=100, gamma_min=0.005, theta_max=-0.05, in_the_money=True)
option_chain = openapi_client.get_option_chain('AAPL', '2023-01-20', option_filter=option_filter)
print(option_chain)

# 也可直接用指标名称过滤
option_chain = openapi_client.get_option_chain('AAPL', '2023-01-20', implied_volatility_min=0.5, open_interest_min=200, vega_min=0.1, rho_max=0.9)
                                      
# 转换 expiry 时间格式
option_chain['expiry_date'] = pd.to_datetime(option_chain['expiry'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**
```python
  symbol         expiry             identifier strike put_call  bid_price  bid_size  ask_price  ask_size  volume  latest_price  pre_close  open_interest  multiplier               expiry_date
0   AAPL  1645160400000  AAPL  220218C00095000   95.0     CALL      83.35        40      85.45        39     345         84.61      77.61            511         100 2022-02-18 00:00:00-05:00
1   AAPL  1645160400000  AAPL  220218C00100000  100.0     CALL      79.15        30      80.90        35       9         79.96      80.38           1121         100 2022-02-18 00:00:00-05:00
2   AAPL  1645160400000  AAPL  220218C00105000  105.0     CALL      74.20         2      75.95        34       0         69.44      69.44            578         100 2022-02-18 00:00:00-05:00
3   AAPL  1645160400000  AAPL  220218C00110000  110.0     CALL      69.25         2      71.05        29       1         69.80      65.20            211         100 2022-02-18 00:00:00-05:00
4   AAPL  1645160400000  AAPL  220218C00115000  115.0     CALL      64.25        35      66.00        36       5         64.55      59.35            797         100 2022-02-18 00:00:00-05:00
5   AAPL  1645160400000  AAPL  220218C00120000  120.0     CALL      59.35         8      61.20         4       2         61.48      59.13           1483         100 2022-02-18 00:00:00-05:00
6   AAPL  1645160400000  AAPL  220218C00125000  125.0     CALL      54.40         2      55.15         3       3         54.85      55.30           1645         100 2022-02-18 00:00:00-05:00
7   AAPL  1645160400000  AAPL  220218C00135000  135.0     CALL      44.60         3      46.35         3       9         44.74      45.70           1184         100 2022-02-18 00:00:00-05:00
```
---


### get\_option\_briefs 获取期权最新行情

`QuoteClient.get_option_briefs(identifiers)`

**说明**

获取期权最新行情，包括盘口数据，最新成交数据等

**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
identifiers|str|期权代码列表，单次上线30只，如 ['AAPL 220128C000175000']

**返回**

`pandas.DataFrame`

结构如下：

参量名|类型|描述
----|----|----
identifier|str|期权代码
symbol|str|期权对应的正股代码
expiry|int|到期日，毫秒级时间戳
strike|float|行权价
put_call|str|期权方向
multiplier|float|乘数
ask_price|float|卖价
ask_size|int|卖量
bid_price|float|买价
bid_size|int|买量
pre_close|float|前收价
latest_price|float|最新价
latest_time|int|最新交易时间
volume|int|成交量
open_interest|int|未平仓数量    
open|float|开盘价
high|float|最高价
low|float|最低价


**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

briefs = quote_client.get_option_briefs(['AAPL  220128P000175000', 'AAPL 220128C000175000'])
print(briefs)

# 转换 expiry 时间格式
briefs['expiry_date'] = pd.to_datetime(briefs['expiry'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**
```python
              identifier symbol         expiry strike put_call  multiplier  ask_price  ask_size  bid_price  bid_size  pre_close  latest_price latest_time  volume  open_interest  open   high   low  rates_bonds volatility               expiry_date
0  AAPL  220128P00175000   AAPL  1643346000000  175.0      PUT         100        4.5       398       4.25       115       4.12          4.40        None     504            700  4.05   4.65  3.90       0.0039     30.34% 2022-01-28 00:00:00-05:00
1  AAPL  220128C00175000   AAPL  1643346000000  175.0     CALL         100        9.1        24       8.65        25       9.50          8.85        None    1110           2816  9.38  10.05  8.52       0.0039     30.34% 2022-01-28 00:00:00-05:00
```

---

### get\_option\_bars 获取期权日K数据

`QuoteClient.get_option_bars(identifiers, begin_time=-1, end_time=4070880000000)`

**说明**

获取期权日线数据


**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
identifiers|str|期权代码列表， 单次上线30只， 如 ['AAPL 220128C000175000']
begin_time|str或int|开始时间，毫秒级时间戳或日期字符串，如 1643346000000 或 '2019-01-01'
end_time|str或int|结束时间，毫秒级时间戳或日期字符串，如 1643346000000 或 '2019-01-01'

**返回**

`pandas.DataFrame`

结构如下：

参量名|类型|描述
----|----|----
identifier|str|期权代码
symbol|str|期权对应的正股代码
expiry|int|到期日，毫秒级时间戳
put_call|str|期权方向
strike|float|行权价
time|int|Bar对应的时间，毫秒级时间戳
open|float|开盘价
high|float|最高价
low|float|最低价
close|float|收盘价
volume|int|成交量
open_interest|int|未平仓数量


**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

bars = quote_client.get_option_bars(['AAPL 190104P00134000'])
print(bars)

# 转换 time 时间格式
bars['expiry_date'] = pd.to_datetime(bars['expiry'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
bars['time_date'] = pd.to_datetime(bars['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
```

**返回示例**

```python
               identifier symbol         expiry put_call  strike           time   open   high   low  close  volume  open_interest               expiry_date                 time_date
0   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639026000000   8.92   9.80  8.00   8.20     364              0 2022-01-28 00:00:00-05:00 2021-12-09 00:00:00-05:00
1   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639112400000   9.05  10.80  7.80  10.80     277            177 2022-01-28 00:00:00-05:00 2021-12-10 00:00:00-05:00
2   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639371600000  11.70  12.50  8.72   8.75     304            328 2022-01-28 00:00:00-05:00 2021-12-13 00:00:00-05:00
3   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639458000000   8.60   9.82  7.00   7.80    1139            427 2022-01-28 00:00:00-05:00 2021-12-14 00:00:00-05:00
4   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639544400000   8.30  10.70  7.05  10.57     927           1207 2022-01-28 00:00:00-05:00 2021-12-15 00:00:00-05:00
5   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639630800000  10.50  11.60  6.60   7.14    1118           1447 2022-01-28 00:00:00-05:00 2021-12-16 00:00:00-05:00
6   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639717200000   6.75   7.25  5.80   6.25     674           2057 2022-01-28 00:00:00-05:00 2021-12-17 00:00:00-05:00
7   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1639976400000   5.00   5.85  4.75   5.40    1034           2363 2022-01-28 00:00:00-05:00 2021-12-20 00:00:00-05:00
8   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640062800000   6.20   6.30  4.85   6.20     756           2549 2022-01-28 00:00:00-05:00 2021-12-21 00:00:00-05:00
9   AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640149200000   6.24   7.40  5.90   7.40    2211           3002 2022-01-28 00:00:00-05:00 2021-12-22 00:00:00-05:00
10  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640235600000   7.35   7.65  7.09   7.40     909           3031 2022-01-28 00:00:00-05:00 2021-12-23 00:00:00-05:00
11  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640581200000   7.75   9.70  7.60   9.50    1165           3324 2022-01-28 00:00:00-05:00 2021-12-27 00:00:00-05:00
12  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640667600000   9.38  10.05  8.52   8.85    1110           2816 2022-01-28 00:00:00-05:00 2021-12-28 00:00:00-05:00
```

---

### get\_option\_trade\_ticks 获取期权的逐笔成交数据

`QuoteClient.get_option_trade_ticks(identifiers)`

**说明**

获取期权的逐笔成交数据


**请求频率**

频率限制请参考：[接口请求限制](/zh/python/appendix3/requestLimit)

**参数**

参量名|类型|描述
----|----|----
identifiers|str|期权代码列表，如 ['AAPL 220128C000175000']

**返回**

`pandas.DataFrame`

结构如下：

参量名|类型|描述
----|----|----
symbol|str|期权对应的正股代码
expiry|str|期权到期时间， YYYY-MM-DD 格式的字符串
put_call|str|期权方向
strike|float|行权价
time|int|成交时间
price|float|成交价格
volume|int|成交量

**示例**

```python
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

option_trade_ticks = quote_client.get_option_trade_ticks(['AAPL  190111P00134000'])
```

**返回示例**
```python
                identifier symbol         expiry put_call  strike           time  price  volume
0    AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640701803177   9.38       9
1    AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640701803177   9.38       1
2    AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640701803846   9.46       7
3    AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640701806266   9.55       1
4    AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640701918302   9.08       1
..                     ...    ...            ...      ...     ...            ...    ...     ...
111  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640722112754   8.91      25
112  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640723067491   9.00       4
113  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640723585351   8.85       4
114  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640724302670   9.13       2
115  AAPL  220128C00175000   AAPL  1643346000000     CALL   175.0  1640724600973   8.85       1
```