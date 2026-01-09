---
title: 期权
---
### OptionExpirationQueryRequest 获取期权过期日 

**说明**

获取期权过期日，请求上限为30支股票

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols|array|Yes|股票代码列表，上限为：30

**返回**
`com.tigerbrokers.stock.openapi.client.https.response.option.OptionExpirationResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/option/OptionExpirationResponse.java)

结构如下：
```java
public class OptionExpirationResponse extends TigerResponse {
  @JSONField(name = "data")
  private List<OptionExpirationItem> optionExpirationItems;
}
```

返回数据可通过`OptionExpirationResponse.getOptionExpirationItems()`方法访问，返回`OptionExpirationItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionExpirationItem` 属性如下:

名称|类型|说明
--- | --- | ---
symbol|string|股票代码
count|int|过期日期个数
dates|array|过期时间,日期格式，如：2018-12-01
timestamps|array|过期日期，时间戳格式，如：1544763600000(美国NewYork时间对应的时间戳)

具体字段可通过对象的get方法，如`getSymbol()`进行访问，或通过对象的`toString()`方法转换为字符串

**示例**

```java
OptionExpirationResponse response = client.execute(new OptionExpirationQueryRequest(List.of("AAPL", "GOOG")));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getOptionExpirationItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

### OptionChainQueryV3Request 获取期权链

**说明**

获取期权链

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbol|string|Yes|股票代码，symbol和expiry组合上限为：30
expiry|long|Yes|期权过期日（美国NewYork时间当天0点所对应的毫秒值）

筛选参数：

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
implied_volatility|double|No|隐含波动率
in_the_money|boolean|No|是否价内
open_interest|int|No|未平仓量
delta|double|No|delta
gamma|double|No|gamma
theta|double|No|theta
vega|double|No|vega
rho|double|No|rho

**返回**
`com.tigerbrokers.stock.openapi.client.https.response.option.OptionChainResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/option/OptionChainResponse.java)

结构如下：
```java
public class OptionChainResponse extends TigerResponse {
  @JSONField(name = "data")
  private List<OptionChainItem> optionChainItems;
}
```

返回数据可通过`OptionChainResponse.getOptionChainItems()`方法访问，返回`OptionChainItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionChainItem` 属性如下:

名称|类型|说明
--- | --- | ---
symbol|string|标的股票代码
expiry|long|期权过期日
items|List\<OptionRealTimeQuoteGroup\>|列表，包含OptionRealTimeQuoteGroup对象，保存期权链数据，说明见下文

`OptionRealTimeQuoteGroup`对象结构：

名称|类型|说明
--- | --- | ---
put |OptionRealTimeQuote|看跌期权合约
call|OptionRealTimeQuote|看涨期权合约

`OptionRealTimeQuote`对象结构：

名称|类型|说明
--- | --- | ---
symbol|string|标的股票代码
expiry|long|期权过期日
askPrice|double|卖盘价格
askSize|int|卖盘数量
bidPrice|double|买盘价格
bidSize|int|买盘数量
identifier|string|期权标识,如:AAPL210115C00095000
lastTimestamp|long|最新成交时间 如:1543343800698
latestPrice|double|最新价
multiplier|double|乘数，美股期权默认100
openInterest|int|未平仓量
preClose|double|前一交易日的收盘价
right|string|期权方向 PUT/CALL
strike|double|行权价
volume|long|成交量

具体字段可通过对象的get方法，如`getSymbol()`进行访问，或通过对象的`toString()`方法转换为字符串

**示例**
```java
OptionChainModel basicModel = new OptionChainModel("AAPL", "2022-02-18");
OptionChainFilterModel filterModel = new OptionChainFilterModel().inTheMoney(true).impliedVolatility(0.6837, 0.7182).openInterest(0, 100).greeks(new OptionChainFilterModel.Greeks().delta(-0.5, 0.5).gamma(0.069, 0.071).vega(0.019, 0.023).theta(-0.074, -0.036).rho(-0.012, 0.001));
OptionChainResponse response = client.execute(OptionChainQueryV3Request.of(basicModel, filterModel));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getOptionChainItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "message": "success",
  "data": [{
    "expiry": 1610686800000,
    "items": [{
      "call": {
        "askPrice": 567.1,
        "askSize": 1,
        "bidPrice": 557.2,
        "bidSize": 1,
        "identifier": "GOOG210115C00520000",
        "lastTimestamp": 1543943398344,
        "latestPrice": 607.57,
        "multiplier": 100,
        "openInterest": 2,
        "preClose": 607.57,
        "right": "call",
        "strike": "520.0",
        "volume": 0
      },
      "put": {
        "askPrice": 11.0,
        "askSize": 1,
        "bidPrice": 7.8,
        "bidSize": 1,
        "identifier": "GOOG210115C00520000",
        "lastTimestamp": 1543601911584,
        "latestPrice": 8.1,
        "multiplier": 100,
        "openInterest": 4,
        "preClose": 8.1,
        "right": "put",
        "strike": "520.0",
        "volume": 0
      }
    }],
    "symbol": "GOOG"
  }],
  "timestamp": 1544511515566,
  "sign": "la0Nfw8VM3javYu41JugFTR1U5xiJ7++FRgL0BmrrNlQiukA0KHsvZYCzKjkoGjeiE3heBW9KSy3Q5oDiAmC3C + iEqyRApq8oLDdJH17gyU6gMoSEfQX7xHTHkbPfBq3M3j / XoFTclOFKMjJ1SU9UkHtgnw6vXi + MHpLKknWqfQ = "
}
```

### OptionBriefQueryRequest 获取期权行情摘要

**说明**

获取期权行情摘要

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbol|string|yes|股票代码,上限为30个
right|string|yes|看多或看空（CALL/PUT）
expiry|long|yes|到期时间（美国NewYork时间当天0点所对应的毫秒值）
strike|string|yes|行权价

**返回**
`com.tigerbrokers.stock.openapi.client.https.request.option.OptionBriefResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/option/OptionBriefResponse.java)

结构如下：
```java
public class OptionBriefResponse extends TigerResponse {
  @JSONField(name = "data")
  private List<OptionBriefItem> optionBriefItems;
}
```

返回数据可通过`TigerResponse.getOptionBriefItems()`方法访问，返回`OptionBriefItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionBriefItem` 属性如下:

字段|类型|说明
--- | --- | ---
symbol|string|股票代码
strike|string|行权价
bidPrice|double|买盘价格
bidSize|int|买盘数量
askPrice|double|卖盘价格
askSize|int|卖盘数量
latestPrice|double|最新价格
timestamp|long|最新成交时间
latestTime|string|最新成交时间(美东时间，yyyy-MM-dd HH:mm:ss.SSS)
volume|int|成交量
high|double|最高价
low|double|最低价
open|double|开盘价
preClose|double|前一交易日收盘价
openInterest|int|未平仓量
change|double|涨跌额
multiplier|int|乘数，美股期权默认100
right|string|方向 (PUT/CALL)
volatility|string|历史波动率
expiry|long|到期时间（毫秒，当天0点）

具体字段可通过对象的get方法，如`getSymbol()`进行访问， 或通过对象的`toString()`方法转换为字符串

**示例**
```java
OptionCommonModel model = new OptionCommonModel();
model.setSymbol("AAPL");
model.setRight("CALL");
model.setStrike("160.0");
model.setExpiry("2021-12-03");
OptionBriefResponse response = client.execute(OptionBriefQueryRequest.of(model));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getOptionBriefItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "data": [{
    "askPrice": 1.53,
    "askSize": 128,
    "bidPrice": 1.32,
    "bidSize": 1,
    "change": -2.52,
    "expiry": 1638507600000,
    "high": 3.12,
    "identifier": "AAPL  211203C00160000",
    "latestPrice": 1.35,
    "latestTime": "2021-11-26 12:59:59.936",
    "low": 1.25,
    "multiplier": 100,
    "open": 2.74,
    "openInterest": 79355,
    "preClose": 3.87,
    "ratesBonds": 0.002,
    "right": "call",
    "strike": "160.0",
    "symbol": "AAPL",
    "timestamp": 1637949599936,
    "volatility": "20.43%",
    "volume": 113452
  }],
  "message": "",
  "timestamp": 1543831981533
}
```

### OptionKlineQueryRequest 获取期权K线

**说明**

获取期权K线

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbol|string|yes|股票代码,上限：30
right|string|yes|看多或看空（CALL/PUT）
expiry|long|yes|到期时间（美国NewYork时间当天0点所对应的毫秒值）
strike|string|yes|行权价
begin_time|long|yes|开始时间（美国NewYork时间所对应的毫秒值）
end_time|long|yes|结束时间（美国NewYork时间所对应的毫秒值）

**返回**
`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionKlineResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/option/OptionKlineResponse.java)

结构如下：
```java
public class OptionKlineResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<OptionKlineItem> klineItems;
  }
```

返回数据可通过`TigerResponse.getKlineItems()`方法访问，返回`OptionKlineItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionKlineItem` 属性如下:

名称|类型|说明
--- | --- | ---
symbol|string|股票代码
period|string|周期
right|string|看多或看空，取值CALL/PUT
strike|string|行权价
expiry|long|到期时间，毫秒
items|List\<OptionKlinePoint\>|包含OptionKlinePoint的List，OptionKlinePoint为k线数组，包含的具体数据见下文

OptionKlinePoint对象属性如下：

high|double|最高价
low|double|最低价
open|double|开盘价
close|double|收盘价
time|long|k线时间
volume|int|成交量

具体字段可通过对象的get方法，如`getSymbol()`进行访问, 或通过对象的`toString()`方法转换为字符串

**示例**
```java
OptionKlineModel model = new OptionKlineModel();
model.setSymbol("BABA");
model.setRight("CALL");
model.setStrike("129.0");
model.setExpiry("2019-01-04");
model.setBeginTime("2018-12-10");
model.setEndTime("2019-12-26");

OptionKlineResponse response = client.execute(OptionKlineQueryRequest.of(model));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getKlineItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "data": [{
    "period": "day",
    "symbol": "AAPL",
    "items": [{
      "close": 83.75,
      "high": 83.75,
      "low": 83.75,
      "open": 83.75,
      "time": 1543208400000,
      "volume": 31
    }]
  }],
  "message": "",
  "timestamp": 1543838119424
}
```

### OptionTradeTickQueryRequest 获取期权逐笔成交

**说明**

获取期权逐笔成交数据

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbol|string|yes|股票代码,上限：30
right|string|yes|看多或看空（call/put）
expiry|long|yes|到期时间（美国NewYork时间当天0点所对应的毫秒值）
strike|string|yes|行权价

**返回**
`com.tigerbrokers.stock.openapi.client.https.response.option.OptionTradeTickResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/option/OptionTradeTickResponse.java)

结构如下：
```java
public class OptionTradeTickResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<OptionTradeTickItem> optionTradeTickItems;
}
```

返回数据可通过`TigerResponse.getOptionTradeTickItems()`方法访问，返回`OptionTradeTickItem`对象列表，其中`com.tigerbrokers.stock.openapi.client.https.domain.option.item.OptionTradeTickItem` 属性如下:

名称|类型|说明
--- | --- | ---
symbol|string|标的股票代码
expiry|long|到期时间
strike|string|strike price
right|string|PUT或CALL
items|List\<TradeTickPoint\>|TradeTickPoint对象列表，每个TradeTickPoint对象对应单条逐笔成交数据

TradeTickPoint对象结构如下：
名称|类型|说明
--- | --- | ---
price|double|成交价格
time|long|成交时间
volume|long|成交量

具体字段可通过对象的get方法，如`getSymbol()`进行访问, 或通过对象的`toString()`方法转换为字符串

**示例**
```java
OptionCommonModel model = new OptionCommonModel();
model.setSymbol("AAPL");
model.setRight("CALL");
model.setStrike("95.0");
model.setExpiry("2019-01-11");

OptionTradeTickResponse response = client.execute(OptionTradeTickQueryRequest.of(model));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getOptionTradeTickItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**
```json
{
  "code": 0,
  "message": "success",
  "data": [{
    "expiry": 1545368400000,
    "right": "call",
    "strike": "150.0",
    "symbol": "AAPL",
    "items": [{
      "price": 21.89,
      "time": 1544713469939,
      "volume": 1
    }, {
      "price": 21.48,
      "time": 1544715658232,
      "volume": 1
    }, {
      "price": 21.6,
      "time": 1544718959004,
      "volume": 1
    }, {
      "price": 21.61,
      "time": 1544718959044,
      "volume": 2
    }, {
      "price": 20.7,
      "time": 1544724445500,
      "volume": 10
    }, {
      "price": 21.2,
      "time": 1544726208670,
      "volume": 2
    }, {
      "price": 20.5,
      "time": 1544729490963,
      "volume": 1
    }, {
      "price": 21.4,
      "time": 1544733902794,
      "volume": 3
    }, {
      "price": 21.19,
      "time": 1544734051420,
      "volume": 1
    }]
  }],
  "timestamp": 1544734051420
}
```
### 期权指标计算

**说明**

计算所选期权的各类指标

**参数**

参数|类型|是否必填|描述
----|----|----|----
client|object|yes|SDK Htttp client
symbol|string|yes|股票代码
right|string|yes|看多或看空（CALL/PUT）
strike|string|yes|行权价
expiry|long|yes|到期时间（美国NewYork时间当天0点所对应的毫秒值）

**返回**

名称|类型|说明
----|----|----|----
delta|double|希腊字母delta
gamma|double|希腊字母gamma
theta|double|希腊字母theta
vega|double|希腊字母vega
insideValue|double|内在价值
timeValue|double|时间价值
leverage|double|杠杆率
openInterest|int|未平仓量
historyVolatility|string|历史波动率，百分比格式
premiumRate|string|溢价率，百分比格式
profitRate|string|买入盈利率，百分比格式
volatility|string|隐含波动率，百分比格式

**示例**
```java
OptionFundamentals optionFundamentals = OptionCalcUtils.getOptionFundamentals(client,"BABA", "CALL", "205.0", "2019-11-01");
   System.out.println(JSONObject.toJSONString(optionFundamentals));
```

**返回示例**
```json
{
  "delta":"0.023",
  "gamma":"0.005",
  "historyVolatility":"33.63%",
  "insideValue":"0.00",
  "leverage":"41.08",
  "openInterest":"129",
  "premiumRate":"14.79%",
  "profitRate":"2.08%",
  "theta":"-0.061",
  "timeValue":"0.10",
  "vega":"0.012",
  "volatility":"55.54%"
}
```