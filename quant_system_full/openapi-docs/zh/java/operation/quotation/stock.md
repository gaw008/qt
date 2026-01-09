---
title: 证券类
---
### 获取市场状态 QuoteMarketRequest

**说明**

获取市场状态

**参数**

参数 | 类型| 是否必填| 说明
---|---|---|---
market|string|Yes|US 美股，HK港股,CN A股，ALL 所有
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteMarketResponse`
[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteMarketResponse.java)

其结构如下:
```java
public class QuoteMarketResponse extends TigerResponse {
  @JSONField(
    name = "data"
  )
  private List<MarketItem> marketItems;
}
```

返回数据可用`QuoteMarketResponse.getMarketItems()`方法访问，返回为`marketItem`对象列表，其中 `com.tigerbrokers.stock.openapi.client.https.domain.quote.item.MarketItem` 属性如下：

字段|类型|说明
---|---|---
market|string|市场代码（US:美股，CN:沪深，HK:港股）
marketStatus|string|市场状态描述，非固定值，包括节假日信息等
status|string|市场状态，包括：NOT_YET_OPEN:未开盘,PRE_HOUR_TRADING:盘前交易,TRADING:交易中,MIDDLE_CLOSE:午间休市,POST_HOUR_TRADING:盘后交易,CLOSING:已收盘,EARLY_CLOSED:提前休市,MARKET_CLOSED:休市
openTime|string|最近开盘、交易时间 MM-dd HH:mm:ss 

`MarketItem`的具体字段可通过对象的get方法，如`getMarket()`，进行访问，或通过`toString()`方法转换为字符串的形式

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteMarketResponse response = client.execute(QuoteMarketRequest.newRequest(Market.US));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getMarketItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```
**返回示例**
```json
{
  "code": 0,
  "message": "success",
  "timestamp": 1525938835697,
  "data": [{"market":"US","marketStatus":"Closed Independence Day","status":"CLOSING","openTime":"07-04 09:30:00 EDT"}]
}
```

### 获取股票代号列表和名称 QuoteSymbolNameRequest

**说明**

获取股票代号列表和名称

**参数**

参数 | 类型| 是否必填| 说明
---|---|---|---
market|string|Yes|US 美股，HK港股,CN A股，ALL 所有
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteSymbolNameResponse`
[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteSymbolNameResponse.java)

具体结构如下
```java
public class QuoteSymbolNameResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<SymbolNameItem> symbolNameItems;
}
```

返回数据可通过`QuoteSymbolNameResponse.getSymbolNameItems()`方法访问，返回为包含`SymbolNameItem`对象的List，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.symbolNameItems` 属性如下：

字段|类型|说明
---|---|---
name|string|股票名称
symbol|string|股票代码

`SymbolNameItem`的具体字段可通过对象的get方法，如`getName()`进行访问，或通过`toString()`方法转换为字符串的形式

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteSymbolNameResponse response = client.execute(QuoteSymbolNameRequest.newRequest(Market.US));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getSymbolNameItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "message": "success",
  "timestamp": 1525938835697,
  "data": [{"name":"CKH Holdings","symbol":"00001"},{"name":"CLP","symbol":"00002"}]
}
```

### 获取股票代号列表 QuoteSymbolRequest

**说明**

获取股票代号列表

**参数**

参数 | 类型| 是否必填| 说明
---|---|---|---
market|string|Yes|US 美股，HK港股,CN A股，ALL 所有
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteSymbolResponse`
[source](https://github.com/tigerfintech/openapi-java-sdk/blob/71631121961002aa1dadf601e922a494d14f5ed0/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteSymbolResponse.java)

具体结构如下
```java
public class QuoteSymbolResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<String> symbols;
}
```

返回数据可以通过`QuoteSymbolResponse.getSymbols()`方法调用，结果为包含返回股票代号数据的List

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteSymbolResponse response = client.execute(QuoteSymbolRequest.newRequest(Market.US));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getSymbols().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "message": "success",
  "timestamp": 1525938835697,
  "data": ["A", "A.W", "AA", "AA-B", "AAAP", "AABA", "AAC", "AADR", "AAIT", "AAL", "AALCP", "AAMC", "AAME"]
}
```

### 延时行情 QuoteDelayRequest

**说明**

获取延时行情，此行情接口不需要购买行情权限即可调用，目前免费提供给用户

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols | string | Yes | 股票代码

**示例**

```java
List<String> symbols = new ArrayList<>();
symbols.add("AAPL");
symbols.add("TSLA");
QuoteDelayRequest delayRequest = QuoteDelayRequest.newRequest(symbols);
QuoteDelayResponse response = client.execute(delayRequest);
```

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteDelayResponse` [source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteDelayResponse.java)

具体结构如下：

```java
public class QuoteDelayResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<QuoteDelayItem> quoteDelayItems;
  }
```

返回数据可通过`TigerResponse.getQuoteDelayItems()`方法访问，返回为包含`QuoteDelayItem`对象的List，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.quoteDelayItem` 属性如下：

字段|类型|说明
---|---|---
close|double|收盘价
high|double|最高价
low|double|最低价
open|double|开盘价
preClose|double|昨日收盘价
time|long|时间
volume|long|成交量

`QuoteDelayItem`的具体字段可通过对象的get方法，如`getClose()`进行访问

**返回示例**

```json
[
  {
    "close": 156.81,
    "halted": 0,
    "high": 160.45,
    "low": 156.36,
    "open": 159.565,
    "preClose": 161.94,
    "symbol": "AAPL",
    "time": 1637949600000,
    "volume": 76959752
  },
  {
    "close": 1081.92,
    "halted": 0,
    "high": 1108.7827,
    "low": 1081,
    "open": 1099.47,
    "preClose": 1116,
    "symbol": "TSLA",
    "time": 1637949600000,
    "volume": 11680890
  }
]
```

### 获取分时数据 QuoteTimelineRequest

**说明**

获取分时数据

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols|array|Yes|股票代码列表
period|string|Yes|分时周期，包含：day 和 day5
include_hour_trading|boolean|No|是否包含盘前盘后数据，默认：false
begin_time|long|No|开始时间(毫秒时间戳),默认返回当天数据
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteTimelineResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteTimelineResponse.java)

具体结构如下：

返回数据可通过`QuoteTimelineResponse.getTimelineItems()`方法访问，返回为包含`TimelineItem`对象的List，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.TimelineItem` 属性如下:

字段|类型|说明
--- | --- | ---
symbol|string|股票代码
period|string|周期 day or 5day
preClose|double|昨日收盘价
intraday| object | 盘中分时数组,字段参考下面说明
preMarket|object|(仅美股)  盘前分时数组和起始结束时间,字段参考下面说明
afterHours| object| (仅美股)   盘后分时数组和起始结束时间,字段参考下面说明

`TimelineItem`的具体字段可通过对象的get方法，如`getSymbol()`，进行访问

分时数据intraday字段：

分时字段|说明
--- | -- | ---
volume| 成交量
avgPrice| 平均成交价格
price| 最新价格
time| 当前分时时间

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteTimelineResponse response = client.execute(QuoteTimelineRequest.newRequest(List.of("AAPL"), 1544129760000L));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getTimelineItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "data": [
    {
      "symbol": "AAPL",
      "preMarket": {
        "endTime": 1544106600000,
        "beginTime": 1544086800000,
        "items": [
           
        ]
      },
      "period": "day",
      "preClose": 176.69000244140625,
      "afterHours": {
        "endTime": 1544144400000,
        "beginTime": 1544130000000,
        "items": [
          {
            "volume": 872772,
            "avgPrice": 174.71916,
            "price": 174.69,
            "time": 1544130000000
          },
          {
            "volume": 3792,
            "avgPrice": 174.71893,
            "price": 174.66,
            "time": 1544130060000
          }
        ]
      },
      "intraday": [
        {
          "items": [
            {
              "volume": 201594,
              "avgPrice": 172.0327,
              "price": 174.34,
              "time": 1544129760000
            },
            {
              "volume": 139040,
              "avgPrice": 172.03645,
              "price": 174.4156,
              "time": 1544129820000
            },
            {
              "volume": 178427,
              "avgPrice": 172.0413,
              "price": 174.44,
              "time": 1544129880000
            },
            {
              "volume": 2969567,
              "avgPrice": 172.21619,
              "price": 174.72,
              "time": 1544129940000
            }
          ]
        }
      ]
    }
  ],
  "timestamp": 1544185099595,
  "message": "success"
}
```

### 获取实时行情 QuoteRealTimeQuoteRequest

**说明**

获取实时行情，需要购买行情权限才可以开通

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols|array|Yes|股票代码列表（单次上限50）
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteRealTimeQuoteResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteRealTimeQuoteResponse.java)

返回数据可通过`QuoteRealTimeQuoteResponse.getRealTimeQuoteItems()`方法访问，返回`RealTimeQuoteItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.RealTimeQuoteItem` 属性如下:

字段 | 类型 | 说明
--- | --- | ---
symbol | string| 股票代码
open | double| 开盘价
high | double| 最高价
low | double| 最低价
close | double | 收盘价
preClose| double| 前一交易日收盘价
latestPrice | double | 最新价
latestTime | long | 最新成交时间
latestSize | integer | 最新成交数量
askPrice| double | 卖盘价
askSize| long | 卖盘数量
bidPrice| double | 买盘价
bidSize| long | 买盘数量
volume | long | 成交量
status | short | 交易状态
hourTrading | object | 美股盘前盘后数据

其中`hourTrading`对象包含的字段：

字段 | 类型 | 说明
--- | --- | ---
tag | string | 盘前、盘后标识
latestPrice | double | 最新价
preClose | double | 昨收价
latestTime | string | 最新成交时间
volume | long | 成交量
timestamp | long | 最新成交时间

具体字段可通过对象的get方法，如`getSymbol()`，进行访问

**示例**

```java
QuoteRealTimeQuoteResponse response = client.execute(QuoteRealTimeQuoteRequest.newRequest(List.of("AAPL")));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getRealTimeQuoteItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json    
{
  "code": 0,
  "data": [
    {
      "symbol": "AAPL",
      "latestPrice": 174.72,
      "askPrice": 173.79,
      "bidSize": 2,
      "bidPrice": 173.5,
      "volume": 43098410,
      "high": 174.78,
      "preClose": 176.69,
      "low": 170.42,
      "latestTime": 1544130000000,
      "close": 174.72,
      "askSize": 1,
      "open": 171.76,
      "status": 0.0
    }
  ],
  "timestamp": 1544178615507,
  "message": "success"
}
```

### 获取K线数据 QuoteKlineRequest

**说明**

获取K线数据

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols|array|Yes|股票代码列表，上限：50，其中A股上限：10
period|string|Yes|K线类型，取值范围 (day: 日K,week: 周K,month:月K ,year:年K,1min:1分钟,5min:5分钟,15min:15分钟,30min:30分钟,60min:60分钟)
right|string|No|复权选项 ，br: 前复权（默认），nr: 不复权
begin_time|long|No|开始时间，默认：-1，单位:毫秒(ms),前闭后开区间，即查询结果会包含起始时间数据，如查询周/月/年K线，会返回包含当前周期的数据(如:起始时间为周三，会返回从这周一的K线)
end_time|long|No|结束时间，默认：-1，单位:毫秒(ms)
limit|integer|No|单次请求返回K线数量，不传默认是300，limit不能超过1200，如果limit设置大于1200，只会返回1200条数据
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteKlineResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteKlineResponse.java)

返回数据可通过`QuoteKlineResponse.getQuoteKlineItems()`方法访问，返回`KlineItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.QuoteKlineItem` 属性如下:

字段|类型|说明
--- | --- | ---
symbol  |string|股票代码
period  |string|K线周期
items |array| K时数组,字段参考下面说明

其中K线数据items属性如下：

K线字段|类型|说明
---|---|---
close|double|收盘价
high|double|最高价
low|double|最低价
open|double|开盘价
time|long|时间
volume|long|成交量

具体字段可通过对象的get方法，如`getSymbol()`，进行访问

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteKlineResponse response = client.execute(QuoteKlineRequest.newRequest(List.of("AAPL"), KType.day, "2018-10-01", "2018-12-25")
        .withLimit(1000)
        .withRight(RightOption.br));
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
  "message": "success",
  "timestamp": 1525938835697,
  "data": {
    "symbol": "AAPL",
    "period": "day",
    "items": [
      {
        "time": 1523851200000,
        "volume": 21578420,
        "open": 175.02999877929688,
        "close": 175.82000732421875,
        "high": 176.19000244140625,
        "low": 174.8300018310547
      },
      {
        "time": 1523937600000,
        "volume": 26597070,
        "open": 176.49000549316406,
        "close": 178.24000549316406,
        "high": 178.93600463867188,
        "low": 176.41000366210938
      }
    ]
  }
}
```

### 获取逐笔成交 QuoteTradeTickRequest

**说明**

获取逐笔成交数据

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols|array|Yes|股票代码列表,支持陆股通股、港股、美股
limit|integer|No|数量，默认为2000
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteTradeTickResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteTradeTickResponse.java)

结构如下：
```java
public class QuoteTradeTickResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<TradeTickItem> tradeTickItems;
}
```

返回数据可通过`QuoteTradeTickResponse.getTradeTickItems()`方法访问，返回`TradeTickItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.TradeTickItem` 属性如下:

字段|类型|说明
--- | --- | ---
beginIndex|long|返回数据的实际开始索引
endIndex|long|返回数据的实际结束索引
symbol|string|请求的股票代码
items|List\<TickPoint\>|包含逐笔成交数据的List,单条逐笔成交数据保存在TickPoint对象中

其中`TickPoint`对象字段如下：

字段|类型|说明
--- | --- | ---
time  |long |交易时间戳
price |double |成交价
volume| long  |成交量
type  |string |*表示不变，+表示涨，-表示跌

具体字段可通过对象的get方法，如`getTime()`进行访问, 或通过对象的`toString()`方法转换为字符串

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
QuoteTradeTickResponse response = client.execute(QuoteTradeTickRequest.newRequest(List.of("AAPL")));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getTradeTickItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**
```json
{
  "code": 0,
  "message": "success",
  "timestamp": 1525938835697,
  "data": {
    "items": [
      {
        "time": 1452827921850,
        "price": 4.1,
        "volume": 300,
        "type": "*"
      },
      {
        "time": 1452827922850,
        "price": 4.12,
        "volume": 1200,
        "type": "+"
      },
      {
        "time": 1452827923850,
        "price": 4.08,
        "volume": 2000,
        "type": "-"
      }
    ],
    "beginIndex": 0,
    "endIndex": 3
  }
}
```

### 获取股票交易信息 QuoteStockTradeRequest

**说明**

获取股票交易所需的信息，包括每手股数，报价精度及股价最小变动单位

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols |array|Yes|股票代码列表，上限为：100

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteStockTradeResponse`[source](https://github.com/tigerfintech/openapi-java-sdk/blob/master/src/main/java/com/tigerbrokers/stock/openapi/client/https/response/quote/QuoteStockTradeResponse.java)

结构如下：
```java
public class QuoteStockTradeResponse extends TigerResponse {

  @JSONField(name = "data")
  private List<QuoteStockTradeItem> stockTradeItems;

  public List<QuoteStockTradeItem> getStockTradeItems() {
    return stockTradeItems;
  }
}
```

返回数据可通过`QuoteStockTradeResponse.getStockTradeItems()`方法访问，返回`QuoteStockTradeItem`对象，其中`com.tigerbrokers.stock.openapi.client.https.domain.quote.item.QuoteStockTradeItem` 属性如下:

名称|类型|说明
--- | --- | ---
symbol| String| 股票代码
lotSize|    Integer|    每手股数
spreadScale| Integer  | 报价精度
minTick|    Double  |股价最小变动单位

具体字段可通过对象的get方法，如`getSymbol()`进行访问, 或通过对象的`toString()`方法转换为字符串

**示例**

```java
List<String> symbols = new ArrayList<>();
symbols.add("00700");
symbols.add("00810");
QuoteStockTradeResponse response = client.execute(QuoteStockTradeRequest.newRequest(symbols));
if (response.isSuccess()) {
  System.out.println(Arrays.toString(response.getStockTradeItems().toArray()));
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**返回示例**

```json
{
  "code": 0,
  "data": [{
    "lotSize": 100,
    "minTick": 0.2,
    "spreadScale": 0,
    "symbol": "00700"
  }, {
    "lotSize": 6000,
    "minTick": 0.001,
    "spreadScale": 0,
    "symbol": "00810"
  }],
  "message": "success",
  "timestamp": 1546853907390
}
```