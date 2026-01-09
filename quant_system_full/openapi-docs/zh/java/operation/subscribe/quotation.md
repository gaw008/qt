---
title: 行情订阅
---
### subscribeQuote(Set\<String\> symbols, QuoteKeyType quoteKeyType) 行情订阅

**说明**

订阅API，提供了行情数据订阅服务，可以实时获取行情的变化信息。
行情订阅推送接口是异步接口，通过实现ApiComposeCallback接口可以获得异步请求结果。
回调接口返回结果类型为JSONObject，类似于Map结构，可以通过key获取value。具体回调接口可参见此文档。

**输入参数**

| 参数         | 类型              | 是否必填 | 说明                                                         |
| ------------ | ----------------- | -------- | ------------------------------------------------------------ |
| symbols      | Set\<String\> | Yes      | 股票代码列表                                                 |
| quoteKeyType | QuoteKeyType      | No       | 行情类型有4种枚举类型：<br />QuoteKeyType.TRADE 成交数据。 包含的字段： open, latest_price, high, low, preClose, volume, timestamp, latestTime<br />QuoteKeyType.QUOTE 报价数据。 包含的字段: askPrice, askSize, bidPrice, bidSize, timestamp<br /><br />QuoteKeyType.TIMELINE 分时数据。 包含的字段: p: 最新价； a:当日截至当前的平均价; t:所属分钟的时间戳， v: 成交量。<br />QuoteKeyType.ALL 会同时推送上面几类数据 |
股票symbol格式： 如：`AAPL`, `00700`


**回调接口**

`void quoteChange(JSONObject jsonObject)`

**示例**

定义回调接口
```java
package com.tigerbrokers.stock.openapi.demo;

import com.alibaba.fastjson.JSONObject;
import com.tigerbrokers.stock.openapi.client.socket.ApiComposeCallback;
import com.tigerbrokers.stock.openapi.client.struct.SubscribedSymbol;

public class DefaultApiComposeCallback implements ApiComposeCallback {

  /*行情回调*/
  @Override
  public void quoteChange(JSONObject jsonObject) {
    System.out.println("quoteChange:" + jsonObject.toJSONString());
  }

  /*期权行情回调*/
  @Override
  public void optionChange(JSONObject jsonObject) {
    System.out.println("optionChange:" + jsonObject.toJSONString());
  }

  /*期货行情回调*/
  @Override
  public void futureChange(JSONObject jsonObject) {
    System.out.println("futureChange:" + jsonObject.toJSONString());
  }

  /*深度行情回调*/
  @Override
  public void depthQuoteChange(JSONObject jsonObject) {
    System.out.println("depthQuoteChange:" + jsonObject.toJSONString());
  }
  
  /*订阅成功回调*/
  @Override
  public void subscribeEnd(String id, String subject, JSONObject jsonObject) {
    System.out.println("subscribe " + subject + " end. id:" + id + ", " + jsonObject.toJSONString());
  }

  /*取消订阅回调*/
  @Override
  public void cancelSubscribeEnd(String id, String subject, JSONObject jsonObject) {
    System.out.println("cancel subscribe " + subject + " end. id:" + id + ", " + jsonObject.toJSONString());
  }

  /*查询已订阅symbol回调*/
  @Override
  public void getSubscribedSymbolEnd(SubscribedSymbol subscribedSymbol) {
    System.out.println(JSONObject.toJSONString(subscribedSymbol));
  }
}
```

进行订阅
```java
public class WebSocketDemo {

//实际订阅时需要填充tigerId和privateKey，并实现ApiComposeCallback接口，示例里面为DefaultApiComposeCallback
  private static ClientConfig clientConfig = TigerOpenClientConfig.getDefaultClientConfig();
  private static WebSocketClient client =
      WebSocketClient.getInstance().url(clientConfig.socketServerUrl)
        .authentication(ApiAuthentication.build(clientConfig.tigerId, clientConfig.privateKey))
        .apiComposeCallback(new DefaultApiComposeCallback());

  public static void subscribe() {

    client.connect();

    Set<String> symbols = new HashSet<>();

    //股票订阅
    symbols.add("AAPL");
    symbols.add("SPY");

    //期货订阅
    symbols.add("ESmain");
    symbols.add("ES1906");

    //期权的一种订阅方式
    symbols.add("TSLA 20190614 200.0 CALL");

    //期权另外一种订阅方式
    symbols.add("SPY   190508C00290000");

    //定义感兴趣的的行情字段
    List<String> focusKeys = new ArrayList<>();
    focusKeys.add("symbol");
    focusKeys.add("volume");
    focusKeys.add("askPrice");
    focusKeys.add("bidPrice");

    //订阅感兴趣字段
    //client.subscribeQuote(symbols, focusKeys);

    //订阅相关symbol
    client.subscribeQuote(symbols);

    //订阅深度数据
    client.subscribeDepthQuote(symbols);

    //查询订阅详情
    client.getSubscribedSymbols();

    //建议交易时间过后断开连接，断开连接时会自动注销之前的订阅记录
    //client.disconnect();
  }
}
```
**返回示例**




### subscribeOption(Set\<String\> symbols) 订阅期权行情
**说明**

订阅期权行情。
行情订阅推送接口是异步接口，通过实现ApiComposeCallback接口可以获得异步请求结果。
回调接口返回结果类型为JSONObject，类似于Map结构，可以通过key获取value。

**请求参数**

| 参数    | 类型              | 是否必填 | 说明         |
| ------- | ----------------- | -------- | ------------ |
| symbols | Set\<String\> | Yes      | 期权代码列表 |

期权symbol格式：期权symbol支持2种格式。 一种是symbol名称+到期日+行权价格+方向,以空格分隔。如：(AAPL 20190329 182.5 PUT)。 另一种为identifier，查询期权行情时返回该字段。如：(SPY 190508C00290000)

**回调接口**

`void optionChange(JSONObject jsonObject)`

**示例**
```java
Set<String> symbols = new HashSet<>();
//期权的一种订阅方式
symbols.add("TSLA 20190614 200.0 CALL");
//期权另外一种订阅方式
symbols.add("SPY 190508C00290000");

client.subscribeQuote(symbols);
// 或者
client.subscribeOption(symbols);
```
完整示例参见[行情订阅]()

**返回示例**

### subscribeFuture(Set\<String\> symbols) 订阅期货行情

**说明**

订阅期货行情。
行情订阅推送接口是异步接口，通过实现ApiComposeCallback接口可以获得异步请求结果。
回调接口返回结果类型为JSONObject，类似于Map结构，可以通过key获取value。


| 参数    | 类型              | 是否必填 | 说明         |
| ------- | ----------------- | -------- | ------------ |
| symbols | Set\<String\> | Yes      | 期货代码列表 |


**回调接口**
`void futureChange(JSONObject jsonObject)`

**示例**  
```java
Set<String> symbols = new HashSet<>();
//期货订阅
symbols.add("ESmain");
symbols.add("ES1906");

client.subscribeQuote(symbols);
// 或
client.subscribeFuture(symbols);
```
完整示例参见[行情订阅]()


**返回示例**  





### subscribeDepthQuote(Set\<String\> symbols) 订阅深度行情

**说明**

订阅多档深度行情。  
行情订阅推送接口是异步接口，通过实现ApiComposeCallback接口可以获得异步请求结果。
回调接口返回结果类型为JSONObject，类似于Map结构，可以通过key获取value。具体回调接口可参见此文档。

**请求参数** 

| 参数    | 类型              | 是否必填 | 说明         |
| ------- | ----------------- | -------- | ------------ |
| symbols | Set\<String\> | Yes      | 股票代码列表 |

**回调接口**

`void depthQuoteChange(JSONObject jsonObject)`

**示例**
```java
Set<String> symbols = new HashSet<>();
//期货订阅
symbols.add("AAPL");
symbols.add("JD");

client.subscribeQuote(symbols);
// 或
client.subscribeDepthQuote(symbols);
```
完整示例参见[行情订阅]()



### getSubscribedSymbols 查询已订阅标的

**说明**

查询已经订阅过的标的信息

**请求参数**

无

**回调接口**

`void getSubscribedSymbolEnd(SubscribedSymbol subscribedSymbol)`

**示例**
```java
client.getSubscribedSymbols();
```
完整示例参见[行情订阅]()

**返回示例**  


---