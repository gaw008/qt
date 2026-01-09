---
title: 基本功能示例
---
老虎Open API SDK提供了便捷地在自己的程序中调用老虎的服务的工具，本章节将对老虎API的核心功能进行一一演示：包括查询行情，订阅行情，以及调用API进行交易

## 查询行情
以下为一个最简单的调用老虎API的示例，演示了如何调用Open API来主动查询股票行情。接下来的例子分别演示了如何调用Open API来进行交易与订阅行情。除上述基础功能外，Open API还支持查询、交易多个市场的不同标的，以及其他复杂请求。对于其他Open API支持的接口和请求，请在快速入门后阅读文档正文获取列表及使用方法，并参考快速入门以及文档中的例子进行调用

*为方便直接复制运行，以下的说明采用注释的形式*

*请使用JUnit进行测试*

```java
import com.tigerbrokers.stock.openapi.client.config.ClientConfig;
import com.tigerbrokers.stock.openapi.client.https.client.TigerHttpClient;
import com.tigerbrokers.stock.openapi.client.https.request.quote.QuoteKlineRequest;
import com.tigerbrokers.stock.openapi.client.https.response.quote.QuoteKlineResponse;
import com.tigerbrokers.stock.openapi.client.struct.enums.KType;
import com.tigerbrokers.stock.openapi.client.struct.enums.RightOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TigerOpenClientConfig {
  static {
    ClientConfig clientConfig = ClientConfig.DEFAULT_CONFIG;
    clientConfig.tigerId = "your tiger id";
    clientConfig.defaultAccount = "your account"; 
    clientConfig.privateKey = "you private key string";

  }
  public static ClientConfig getDefaultClientConfig() {
    return ClientConfig.DEFAULT_CONFIG;
  }
}


public class Demo {

    private static TigerHttpClient client = new TigerHttpClient(TigerOpenClientConfig.getDefaultClientConfig());
    
    public void kline() {
      List<String> symbols = new ArrayList<>();
      symbols.add("AAPL");
      QuoteKlineResponse response =
          client.execute(QuoteKlineRequest.newRequest(symbols, KType.day, "2018-10-01", "2018-12-25")
              .withLimit(1000)
              .withRight(RightOption.br));
      if (response.isSuccess()) {
        System.out.println(Arrays.toString(response.getKlineItems().toArray()));
      } else {
        System.out.println("response error:" + response.getMessage());
      }
}

}
```
## 订阅行情

除了选择主动查询的方式（见快速入门-查询行情部分），Open API还支持订阅-接受推送的方式来接收行情等信息，具体见下例。需要注意的是，订阅推送相关的请求均为异步处理，故需要用户自定义回调函数，与中间函数进行绑定。某个事件发生，或有最新信息更新被服务器推送时，程序会自动调用用户自定义的回调函数并传入返回接口返回的数据，由用户自定义的回调函数来处理数据。

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
import com.tigerbrokers.stock.openapi.client.config.ClientConfig;
import com.tigerbrokers.stock.openapi.client.socket.ApiAuthentication;
import com.tigerbrokers.stock.openapi.client.socket.WebSocketClient;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class WebSocketDemo {

//实际订阅时需要填充tigerId和privateKey，并实现ApiComposeCallback接口，本示例中为DefaultApiComposeCallback，实现参考以上代码，仅将返回数据简单输出。实际可根据需要调整@Override下方的回调函数
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

## 下单
交易是Open API的另一个主要功能。此例展示了如何使用Open API对美股AAPL下限价单:
```java
import com.tigerbrokers.stock.openapi.client.config.ClientConfig;
import com.tigerbrokers.stock.openapi.client.constant.ApiServiceType;
import com.tigerbrokers.stock.openapi.client.https.client.TigerHttpClient;
import com.tigerbrokers.stock.openapi.client.https.request.TigerHttpRequest;
import com.tigerbrokers.stock.openapi.client.https.response.TigerHttpResponse;
import com.tigerbrokers.stock.openapi.client.struct.enums.*;
import com.tigerbrokers.stock.openapi.client.util.builder.TradeParamBuilder;

public class TigerOpenClientConfig {
  static {
    ClientConfig clientConfig = ClientConfig.DEFAULT_CONFIG;
    clientConfig.tigerId = "your tiger id";
    clientConfig.defaultAccount = "your account"; 
    clientConfig.privateKey = "you private key string";

  }
  public static ClientConfig getDefaultClientConfig() {
    return ClientConfig.DEFAULT_CONFIG;
  }
}

public class Demo {
  public void placeUSStockOrder() {

    private static TigerHttpClient client = new TigerHttpClient(TigerOpenClientConfig.getDefaultClientConfig());

    TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.PLACE_ORDER);
  
    String bizContent = TradeParamBuilder.instance()
        .account("DU575569")
        .symbol("AAPL")
        .secType(SecType.STK)
        .market(Market.US)
        .currency(Currency.USD)
        .action(ActionType.BUY)
        .orderType(OrderType.LMT)
        .limitPrice(182.0)
        .totalQuantity(100)
        .timeInForce(TimeInForce.DAY)
        .buildJson();
    request.setBizContent(bizContent);
  
    TigerHttpResponse response = client.execute(request);
    System.out.println(response)
  }
}
```