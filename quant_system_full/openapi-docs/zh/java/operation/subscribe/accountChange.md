---
title: 账户变动
---
### void subscribe(Subject subject, List\<String\> focusKeys) 订阅

**说明**

交易API提供了交易相关接口，同时提供了订阅接口，可以实时获取账户、资产、持仓的变化信息。
交易推送接口是异步接口，通过实现ApiComposeCallback接口可以获得异步请求结果。
回调接口返回值类型为JSONObject，类似于Map结构，可以通过key获取value。具体回调接口可参见此文档。

**输入参数(订阅主题)**

|参数         | 类型              | 是否必填 | 说明  |                                                    
| ------------ | ---------|-------- | ------ |
| subject | com.tigerbrokers.stock.openapi.client.struct.enums.Subject | Yes | 订阅主题 |
| focusKeys | List\<String\> | No | 指定订阅的字段 |

Subject为订阅主题，主要包括三种，OrderStatus(订单)，Asset(资产)，Position(持仓)：
- **OrderStatus推送**，是指订单状态发生变化时的推送，订单变化主要包括：Submitted(已提交到交易所)，Cancelled(订单已取消)，Inactive(订单被拒绝)，Filled(订单已成交)，其他中间状态时不会推送。
- **Asset推送**，是当账户资产发生变化时的推送。
- **Position推送**，是当账户持仓发生变化时的推送。

**回调数据字段含义**  
资产变动回调  

字段|类型|描述
----|----|----
segment|String|按交易品种划分的分类。S表示股票，C表示期货，summary表示环球账户的汇总资产信息
cash|double|现金额。当前所有币种的现金余额之和
availableFunds|double|可用资金，隔夜剩余流动性
excessLiquidity|double|当前剩余流动性
equityWithLoan|double|含贷款价值总权益。等于总资产 - 美股期权
netLiquidation|double|总资产(净清算值)。总资产就是我们账户的净清算现金余额和证券总市值之和
grossPositionValue|double|证券总价值
initMarginReq|double|初始保证金
maintMarginReq|double|维持保证金
buyingPower|double|购买力。仅适用于股票品种，即segment为S时有意义

持仓变动回调

字段|类型|描述
----|----|----
segment|String|按交易品种划分的分类。S表示股票，C表示期货
symbol|String|持仓标的代码，如 'AAPL', '00700', 'ES', 'CN'
identifier|String|标的标识符。股票的identifier与symbol相同。期货的会带有合约月份，如 'CN2201'
currency|String|币种。USD美元，HKD港币
secType|String|交易品种，标的类型。STK表示股票，FUT表示期货
latestPrice|double|标的当前价格
marketValue|double|持仓市值
quantity|int|持仓数量
averageCost|double|持仓均价
unrealizedPnl|double|持仓盈亏

订单变动回调

字段|类型|描述
----|----|----
segment|String|按交易品种划分的分类。S表示股票，C表示期货
id|long|订单号
symbol|String|持仓标的代码，如 'AAPL', '00700', 'ES', 'CN'
identifier|String|标的标识符。股票的identifier与symbol相同。期货的会带有合约月份，如 'CN2201'
currency|String|币种。USD美元，HKD港币
secType|String|交易品种，标的类型。STK表示股票，FUT表示期货
action|String|买卖方向。BUY表示买入，SELL表示卖出。
orderType|String|订单类型。'MKT'市价单/'LMT'限价单/'STP'止损单/'STP_LMT'止损限价单/'TRAIL'跟踪止损单
quantity|int|下单数量
limitPrice|double|限价单价格
filled|int|成交数量
avgFillPrice|double|成交均价
realized_pnl|double|已实现盈亏
status|String|订单状态
outsideRth|boolean|是否允许盘前盘后交易，仅适用于美股
openTime|long|下单时间


**回调接口**

实现 `ApiComposeCallback` 的对应方法
```
/** 
* 回调接口： 
* 根据不同subject调用不同接口
*/
void assetChange(JSONObject jsonObject)  //对应subject = asset
void positionChange(JSONObject jsonObject) //对应subject = position
void orderStatusChange(JSONObject jsonObject) //对应subject = orderstatus
```

**示例**

实现回调接口示例
```java
package com.tigerbrokers.stock.openapi.demo;

import com.alibaba.fastjson.JSONObject;
import com.tigerbrokers.stock.openapi.client.socket.ApiComposeCallback;

public class DefaultApiComposeCallback implements ApiComposeCallback {

  @Override
  public void orderStatusChange(JSONObject jsonObject) {
    StringBuilder builder = new StringBuilder();
    for (String key : jsonObject.keySet()) {
      builder.append(key).append("=").append(jsonObject.get(key)).append("|");
    }
    System.out.println("order change:" + builder);
  }

  @Override
  public void positionChange(JSONObject jsonObject) {
    StringBuilder builder = new StringBuilder();
    for (String key : jsonObject.keySet()) {
      builder.append(key).append("=").append(jsonObject.get(key)).append("|");
    }
    System.out.println("position change:" + builder);
  }

  @Override
  public void assetChange(JSONObject jsonObject) {
    StringBuilder builder = new StringBuilder();
    for (String key : jsonObject.keySet()) {
      builder.append(key).append("=").append(jsonObject.get(key)).append("|");
    }
    System.out.println("asset change:" + builder);
  }
  
  @Override
  public void subscribeEnd(String id, String subject, JSONObject jsonObject) {
    System.out.println("subscribe " + subject + " end. id:" + id + ", " + jsonObject.toJSONString());
  }
  
  @Override
  public void cancelSubscribeEnd(String id, String subject, JSONObject jsonObject) {
    System.out.println("cancel subscribe " + subject + " end. id:" + id + ", " + jsonObject.toJSONString());
  }
}
```

进行订阅
```java
public class WebSocketDemo {

//实际订阅时需要填充tigerId和privateKey，并实现callback接口
  private static ClientConfig clientConfig = TigerOpenClientConfig.getDefaultClientConfig();
  private static WebSocketClient client =
    WebSocketClient.getInstance().url(clientConfig.socketServerUrl)
      .authentication(ApiAuthentication.build(clientConfig.tigerId, clientConfig.privateKey))
      .apiComposeCallback(new DefaultApiComposeCallback());

  public static void subscribe() {
    //创建连接
    client.connect();

    //订阅 订单/资产/持仓
    client.subscribe(Subject.OrderStatus);    
    client.subscribe(Subject.Asset);
    client.subscribe(Subject.Position);

    //订阅感兴趣的订单字段
    List<String> focusKeys = new ArrayList<>();
    focusKeys.add("symbol");
    focusKeys.add("account");
    focusKeys.add("latestPrice");
    focusKeys.add("errorCode");

    //订阅感兴趣字段的订单变化
    //client.subscribe(Subject.OrderStatus, focusKeys);

    //循环等待
    sleep(60000)
    
    // 取消订阅
    client.cancelSubscribe(Subject.Asset);
    client.cancelSubscribe(Subject.Position);
    client.cancelSubscribe(Subject.OrderStatus);
    
    //非交易时间段建议关闭连接，会自动注销之前的全部订阅信息
    //client.disconnect();
  }
}
```
**返回示例**
```java
// asset change
equityWithLoan=51678.0|grossPositionValue=137118.74|type=asset|excessLiquidity=12017.47|availableFunds=7382.99|initMarginReq=45222.06|buyingPower=50767.25|cashBalance=161989.46|segment=summary|netLiquidation=52606.82|maintMarginReq=40589.35|account=DU575569|timestamp=1640159945678|

// position change
symbol=AAPL|latestPrice=166.35|identifier=AAPL|multiplier=1.0|marketValue=0.0|secType=STK|type=position|market=US|unrealizedPnl=0.0|segment=S|name=Apple Inc|currency=USD|positionScale=0|position=0|averageCost=0.0|account=402901|timestamp=1638345005587|

// order change
symbol=AAPL|orderType=MKT|secType=STK|source=OpenApi|type=orderstatus|filledQuantity=0|totalQuantityScale=0|totalQuantity=10|segment=S|action=BUY|currency=USD|id=24831040033915904|openTime=1638345001000|commissionAndFee=0.0|timestamp=1638345004783|identifier=AAPL|isLong=true|limitPrice=0.0|multiplier=1.0|errorMsg=Cancelled by system|market=US|outsideRth=false|avgFillPrice=0.0|stockId=14|name=Apple Inc|realizedPnl=0.0|account=402901|
```


### void cancelSubscribe(Subject subject) 取消订阅
**说明**

取消订阅账户变动的推送

**输入参数(退订主题)**

| 参数         | 类型       | 是否必填 | 说明  |
| ------------ | --------- | -------- | ------ |
| subject | com.tigerbrokers.stock.openapi.client.struct.enums.Subject | Yes | 订阅主题 |

Subject 为要取消订阅的主题，主要包括三种，OrderStatus(订单)，Asset(资产)，Position(持仓)
含义同订阅主题一致.

**示例**
```java
// 取消订阅
client.cancelSubscribe(Subject.Asset);
client.cancelSubscribe(Subject.Position);
client.cancelSubscribe(Subject.OrderStatus);
```