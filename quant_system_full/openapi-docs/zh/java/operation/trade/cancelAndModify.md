---
title: 取消或修改订单
---
### 修改订单 TigerHttpRequest(ApiServiceType.MODIFY_ORDER)

**说明**

修改订单

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes  |用户授权账户:DU575569
id            |long    |  Yes  |订单id，下单时返回
total_quantity|int     |  No  |订单数量(港股，沪港通，窝轮，牛熊证有最小数量限制)
order_type    |string  |  No  |订单类型. MKT（市价单）, LMT（限价单）, STP(止损单), STP_LMT(止损限价单), TRAIL(跟踪止损单)
limit_price   |double  |  No  |限价，当 order_type 为LMT,STP,STP_LMT时该参数必需
aux_price     |double  |  No  |股票止损价。当 order_type 为STP,STP_LMT时该参数必需
time_in_force |string  |  No  |订单有效期，只能是 DAY（当日有效）和GTC（取消前有效），默认为DAY
secret_key | string| No | 机构用户专用，交易员密钥

**返回**

名称 | 类型 | 说明
--- | --- | ---
id    | long | 唯一单号,可用于查询订单/修改订单/取消订单

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.MODIFY_ORDER);

String bizContent = TradeParamBuilder.instance()
    .account("DU575569")
    .id(147070683398615040L)
    .symbol("AAPL")
    .totalQuantity(200)
    .limitPrice(60.0)
    .orderType(OrderType.LMT)
    .action(ActionType.BUY)
    .secType(SecType.STK)
    .currency(Currency.USD)
    .outsideRth(false)
    .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
JSONObject data = JSON.parseObject(response.getData());
Long id = data.getLong("id");
```

**返回示例**
```json
{
  "code": 0,
  "message": null,
  "timestamp": 1525938835697,
  "data": {
      "id":147070683398615040
  }
}
```

### 取消订单 TigerHttpRequest(ApiServiceType.CANCEL_ORDER)

**说明**

取消订单

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account   | string  |  Yes  |用户授权账户:DU575569
id  | long    |  Yes  |订单号,下单时返回的ID
secret_key | string| No | 机构用户专用，交易员密钥

**返回**

名称 | 类型 | 说明
--- | --- | ---
id    | long | 唯一单号,可用于查询订单/修改订单/取消订单

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.CANCEL_ORDER);

String bizContent = TradeParamBuilder.instance()
    .account("DU575569")
    .id(147070683398615040L)
    .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
JSONObject data = JSON.parseObject(response.getData());
Long id = data.getLong("id");
```

**返回示例**
```json
{
  "code": 0,
  "message": null,
  "timestamp": 1525938835697,
  "data": {
      "id":147070683398615040
  }
}
```