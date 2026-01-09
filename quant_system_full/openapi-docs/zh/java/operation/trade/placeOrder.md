---
title: 下单交易
---
## 创建订单 TradeOrderRequest

### 说明

交易下单接口。关于如何选择标的、订单类型、方向数量等，请见下方说明。**请在运行程序前结合本文档的[概述](/zh/java/overview/introduction.html)部分及[FAQ-交易-支持的订单列表](/zh/java/FAQ/trade.html#支持交易的订单类型)部分，检查您的账户是否支持所请求的订单，并检查交易规则是否允许在程序运行时段对特定标的下单**。若下单失败，可首先阅读文档[FAQ-交易](/zh/java/FAQ/trade.html#下单失败排查方法)部分排查

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes  |用户授权账户:DU575569
order_id      |int     |  No   |订单号，作用是防止重复下单。可以通过订单号接口获取。如果传0，则服务器端会自动生成订单号，传0时无法防止重复下单，请谨慎选择
symbol        |string    |  Yes  |股票代码 如：AAPL；（sec_typ为窝轮牛熊证时,在app窝轮/牛熊证列表中名称下面的5位数字）
sec_type      |string    |  Yes  | 合约类型 (STK 股票 OPT 美股期权 WAR 港股窝轮 IOPT 港股牛熊证 FUT 期货)
action        |string    |  Yes  |交易方向 BUY/SELL
order_type    |string  |  Yes  |订单类型. MKT（市价单）, LMT（限价单）, STP(止损单), STP_LMT(止损限价单), TRAIL(跟踪止损单)
total_quantity|int     |  Yes    |订单数量(港股，沪港通，窝轮，牛熊证有最小数量限制)
limit_price   |double  |  No     |限价，当 order_type 为LMT,STP_LMT时该参数必需
aux_price     |double  |  No     |股票止损价。当 order_type 为STP,STP_LMT时该参数必需，当 order_type 为 TRAIL时，为跟踪额 
trailing_percent|double | No | 跟踪止损单-百分比 ，当 order_type 为 TRAIL时,aux_price和trailing_percent两者互斥
outside_rth   |boolean |  No   |true: 允许盘前盘后交易(美股专属), false: 不允许, 默认允许
market        |string    |  No   |市场 (美股 US 港股 HK 沪港通 CN)
currency      |string  |    No   |货币(美股 USD 港股 HKD 沪港通 CNH)
time_in_force |string  |  No  |订单有效期，只能是 DAY（当日有效）和GTC（取消前有效），默认为DAY
exchange      |string  |    No   |交易所 (美股 SMART 港股 SEHK 沪港通 SEHKNTL 深港通 SEHKSZSE)
expiry        |string    |  No   |过期日(期权、窝轮、牛熊证专属)
strike        |string  |    No   |底层价格(期权、窝轮、牛熊证专属)
right         |string  |    No   |期权方向 PUT/CALL(期权、窝轮、牛熊证专属)
multiplier    |float   |    No   |1手单位(期权、窝轮、牛熊证专属)
local_symbol  |string  |    No   |窝轮牛熊证该字段必填,在app窝轮/牛熊证列表中名称下面的5位数字
secret_key | string| No | 机构用户专用，交易员密钥

附加订单参数

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
attach_type    |string  |   No  |附加订单类型，下附加订单时必填(order_type应为LMT): PROFIT-止盈单,LOSS-止损单,BRACKETS-括号订单（包含止盈单和止损单）
profit_taker_orderId |int|   No  |止盈单号，可以通过订单号接口获取。如果传0，则服务器端会自动生成止盈单号
profit_taker_price |double|  No  |止盈单价格，下止盈单时必填
profit_taker_tif |string |  No   |同time_in_force字段，下止盈单时必填
profit_taker_rth |boolean| No |同outside_rth字段
stop_loss_orderId   |int | No |止损单号，可以通过订单号接口获取。如果传0，则服务器端会自动生成止损单号
stop_loss_price  |double | No |止损单价格，下止损单时必填
stop_loss_tif    |string | No |同time_in_force字段，下止损单时必填

**返回**

名称 | 类型 | 说明
--- | --- | ---
id    | long | 唯一单号,可用于查询订单/修改订单/取消订单

### 构建合约对象

```java
// 美股股票合约
ContractItem contract = ContractItem.buildStockContract("SPY", "USD");

// 港股股票合约
ContractItem contract = ContractItem.buildStockContract("00700", "HKD");

// 港股窝轮合约（需要注意同一个symbol，环球账号和综合账号的expiry可能不同）
ContractItem contract = ContractItem.buildWarrantContract("13745", "20211217", 719.38D, Right.CALL.name());
// 港股牛熊证合约
ContractItem contract = ContractItem.buildCbbcContract("50296", "20220331", 457D, Right.CALL.name());

// 美股期权合约
ContractItem contract = ContractItem.buildOptionContract("AAPL  190118P00160000");
ContractItem contract = ContractItem.buildOptionContract("AAPL", "20211119", 150.0D, "CALL");

// 期货合约

// 环球账户
ContractItem contract = ContractItem.buildFutureContract("CL", "USD", "SGX", "20190328", 1.0D);

// 综合账户
ContractItem contract = ContractItem.buildFutureContract("CL2112", "USD");
```



### 市价单（MKT）

```java
// get contract(use default account)
ContractRequest contractRequest = ContractRequest.newRequest(
  new ContractModel("AAPL"));
ContractResponse contractResponse = client.execute(contractRequest);
ContractItem contract = contractResponse.getItem();
// market order(use default account)
TradeOrderRequest request = TradeOrderRequest.buildMarketOrder(contract, ActionType.BUY, 10);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// get contract(use account parameter)
ContractRequest contractRequest = ContractRequest.newRequest(
  new ContractModel("AAPL"), "402901");
ContractResponse contractResponse = client.execute(contractRequest);
ContractItem contract = contractResponse.getItem();
// market order(use account parameter)
request = TradeOrderRequest.buildMarketOrder("402901", contract, ActionType.BUY, 10);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 限价单（LMT）

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildLimitOrder(
  contract, ActionType.BUY, 1, 100.0d);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildLimitOrder(
  "402901", contract, ActionType.BUY, 1, 100.0d);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 止损单（STP）

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildStopOrder(
  contract, ActionType.BUY, 1, 120.0d);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildStopOrder(
  "402901", contract, ActionType.BUY, 1, 120.0d);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 止损限价单(STP_LMT)

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildStopLimitOrder(
  contract, ActionType.BUY, 1,
  150d,130.0d);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildStopLimitOrder(
  "402901", contract, ActionType.BUY, 1,
  150d,130.0d);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 跟踪止损单(TRAIL)

```json
// use default account
TradeOrderRequest request = TradeOrderRequest.buildTrailOrder(
  contract, ActionType.BUY, 1,
  10d,130.0d);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter. standard account currently not supported
request = TradeOrderRequest.buildTrailOrder(
  "402901", contract, ActionType.BUY, 1,
  10d,130.0d);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 主订单+附加止盈单

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildLimitOrder(
  contract, ActionType.BUY, 1, 199d);
TradeOrderRequest.addProfitTakerOrder(request, 250D, TimeInForce.DAY, Boolean.FALSE);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildLimitOrder(
  "402901", contract, ActionType.BUY, 1, 199d);
TradeOrderRequest.addProfitTakerOrder(request, 250D, TimeInForce.DAY, Boolean.FALSE);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 主订单+附加止损单

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildLimitOrder(
  contract, ActionType.BUY, 1, 129d);
TradeOrderRequest.addStopLossOrder(request, 100D, TimeInForce.DAY);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildLimitOrder(
  "402901", contract, ActionType.BUY, 1, 129d);
// 添加附加止损市价单，附加止损价格是触发价(不支持期权标的)
TradeOrderRequest.addStopLossOrder(request, 100D, TimeInForce.DAY);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// 期权可以使用附加止损限价单
ContractItem optionContract = ContractItem.buildOptionContract("AAPL",
                                                               "20211231", 175.0D, "CALL");
request = TradeOrderRequest.buildLimitOrder(
  "402901", optionContract, ActionType.BUY, 1, 2.0d);
// 添加附加止损限价单，其中第一个价格是触发价，第二个价格是附加止损单的挂单限价(暂只支持综合账号)
TradeOrderRequest.addStopLossLimitOrder(request, 1.7D, 1.69D, TimeInForce.DAY);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```

### 主订单+附加括号订单

```java
// use default account
TradeOrderRequest request = TradeOrderRequest.buildLimitOrder(
  contract, ActionType.BUY, 1, 199d);
TradeOrderRequest.addBracketsOrder(request, 250D, TimeInForce.DAY, Boolean.FALSE,
                                   180D, TimeInForce.GTC);
TradeOrderResponse response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));

// use account parameter
request = TradeOrderRequest.buildLimitOrder(
  "DU575569", contract, ActionType.BUY, 1, 199d);
TradeOrderRequest.addBracketsOrder(request, 250D, TimeInForce.DAY, Boolean.FALSE,
                                   180D, TimeInForce.GTC);
response = client.execute(request);
System.out.println(JSONObject.toJSONString(response));
```