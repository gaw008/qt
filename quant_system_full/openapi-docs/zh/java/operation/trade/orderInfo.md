---
title: 获取订单信息
---
### 预览订单(仅环球账户) TigerHttpRequest(ApiServiceType.PREVIEW_ORDER)

**说明**

预览订单

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes  |用户授权账户:DU575569，目前仅支持环球账户
symbol        |string    |  Yes  |股票代码 如：AAPL
sec_type      |string    |  Yes  | 合约类型 (STK 股票 OPT 美股期权 WAR 港股窝轮 IOPT 港股牛熊证)
action        |string    |  Yes  |交易方向 BUY/SELL
order_type    |string  |  Yes  |订单类型. MKT（市价单）, LMT（限价单）, STP(止损单), STP_LMT(止损限价单), TRAIL(跟踪止损单)
total_quantity|int     |  Yes    |订单数量(港股，沪港通，窝轮，牛熊证有最小数量限制)
limit_price   |double  |  No     |限价，当 order_type 为LMT,STP,STP_LMT时该参数必需
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

**返回**

名称 | 类型 | 说明
--- | --- | ---
account|string|用户账户
status|string|根据订单参数判断真正下单后可能会出现的状态
initMargin|double|初始保证金金额,仅保证金账户有此属性
maintMargin|double|维持保证金金额,仅保证金账户有此属性
equityWithLoan|double|具有借贷价值的资产值, 对于现金账户 = cashBalance, 对于保证金账户 = cashBalance + grossPositionValue
initMarginBefore|double|下单前初始保证金
maintMarginBefore|double|下单前维持保证金
equityWithLoanBefore|double|下单前借贷资产值
marginCurrency|string|保证金账户主币种
commission|double|佣金金额
minCommission|double|当佣金无法确认时，计算出的最低佣金
maxCommission|double|当佣金无法确认时，计算出的最多佣金
commissionCurrency|double|佣金币种，比如账户主币种是港币，下美股订单时，佣金币种是USD
warningText|string|错误信息提示

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.PREVIEW_ORDER);

String bizContent = TradeParamBuilder.instance()
    .account("DU575569")
    .orderId(0)
    .symbol("AAPL")
    .totalQuantity(500)
    .limitPrice(61.0)
    .orderType(OrderType.LMT)
    .action(ActionType.BUY)
    .secType(SecType.STK)
    .currency(Currency.USD)
    .outsideRth(false)
    .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
JSONObject data = JSON.parseObject(response.getData());
BigDecimal equityWithLoan = data.getBigDecimal("equityWithLoan");
System.out.println(data)
```

**返回示例**

```json
{
	"account": "DU575569",
	"status": "Inactive",
	"initMargin": 73803.41,
	"maintMargin": 66088.28,
	"equityWithLoan": 48570.99,
	"initMarginBefore": 45604.94,
	"maintMarginBefore": 40601.39,
	"equityWithLoanBefore": 48570.99,
	"marginCurrency": "USD",
	"commissionCurrency": "",
	"warningText": "YOUR ORDER IS NOT ACCEPTED. IN ORDER TO OBTAIN THE DESIRED POSITION YOUR EQUITY 	WITH LOAN VALUE [47636.67 USD] MUST EXCEED THE INITIAL MARGIN [72635.51 USD]"
}
```

### 获取订单 TigerHttpRequest(ApiServiceType.ORDERS)

**说明**

获取订单

**参数**

获取指定单个订单 
参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account  | string |  Yes   | 用户授权账户:DU575569
id |  int |  Yes   | 下单成功后返回的订单号
secret_key | string| No | 机构用户专用，交易员密钥

获取订单列表 
参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes | 用户授权账户:DU575569
sec_type      |string  |  No  | ALL/STK/OPT/FUT/FOP/CASH 默认 ALL
market        |string  |  No  | ALL/US/HK/CN 默认 ALL（综合账户暂不支持）
symbol        |string  |  No  | 股票代码
start_date    |string  |  No  | '2018-05-01' 或者 "2018-05-01 10:00:00"（东八区），闭区间
end_date      |string  |  No  | '2018-05-15' 或者 "2018-05-01 10:00:00"（东八区），开区间
states        |array   |  No  | 订单状态, 默认查有效订单，参考:[订单状态](/zh/java/appendix2/#订单状态)
isBrief       |boolean |  No  | 是否返回精简的订单信息
limit         |integer |  No  | 默认：100, 最大限制: 300
secret_key | string| No | 机构用户专用，交易员密钥


**返回**

`com.tigerbrokers.stock.openapi.client.https.response.TigerHttpResponse`

名称 | 示例 | 说明
--- | --- | ---
id|135482687464472583|订单全局唯一订单号
orderId|1000003917|用户的自增单号，非全局唯一
parentId|0|母定单的定单代号，用于自动跟踪止损单
account|DU575569|交易账户
action|BUY|交易方向,BUY or SELL
orderType|LMT|订单类型
limitPrice|36.91|现价单价格
auxPrice|0.0|止损单辅助价格-跟踪额
trailingPercent|5|跟踪止损单的跟踪百分比
totalQuantity|50|总数
timeInForce|DAY|DAY/GTC
outsideRth|false|是否允许盘前、盘后
filledQuantity|100|成交数量
lastFillPrice|0.0|最后执行价
avgFillPrice|12.0|包含佣金的平均成交价
liquidation|false|是否清算
remark|订单号错误|错误描述
status|Filled|订单状态,参考:[订单状态](/zh/java/appendix2/#订单状态)
attrDesc|Exercise|订单描述信息,参考:[订单描述](/zh/java/appendix2/#订单描述)
commission|0.99|包含佣金、印花税、证监会费等系列费用
commissionCurrency|USD|佣金币种
realizedPnl|0.0|已实现盈亏
percentOffset|0.0|相对订单的百分比抵消额
openTime|2019-01-01|下单时间
tradeTime|2019-01-01|最新成交时间
latestTime|2019-01-01|最新状态更新时间
symbol|JD|股票代码
currency|USD|货币
market|US|交易市场
multiplier|0.0|每手股数
secType|STK|交易类型

**示例**

获取单个订单
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ORDERS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .id(147070683398615040L)
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);


JSONObject data = JSON.parseObject(response.getData());
Long id = data.getLong("id");
String action = data.getString("action");

```

获取订单列表
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ORDERS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .secType(SecType.STK)
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

JSONArray orders = JSONObject.parseObject(response.getData()).getJSONArray("items");
JSONObject order1 = orders.getJSONObject(0);
String symbol = order1.getString("symbol");
Long id = order1.getLong("id");

```

**返回示例**

单个订单
```json
{
	"code": 0,
	"message": "success",
	"data": {
		"id":135482687464472583,//全局唯一订单ID
		"account": "DU575569", //交易账户
		"action": "BUY", //
		"auxPrice": 0.0, //止损单辅助价格-跟踪额
		"avgFillPrice": 0.0, //买入价格
		"currency": "USD",
		"filledQuantity": 0, //执行数目
		"lastFillPrice": 0.0, //最后执行价
		"limitPrice": 36.91, //现价单价格
		"market": "US",
		"multiplier": 0.0,
		"orderId": 1000003917,//用户的订单自增ID，全局不唯一
		"orderType": "LMT", 
		"outsideRth": false, //是否允许盘前、盘后
		"realizedPnl": 0.0, //实际盈亏
		"secType": "STK",
		"status": "Filled",
		"symbol": "JD",
		"timeInForce": "DAY", //DAY/GTC
		"totalQuantity": 50 //总数
	},
	"timestamp": 1527830042620
}
```

订单列表

```json
{
	"code": 0,
	"message": "success",
	"data":{
		"items":[{
			"account": "DU575569", //交易账户
			"id":135482687464472583,//全局唯一订单ID
			"action": "BUY", //
			"auxPrice": 0.0, //止损单辅助价格-跟踪额
			"avgFillPrice": 0.0, //买入价格
			"currency": "USD",
			"filledQuantity": 0, //执行数目
			"limitPrice": 36.91, //现价单价格
			"market": "US",
			"multiplier": 0.0,
			"orderId": 1000003917,
			"orderType": "LMT", 
			"outsideRth": false, //是否允许盘前、盘后
			"realizedPnl": 0.0, //实际盈亏
			"secType": "STK",
			"status":3,
			"symbol": "JD",
			"timeInForce": "DAY", //DAY/GTC
			"totalQuantity": 50 //总数
		}, {
			"account": "DU575569",
			"id":135482687464472583,//全局唯一订单ID
			"action": "BUY",
			"auxPrice": 0.0,
			"avgFillPrice": 0.0,
			"currency": "USD",
			"filledQuantity": 0,
			"limitPrice": 19.21,
			"market": "US",
			"multiplier": 0.0,
			"orderId": 1000003874,
			"orderType": "LMT",
			"outsideRth": true,
			"realizedPnl": 0.0,
			"secType": "STK",
			"symbol": "JKS",
			"timeInForce": "DAY",
			"totalQuantity": 4858,
			"trailStopPrice": 0.0, //跟踪止损单价格
			"trailingPercent": 0.0 
		}]
	},
	"timestamp": 1527830042620
}
```

### 获取已成交订单列表 TigerHttpRequest(ApiServiceType.FILLED_ORDERS)

**说明**

获取状态为成交的订单列表

**参数**

参考获取订单，其中start_date和end_date为必传参数。


**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.FILLED_ORDERS);

String bizContent = AccountParamBuilder.instance()
        .account("402901")
        .secType(SecType.STK)
        .startDate("2021-11-15 22:34:30")
        .endDate("2022-01-06 22:34:31")
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
```


**返回**

参考获取订单


### 获取待成交订单列表 TigerHttpRequest(ApiServiceType.ACTIVE_ORDERS)

**参数**

参考获取订单

**示例**

```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ACTIVE_ORDERS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .secType(SecType.STK)
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
```

**返回**

参考获取订单

### 获取已撤销订单列表 TigerHttpRequest(ApiServiceType.INACTIVE_ORDERS)
**参数**

参考获取订单

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.INACTIVE_ORDERS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .secType(SecType.STK)
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
```
**返回**

参考获取订单


### 获取成交记录 TigerHttpRequest(ApiServiceType.ORDER_TRANSACTIONS)

**说明**  

获取订单的成交记录


**参数**
参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account | String | Yes | 账户，目前仅支持综合账户
orderId | long | Yes |  订单ID。 order_id 和 symbol其中一个必传。 使用orderId后，symbol参数不生效, 
symbol | String | Yes | 股票代码。order_id 和 symbol其中一个必传。
secType | String | No, 指定symbol查询时必传 | STK:股票/FUT:期货/OPT:期权/WAR:涡轮/IOPT:牛熊证, 未指定查全部。
expiry | String | No, sect_type为OPT/WAR/IOPT类型时必传 | 到期日
right | String | No, sect_type为OPT/WAR/IOPT类型时必传 | CALL/PUT
startDate | String | No | 起始日期，东八区
endDate | String | No | 截止日期，东八区
limit | int | No | 返回数据数量限制，默认20, 最大100
secretKey | String | No | 机构用户专用，交易员密钥

**返回**

字段 | 示例 | 说明
--- | --- | ---
id | 24653027221308416 | 成交记录ID
accountId | 402190 | 账号
orderId | 24637316162520064 | 订单ID
secType | STK | 证券类型
symbol | CII | symbol
currency | USD | 币种
market | US | 市场
action | BUY | 动作, BUY/SELL
filledQuantity | 100 | 成交数量
filledPrice | 21 | 成交价
filledAmount | 2167.0 | 成交金额
transactedAt | 2021-11-15 22:34:30 | 成交时间

**示例**
```java
// 按照symbol查询
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ORDER_TRANSACTIONS);
String bizContent = AccountParamBuilder.instance()
    .account("402501")
    .secType(SecType.STK)
    .symbol("CII")
    .limit(30)
    .startDate("2021-11-15 22:34:30")
    .endDate("2021-11-15 22:34:31")
    .buildJson();
request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

JSONArray data = JSON.parseObject(response.getData()).getJSONArray("items");
JSONObject trans1 = data.getJSONObject(0);
 
 
// 按照orderId查询
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ORDER_TRANSACTIONS);
    String bizContent = AccountParamBuilder.instance()
        .account("402501")
        .orderId(24637316162520064L)
        .limit(30)
        .buildJson();
request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

JSONArray data = JSON.parseObject(response.getData()).getJSONArray("items");
JSONObject trans1 = data.getJSONObject(0);

```

**返回示例**
```json
{
	"items": [
		{
			"id": 24653027221308416,
			"accountId": 402901,
			"orderId": 24637316162520064,
			"secType": "STK",
			"symbol": "CII",
			"currency": "USD",
			"market": "US",
			"action": "BUY",
			"filledQuantity": 100,
			"filledPrice": 21.67,
			"filledAmount": 2167,
			"transactedAt": "2021-11-15 22:34:30"
		}
	]
}
```