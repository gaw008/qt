---
title: 查询账户信息
---
### 账户列表 TigerHttpRequest(ApiServiceType.ACCOUNTS)

**说明**

获取管理的账户列表

**参数**
参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account |string  |  No |用户授权账号（不支持传入模拟账号），不传会返回全部账号，包括环球，标准，模拟。

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.TigerHttpResponse`
具体数据为json格式，解析步骤参照下方示例代码。  

名称 | 示例 | 说明
--- | --- | ---
account|DU575569|交易账户
capability|MRGN|账户类型(CASH:现金账户, RegTMargin: Reg T 保证金账户, PMGRN: 投资组合保证金)
status|Funded|状态(New, Funded, Open, Pending, Abandoned, Rejected, Closed, Unknown)
accountType|STANDARD|账户分类, GLOBAL环球；STANDARD综合

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ACCOUNTS);

String bizContent = AccountParamBuilder.instance()
        .account("123456")
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

# 获取具体字段数据
JSONArray accounts = JSON.parseObject(response.getData()).getJSONArray("items");
JSONObject account1 = accounts.getJSONObject(0);
String capability = account1.getString("capability");
String accountType = account1.getString("accountType");
String account = account1.getString("account");
String status = account1.getString("status");
```

**返回示例**

```json
{
	"code": 0,
	"message": "success",
	"data":{
		"items":[{
				"account": "123456",
				"capability": "RegTMargin",
				"status": "Funded",
				"accountType": "STANDARD",
		}]
	}
}
```

### 账户持仓 TigerHttpRequest(ApiServiceType.POSITIONS)

**说明**

查询账户持仓

**参数**  

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string   |  Yes  |用户授权账户:DU575569
sec_type      |string   |  No   |证券类型，包括：STK/OPT/FUT， 默认 STK
currency      |string  |  No   |货币类型，包括：ALL/USD/HKD/CNH 默认 ALL
market        |string  |  No   |市场分类，包括：ALL/US/HK/CN 默认 ALL
symbol        |string  |  No  |股票代码 如：600884 / SNAP,期货类型时：CL1901，期权类型时 symbol格式为identifier，如：'AAPL  190111C00095000'，格式固定为21位，symbol为6位，不足的补空格 
secret_key | string| No | 机构用户专用，交易员密钥

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.TigerHttpResponse`
具体数据为json格式，解析步骤参照下方示例代码。 

名称 | 类型 | 说明
--- | --- | ---
account|string|交易账户
position|int|持仓数量
averageCost|double|平均成本
marketPrice|double|市价
marketValue|double|市值
realizedPnl|double|已实现盈亏
unrealizedPnl|double|浮动盈亏
salable|double|A股当天可卖数量
currency|string|交易货币币种
expiry|string|期权过期日
multiplier|double|每手数量
right|string|期权方向
secType|string|交易类型
strike|double|期权底层价格
symbol|string|股票代码

**示例**
```java
TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.POSITIONS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .secType(SecType.STK)
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

# 解析具体字段
JSONArray positions = JSON.parseObject(response.getData()).getJSONArray("items");
JSONObject position1 = positions.getJSONObject(0);
String account = position1.getString("account");
Double averageCost = position1.getDouble("averageCost");
String symbol = position1.getString("symbol");
```

**返回示例**
```json
{
	"code": 0,
	"message": "success",
	"data": {
		"items": [{
			"account": "DU575569",
			"averageCost": 111.3035,
			"contractId": 4347,
			"currency": "USD",
			"latestPrice": 91.83,
			"localSymbol": "ALB",
			"marketValue": 91.83,
			"multiplier": 0.0,
			"position": 1,
			"preClose": 0.0,
			"realizedPnl": 0.0,
			"secType": "STK",
			"status": 0,
			"stockId": "ALB.US",
			"symbol": "ALB",
			"unrealizedPnl": -18.97
		},
		{
			"account": "DU575569",
			"averageCost": 161.8235,
			"contractId": 6497,
			"currency": "USD",
			"latestPrice": 170.78,
			"localSymbol": "MCO",
			"marketValue": 170.78,
			"multiplier": 0.0,
			"position": 1,
			"preClose": 0.0,
			"realizedPnl": 0.0,
			"secType": "STK",
			"status": 0,
			"stockId": "MCO.US",
			"symbol": "MCO",
			"unrealizedPnl": 0.83
		}]
	},
	"timestamp": 1527830042620
}
```

### 环球账户资产 TigerHttpRequest(ApiServiceType.ASSETS)

**说明**

获取环球账户资产
主要适用于环球账户，综合/模拟账号虽然也可使用此接口，但有较多字段为空，请使用 [PrimeAssetRequest](/zh/java/operation/trade/accountInfo.html#综合-模拟账号获取资产-primeassetrequest) 查询综合/模拟账户资产

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account|string|Yes|用户授权账户:DU575569
segment|boolean|No|是否包含证券/期货分类， 默认 False
market_value|boolean|No|是否包含分市场市值，默认 False，仅环球账户支持
secret_key|string|No|机构用户专用，交易员密钥

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.TigerHttpResponse` 具体数据为json格式，解析步骤参照下方示例代码。 

名称 | 示例 | 说明
--- | --- | ---
account|DU575569|交易账户
capability|RegTMargin|账户类型，保证金:RegTMargin，现金:Cash
netLiquidation|1233662.93|净清算值
equityWithLoan|1233078.69|含借贷值股权(含贷款价值资产)。 证券 Segment: 现金价值 + 股票价值， 期货 Segment: 现金价值 - 维持保证金
initMarginReq|292046.91|初始保证金要求
maintMarginReq|273170.84|维持保证金要求
availableFunds|941031.78|可用资金(可用于交易)，计算方法为 equity_with_loan - initial_margin_requirement
dayTradesRemaining|-1|剩余日内交易次数，-1表示无限制
excessLiquidity|960492.09|剩余流动性，用来表示日内风险数值。 证券 Segment 计算方法: equity_with_loan - maintenance_margin_requirement。 期货 Segment 计算方法: net_liquidation - maintenance_margin_requirement
buyingPower|6273545.18|购买力。 预估您还可以购入多少美元的股票资产。保证金账户日内最多有四倍于资金（未被占用做保证金的资金）的购买力。隔夜最多有两倍的购买力。
cashValue|469140.39|证券账户金额+期货账户金额
accruedCash|-763.2|当前月份的累积应付利息，按照日频更新。
accruedDividend|0.0|累计分红. 指的是所有已执行但仍未支付的分红累加值
grossPositionValue|865644.18|证券总价值: 做多股票的价值+做空股票价值+做多期权价值+做空期权价值。
SMA|0.0|特殊备忘录账户，隔夜风险数值（App）
regTEquity|0.0|仅针对证券Segment，即根据 Regulation T 法案计算的 equity with loan（含借贷股权值）
regTMargin|0.0|仅针对证券Segment， 即根据 Regulation T 法案计算的 initial margin requirements（初始保证金）
cushion|0.778569|剩余流动性占总资产的比例，计算方法为: excess_liquidity/net_liquidation
currency|USD|货币币种
realizedPnl|-248.72|实际盈亏
unrealizedPnl|-17039.09|浮动盈亏
updateTime|0|更新时间
**segments**||按照交易品种区分的账户信息。内容是一个Map，分别有两个key，'S'表示证券， 'C' 表示期货； value 是一个 Account 对象。
**marketValues**||分市场的市值信息。内容是一个Map， 'USD' 表示美国市场， 'HKD' 表示香港市场; value 是一个 MarketValue 对象。

**``segments``说明：**

名称 | 示例 | 说明
--- | --- | ---
account|DU575569|交易账户
category|S |底层证券的行业分类 C(US Commodities 期货) or S(US Securities 证券)
title|标题|标题
netLiquidation|1233662.93|净清算值
cashValue|469140.39|证券账户金额+期货账户金额
availableFunds|941031.78|可用资金(可用于交易)
equityWithLoan|1233078.69|含借贷值股权
excessLiquidity|960492.09|剩余流动性，为保持当前拥有的头寸，必须维持的缓冲保证金的数额，日内风险数值（App）
accruedCash|-763.2|净累计利息
accruedDividend|0.0|净累计分红
initMarginReq|292046.91|初始保证金要求
maintMarginReq|273170.84|维持保证金要求
regTEquity|0.0|RegT资产
regTMargin|0.0|RegT保证金
SMA|0.0|特殊备忘录账户，隔夜风险数值（App）
grossPositionValue|865644.18|持仓市值
leverage|1|杠杆
updateTime|1526368181000|更新时间

**``marketValues``说明：**

名称 | 示例 | 说明
--- | --- | ---
account|DU575569|交易账户
currency|USD|货币币种
netLiquidation|1233662.93|总资产(净清算价值)
cashBalance|469140.39|现金
exchangeRate|0.1273896|对账户主币种的汇率
netDividend|0.0|应付股息与应收股息的净值
futuresPnl|0.0|盯市盈亏
realizedPnl|-248.72|已实现盈亏
unrealizedPnl|-17039.09|浮动盈亏
updateTime|1526368181000|更新时间
stockMarketValue|943588.78|股票市值
optionMarketValue|0.0|期权市值
futureOptionValue|0.0|期货市值
warrantValue|10958.0|窝轮市值


**示例**

```java

TigerHttpClient client = new TigerHttpClient(serverUrl, tigerId, privateKey);
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.ASSETS);

String bizContent = AccountParamBuilder.instance()
        .account("DU575569")
        .buildJson();

request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);

# 解析具体字段
JSONArray assets = JSON.parseObject(response.getData()).getJSONArray("items");
JSONObject asset1 = assets.getJSONObject(0);
String account = asset1.getString("account");
Double cashBalance = asset1.getDouble("cashBalance");
JSONArray segments = asset1.getJSONArray("segments");
JSONObject segment = segments.getJSONObject(0);
String category = segment.getString("category"); // "S" 股票， "C" 期货
```

**返回示例**

```json

{
	"code": 0,
	"message": "success",
	"data": {
		"items": [{
			"account": "DU575569",
			"accruedCash": -763.2,
			"accruedDividend": 0.0,
			"availableFunds": 941031.78,
			"buyingPower": 6273545.18,
			"capability": "Reg T Margin",
			"cashBalance": 469140.39,
			"cashValue": 469140.39,
			"currency": "USD",
			"cushion": 0.778569,
			"dayTradesRemaining": -1,
			"equityWithLoan": 1233078.69,
			"excessLiquidity": 960492.09,
			"grossPositionValue": 865644.18,
			"initMarginReq": 292046.91,
			"maintMarginReq": 273170.84,
			"netLiquidation": 1233662.93,
			"netLiquidationUncertainty": 583.55,
			"previousEquityWithLoanValue": 1216291.68,
			"previousNetLiquidation": 1233648.34,
			"realizedPnl": -31.68,
			"unrealizedPnl": 1814.01,
			"regTEquity": 0.0,
			"regTMargin": 0.0,
			"SMA": 0.0,
			"segments": [{
				"account": "DU575569",
				"accruedDividend": 0.0,
				"availableFunds": 65.55,
				"cashValue": 65.55,
				"category": "S",
				"equityWithLoan": 958.59,
				"excessLiquidity": 65.55,
				"grossPositionValue": 893.04,
				"initMarginReq": 893.04,
				"leverage": 0.93,
				"maintMarginReq": 893.04,
				"netLiquidation": 958.59,
				"previousDayEquityWithLoan": 969.15,
				"regTEquity": 958.59,
				"regTMargin": 446.52,
				"sMA": 2172.47,
				"title": "US Securities",
				"tradingType": "STKMRGN",
				"updateTime": 1541124813
			}],
			"marketValues": [{
				"account": "DU575569",
				"accruedCash": 0.0,
				"cashBalance": -943206.03,
				"currency": "HKD",
				"exchangeRate": 0.1273896,
				"futureOptionValue": 0.0,
				"futuresPnl": 0.0,
				"netDividend": 0.0,
				"netLiquidation": 11223.29,
				"optionMarketValue": 0.0,
				"realizedPnl": -248.72,
				"stockMarketValue": 943588.78,
				"unrealizedPnl": -17039.09,
				"updateTime": 1526368181000,
				"warrantValue": 10958.0
			},{
				"account": "DU575569",
				"accruedCash": 0.0,
				"cashBalance": -1635.23,
				"currency": "GBP",
				"exchangeRate": 1.35566495,
				"futureOptionValue": 0.0,
				"futuresPnl": 0.0,
				"netDividend": 0.0,
				"netLiquidation": 170.39,
				"optionMarketValue": 0.0,
				"realizedPnl": 0.0,
				"stockMarketValue": 1805.62,
				"unrealizedPnl": 177.58,
				"updateTime": 1526368181000,
				"warrantValue": 0.0
			},{
				"account": "DU575569",
				"accruedCash": 0.0,
				"cashBalance": 703542.12,
				"currency": "USD",
				"exchangeRate": 1.0,
				"futureOptionValue": 0.0,
				"futuresPnl": 0.0,
				"netDividend": 0.0,
				"netLiquidation": 1208880.15,
				"optionMarketValue": -64.18,
				"realizedPnl": 0.0,
				"stockMarketValue": 505780.03,
				"unrealizedPnl": 19886.87,
				"updateTime": 1526359227000,
				"warrantValue": 0.0
			}, {
				"account": "DU575569",
				"accruedCash": 0.0,
				"cashBalance": -714823.64,
				"currency": "CNH",
				"exchangeRate": 0.1576904,
				"futureOptionValue": 0.0,
				"futuresPnl": 0.0,
				"netDividend": 0.0,
				"netLiquidation": 142250.72,
				"optionMarketValue": 0.0,
				"realizedPnl": 0.0,
				"stockMarketValue": 859152.75,
				"unrealizedPnl": -102371.43,
				"updateTime": 1526368181000,
				"warrantValue": 0.0
		}]
	},
	"timestamp": 1527830042620
}
```

### 综合/模拟账号获取资产 PrimeAssetRequest

**说明**
查询综合/模拟账户的资产。  

**参数**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes  |用户授权账户: 123123
secret_key  |string |No |机构用户专用，交易员密钥

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.trade.PrimeAssetResponse`

可用 `PrimeAssetItem.Segment segment = primeAssetResponse.getSegment(Category.S)` 获取按交易品种划分的资产；  
对于每个品种的资产，可用 `PrimeAssetItem.CurrencyAssets assetByCurrency = segment.getAssetByCurrency(Currency.USD)`获取对应币种的资产。
具体参见示例代码

**``PrimeAssetItem.Segment``说明：**

名称 |类型| 示例 | 说明
--- |--- | --- | ---
account|String|123123|交易账户
currency|String|USD|货币的币种
category|String|S|交易品种分类 C(US Commodities 期货) or S(US Securities 证券)
capability|String|RegTMargin|账户类型, 保证金账户: RegTMargin, 现金账户: Cash。保证金账户支持融资融券功能，T+0交易次数不受限制，最大购买力于日内最高4倍，隔日最高2倍。
buyingPower|Double|6273545.18|最大购买力。 最大购买力是账户最大的可用购买金额。可以用最大购买力的值来估算账户最多可以买多少金额的股票，但是由于每只股票的最多可加的杠杆倍数不一样，所以实际购买单支股票时可用的购买力要根据具体股票保证金比例来计算。算法，最大购买力=4*可用资金。举例：小虎的可用资金是10万美元，那么小虎的最大购买力是40万美元，假设当前苹果股价是250美元一股，苹果的初始保证金比例是30%，小虎最多可以买100000/30%=33.34万美金的苹果，假设苹果的初始保证金比例为25%，小虎最多可以买100000/25%=40万美金的苹果。保证金账户日内最多有四倍于资金（未被占用做保证金的资金）的购买力 隔夜最多有两倍的购买力
cashAvailableForTrade|Double|1233662.1|可用资金。可用资金用来检查是否可以开仓或打新。开仓是指做多买入股票、做空融券卖出股票等交易行为。需要注意的是可用资金不等于可用现金，是用总资产、期权持仓市值、冻结资金和初始保证金等指标算出来的一个值，当可用资金大于0时，代表该账户可以开仓，可用资金\*4 为账户的最大可用购买力。算法，可用资金=总资产-美股期权市值-当前总持仓的初始保证金-冻结资金。其中初始保证金的算法是∑（持仓个股市值\*当前股票的开仓保证金比例）。举例：小虎当前账户总资产为10000美元，持有1000美元美股期权，持有总市值为2000美元的苹果，苹果的初始保证金比例当前为45%，没有冻结资金，小虎的可用资金=10000-1000-2000*45%=8100美元
cashAvailableForWithdrawal|Double|1233662.1|当前账号内可以出金的现金金额
cashBalance|Double|469140.39|现金额。现金额就是当前所有币种的现金余额之和。如果您当前帐户产生了融资或借款，需要注意的是利息一般是按天算，按月结，具体以融资天数计算。每日累计，下个月初5号左右统一扣除，所以在扣除利息之前，用户看到的现金余额是没有扣除利息的，如果扣除利息前现金余额为零，可能扣除后会产生欠款，即现金余额变为负值。
grossPositionValue|Double|865644.18|证券总价值，是账户持仓证券的总市值之和，即全部持仓的总市值。算法，持仓证券总市值之和；备注：所有持仓市值都会按主币种计算；举例1，若小虎同时持有市值3000美元苹果（也就是做多），持有市值-1000美元的谷歌（也就是做空），小虎证券总价值=3000美元苹果+（-1000美元谷歌）=2000美元；举例2，小虎持有市值1万美元的苹果，市值5000美元的苹果做多期权，小虎证券总价值=10000+5000=15000美元
initMargin|Double|292046.91|初始保证金。当前所有持仓合约所需的初始保证金要求之和。初次执行交易时，只有当含贷款价值总权益大于初始保证金才允许开仓。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。
maintainMargin|Double|273170.84|维持保证金。当前所有持仓合约所需的维持保证金要求之和。持有头寸时，当含贷款价值总权益小于维持保证金会引发强平。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。
overnightMargin|Double|273170.84|隔夜保证金。隔夜保证金是在收盘前15分钟开始检查账户所需的保证金，为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。老虎国际的隔夜保证金比例均在50%以上。如果账户含货款价值总权益低于隔夜保证金，账户在收盘前15分钟存在被强制平仓的风险。算法，∑（持仓个股隔夜时段的维持保证金）。个股的维持保证金率用户能够在“个股详情页-报价区-点击融资融券标识"查询。举例：小虎账户总资产为10万美元，苹果开仓保证金比例是40%，日内维持保证金是20%，日内全仓买入苹果共25万美元，到收盘前15分钟（隔夜时段）维持保证金比例将被提高到50%，此时的用户隔夜保证金为：25万*50%=12.5万，用户的含贷款价值总权益为10万小于12.5万，用户将会被平仓部分股票。
excessLiquidation|Double|960492.09|当前剩余流动性。当前剩余流动性是衡量当前账户潜在的被平仓风险的指标，当前剩余流动性越低账户被平仓的风险越高，当小于0时会被强制平掉部分持仓。具体的算法为：当前剩余流动性=含货款价值总权益(equityWithLoan)-账户维持保证金(maintainMargin) 。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。举例：(1)、客户总资产1万美金，买入苹果12000美金（假设苹果开仓和维持保证金比例为50%）。当前剩余流动性=总资产10000-账户维持保证金6000=4000。(2)、随着股票的下跌，假设股票市值跌到8000，这个时候当前剩余流动性=总资产 （-2000+8000）-账户维持保证金4000=2000。(3)、此时用户又买入了1000美金的美股期权，那么账户的当前剩余流动性还剩下2000-1000=1000。若您的账户被强制平仓，则会以市价单在强制平仓时进行成交，强平的股票对象由券商自行决定，请您注意风控值和杠杆等指标。|
overnightLiquidation|Double|1233662.93|隔夜剩余流动性。隔夜剩余流动性是指用 含货款价值总权益(equityWithLoan)-隔夜保证金(overnightMargin) 算出来的值。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。如果账户的隔夜剩余流动性低于0，在收盘前15分钟起账户存在被强行平掉部分持仓的风险。若您的账户被强制平仓，则会以市价单在强制平仓时进行成交，强平的股票对象由券商自行决定，请您注意风控值和杠杆等指标。
netLiquidation|Double|1233662.93|总资产(净清算值)。总资产就是我们账户的净清算现金余额和证券总市值之和，通常用来表示目前账户中有多少资产。算法，总资产=证券总市值+现金余额+应计分红-应计利息；举例：小虎账户有1000美元现金，持仓价值1000美元的苹果（即做多苹果），没有待发放的股息和融资利息和未扣除的利息，那么小虎的总资产=现金1000+持仓价值1000=2000美元，若小虎账户有1000美元现金，做空1000美元的苹果，此时证券总市值-1000美元，现金2000美元，用户总资产=现金2000+（持仓市值-1000），账户总资产共计1000美元。
equityWithLoan|Double|1233078.69|含贷款价值总权益，即ELV，ELV是用来计算开仓和平仓的数据指标；算法，现金账户=现金余额，保证金账户=现金余额+证券总市值-美股期权市值；ELV = 总资产 - 美股期权
realizedPL|Double|-248.72|已实现盈亏
unrealizedPL|Double|-17039.09|持仓盈亏。定义，持仓个股、衍生品的未实现盈亏金额；算法，当前价\*股数-持仓成本
leverage|Double|0.5|杠杆。杠杆是衡量账户风险程度的重要指标，可以帮助用户快速了解账户融资比例和风险程度；算法，杠杆=证券市值绝对值之和/总资产；备注1，保证金账户日内最大杠杆4倍，隔夜2倍；备注2，老虎考虑到历史波动、流动性和风险等因素，并不是每只股票都能4倍杠杆买入，一般做多保证金比例范围在25%-100%之间，保证金比例等于25%的股票，可以理解为4倍杠杆买入，保证金比例等于100%的股票，可以理解为0倍杠杆买入，即全部用现金买入。需要留意做空保证金可能会大于100%。备注3，个股保证金比例用户能够在“个股详情页-报价区-点击融资融券标识”查询；举例，小虎账户总资产10万美元，想买苹果，苹果当前个股做多初始保证金比例50%（1/50%=2倍杠杆），小虎最多只能买入20万市值的苹果股票；小虎想做空谷歌，谷歌做空保证金比例200%，小虎最多能做空10/200%=5万谷歌股票；小虎想买入微软，微软初始保证金100%（1/100%=1倍杠杆），小虎最多只能买入10万美元的微软股票
**currencyAssets**|CurrencyAssets||按照交易币种区分的账户资产信息。详细说明见下方描述。

**``PrimeAssetItem.CurrencyAssets``说明：**

名称 | 示例 | 说明
--- | --- | ---
currency|USD|当前的货币币种，常用货币包括： USD-美元，HKD-港币，SGD-新加坡币，CNH-人民币
cashBalance|469140.39|可以交易的现金，加上已锁定部分的现金（如已购买但还未成交的股票，还包括其他一些情形也会有锁定现金情况）
cashAvailableForTrade|0.1273896|当前账号内可以交易的现金金额
realizedPL|-248.72|账号内已实现盈亏
unrealizedPL|-17039.09|账号内浮动盈亏
stockMarketValue|943588.78|股票的市值
optionMarketValue|0.0|期权的市值
futuresMarketValue|0.0|期货的市值，category为S（证券类型）时，不会有期货市值

**示例**
```java
PrimeAssetRequest assetRequest = PrimeAssetRequest.buildPrimeAssetRequest("402901");
PrimeAssetResponse primeAssetResponse = client.execute(assetRequest);
//查询证券相关资产信息
PrimeAssetItem.Segment segment = primeAssetResponse.getSegment(Category.S);
System.out.println("segment: " + JSONObject.toJSONString(segment));
//查询账号中美元相关资产信息
if (segment != null) {
  PrimeAssetItem.CurrencyAssets assetByCurrency = segment.getAssetByCurrency(Currency.USD);
  System.out.println("assetByCurrency: " + JSONObject.toJSONString(assetByCurrency));
}

>>> segment: {"buyingPower":482651.8815597,"capability":"RegTMargin","cashAvailableForTrade":120662.9703899,"cashAvailableForWithdrawal":120662.9703899,"cashBalance":109829.460975,"category":"S","currency":"USD","currencyAssets":[{"cashAvailableForTrade":96353.34,"cashBalance":97008.79,"currency":"USD","grossPositionValue":8933.81,"optionMarketValue":0.0,"realizedPL":-3256.0,"stockMarketValue":8933.81,"unrealizedPL":3656.5633333},{"cashAvailableForTrade":98889.91,"cashBalance":99999.99,"currency":"HKD","grossPositionValue":95510.8,"optionMarketValue":0.0,"realizedPL":0.0,"stockMarketValue":95510.8,"unrealizedPL":-44555.184536},{"cashAvailableForTrade":0.0,"cashBalance":0.0,"currency":"CNH"}],"equityWithLoan":131008.3976131,"excessLiquidation":122581.4537219,"grossPositionValue":21178.9366381,"initMargin":9547.8475046,"leverage":0.1614001,"maintainMargin":8426.9438913,"netLiquidation":132373.3965882,"overnightLiquidation":121191.7607219,"overnightMargin":9816.6368913,"realizedPL":-3256.0,"unrealizedPL":-2055.7108496}
>>> assetByCurrency: {"cashAvailableForTrade":96353.34,"cashBalance":97008.79,"currency":"USD","grossPositionValue":8933.81,"optionMarketValue":0.0,"realizedPL":-3256.0,"stockMarketValue":8933.81,"unrealizedPL":3656.5633333}
```

**返回示例**
```json
{
	"code": 0,
	"message": "success",
	"timestamp": 1638357353112,
	"data": {
		"accountId": "402901",
		"segments": [{
				"capability": "RegTMargin",
				"category": "S",
				"currency": "USD",
				"cashBalance": 67819.3999309,
				"cashAvailableForTrade": 102137.7702566,
				"cashAvailableForWithdrawal": 102137.7702566,
				"grossPositionValue": 66855.3779781,
				"equityWithLoan": 134674.777909,
				"netLiquidation": 136039.7770055,
				"initMargin": 32373.9849708,
				"maintainMargin": 26753.6383093,
				"overnightMargin": 34002.253989,
				"unrealizedPL": -3807.1245493,
				"realizedPL": 0.0,
				"excessLiquidation": 107921.1395997,
				"overnightLiquidation": 100672.52392,
				"buyingPower": 408551.0810264,
				"leverage": 0.4928613,
				"currencyAssets": [{
					"currency": "USD",
					"cashBalance": 79709.88,
					"cashAvailableForTrade": 79689.28,
					"grossPositionValue": 54877.2,
					"stockMarketValue": 54877.2,
					"optionMarketValue": 0.0,
					"unrealizedPL": 2209.0633333,
					"realizedPL": 0.0
				}, {
					"currency": "HKD",
					"cashBalance": -92554.07,
					"cashAvailableForTrade": -93664.15,
					"grossPositionValue": 93236.7,
					"stockMarketValue": 93236.7,
					"optionMarketValue": 0.0,
					"unrealizedPL": -46829.284536,
					"realizedPL": 0.0
				}, {
					"currency": "CNH",
					"cashBalance": 0.0,
					"cashAvailableForTrade": 0.0
				}]
		}, {
			"capability": "RegTMargin",
			"category": "C",
			"currency": "USD",
			"cashBalance": 3483712.02,
			"cashAvailableForTrade": 3481732.02,
			"cashAvailableForWithdrawal": 3481732.02,
			"grossPositionValue": 1000000.0,
			"equityWithLoan": 3481912.02,
			"netLiquidation": 3483712.02,
			"initMargin": 1980.0,
			"maintainMargin": 1800.0,
			"overnightMargin": 1800.0,
			"unrealizedPL": 932722.41,
			"realizedPL": 0.0,
			"excessLiquidation": 3481912.02,
			"overnightLiquidation": 3481912.02,
			"buyingPower": 0.0,
			"leverage": 0.0,
			"currencyAssets": [{
				"currency": "USD",
				"cashBalance": 3483712.02,
				"cashAvailableForTrade": 3483712.02,
				"grossPositionValue": 1000000.0,
				"futuresMarketValue": 1000000.0,
				"unrealizedPL": 932722.41,
				"realizedPL": 0.0
			}, {
				"currency": "HKD",
				"cashBalance": 0.0,
				"cashAvailableForTrade": 0.0
			}, {
				"currency": "CNH",
				"cashBalance": 0.0,
				"cashAvailableForTrade": 0.0
			}]
		}]
	},
	"sign": "Uout3iD/74I9nGT++UsJlBcH4qcm01E+aQ+lf6Kc7VdC8vQ+qL8lswmhztyfpzlSb7itwrUuEubjozRk					+mmXHBY8ac3SxGJXJIm3eERkZmAll6KouV7HO8O14iPc1tQyNMjkQ="
}
```