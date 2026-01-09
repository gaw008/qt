---
title: 获取合约
---
### 合约简介
合约是指交易的买卖对象或者标的物（比如一只股票，或者一个期权），合约是由交易所统一制定的。比如购买老虎证券的股票，可以通过TIGR这个字母代号和市场信息（即market=’US‘，美国市场）来唯一标识。类似的在购买期权或者期货产品时，可能会需要用到其他一些标识字段。通过合约信息，我们在下单或者获取行情时就可以唯一的确定一个标的物了。

常见的合约包括股票合约，期权合约，期货合约等。

大部分合约包括如下几个要素:

* 标的代码(symbol)，一般美股、英股等合约代码都是英文字母，港股、A股等合约代码是数字，比如老虎证券的symbol是TIGR。
* 合约类型(security type)，常见合约类型包括：STK（股票），OPT（期权），FUT（期货），CASH（外汇），比如老虎证券股票的合约类型是STK。
* 货币类型(currency)，常见货币包括 USD（美元），HKD（港币）。
* 交易所(exchange)，STK类型的合约一般不会用到交易所字段，订单会自动路由，期货合约都用到交易所字段。

绝大多数股票，差价合约，指数或外汇对可以通过这四个属性来唯一确定。

由于其性质，更复杂的合约（如期权和期货）需要一些额外的信息。

以下是几种常见类型合约，以及其由哪些要素构成。

**股票**

```java
ContractItem contract = new ContractItem();
contract.symbol ="TIGR";
contract.secType ="STK";
contract.currency ="USD"; //非必填，下单时默认为USD
contract.market = "US"; //非必填，合约市场，包括US（美国市场），HK（香港市场），CN（国内市场）等。下单时默认为US
```


**外汇**

```java
ContractItem contract = new ContractItem();
contract.symbol ="EUR";
contract.secType ="CASH";
contract.currency ="GBP";
contract.exchange ="IDEALPRO"; //非必填，交易所字段
```



**期权**

```java
ContractItem contract = new ContractItem();
contract.symbol ="AAPL";
contract.secType ="OPT";
contract.currency ="USD";
contract.expiry="20180821";
contract.strike = 30;
contract.right ="CALL";
contract.multiplier =100.0;
contract.market = "US"; //非必填
```

**期货**

```java
ContractItem contract = new ContractItem();
contract.symbol ="CL1901";
contract.secType ="FUT";
contract.exchange ="SGX";
contract.currency ="USD";
contract.expiry="20190328";
contract.multiplier=1.0;
```


### 获取合约信息 ContractRequest/ContractsRequest

**说明**

获取交易需要的合约信息。
`ContractRequest` 请求单个
`ContractsRequest` 请求多个

需要注意环球账户、综合账户返回`ContractItem`字段值数目会不同，建议获取合约和下单使用相同的账户。

**输入参数**

`com.tigerbrokers.stock.openapi.client.https.request.contract.ContractRequest`
参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
account       |string  |  Yes  |用户授权账户:DU575569
symbol        |string  |  Yes  |股票代码 如：600884 / SNAP
sec_type      |string  |  Yes  |STK/OPT
currency      |string  |  No  |USD/HKD/CNH
expiry        |string  |  No   |到期日 交易品种是期权时必传 yyyyMMdd
strike        |double  |  No   |行权价 交易品种是期权时必传
right         |string  |  No   |CALL/PUT 交易品种是期权时必传
exchange      |string  |  No   |交易所 (美股 SMART 港股 SEHK 沪港通 SEHKNTL 深港通 SEHKSZSE)
secret_key | string| No | 机构用户专用，交易员密钥

**返回**

`com.tigerbrokers.stock.openapi.client.https.response.contract.ContractResponse` 

其中数据项字段如下
`com.tigerbrokers.stock.openapi.client.https.domain.contract.item.ContractItem`
名称 | 示例 | 说明
--- | --- | ---
identifier| CL2109/AAPL| 唯一标识
contractId| 1 | 债券的IB合约代码
symbol| LRN | 股票代码
secType|STK |STK/OPT/WAR/IOPT 默认 STK
name| K12 INC | 股票名称
localSymbol|1033 |港股用于识别窝轮和牛熊证
currency|USD |USD/HKD/CNH
exchange|NYSE |股票交易所
primaryExchange|NYSE|股票上市交易所
market| US | 市场 /US/HK/CN
expiry| 20171117 | 期权过期日
contractMonth|201804 |合约月份
right| PUT |期权方向
strike| 24.0|期权底层价格
multiplier| 0.0 |每手数量
minTick|0.001|最小报价单位
marginable|true|是否可融资
shortable|true|能否做空
longInitialMargin|1|做多初始保证金
longMaintenanceMargin|1|做多维持保证金
shortMargin|0.35|做空保证金(将要废弃，请使用shortInitialMargin)
shortInitialMargin|0.35|做空初始保证金比例
shortMaintenanceMargin|0.3|做空维持保证金比例（综合账号有值，环球账号合约没有值）
shortableCount|10000000|可做空数目
shortFeeRate|0|做空费率
tradingClass|LRN |合约的交易级别名称
tradeable|true |是否可交易（仅限于STK类别）
continuous|false|期货专有，是否连续合约
trade|false|期货专有，是否可交易
lastTradingDate|2019-01-01|期货专有，最后交易日
firstNoticeDate|2019-01-01|期货专有，第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
lastBiddingCloseTime|0|期货专有，竞价截止时间

**示例**
```java
// init client instance
TigerHttpClient client = new TigerHttpClient(TigerOpenClientConfig.getDefaultClientConfig());
// use default account(clientConfig.defaultAccount)
ContractRequest contractRequest = ContractRequest.newRequest(
    new ContractModel("AAPL"));
ContractResponse contractResponse = client.execute(contractRequest);
System.out.println(JSONObject.toJSONString(contractResponse));
// use account parameter
contractRequest = ContractRequest.newRequest(
    new ContractModel("AAPL"), "402901");
contractResponse = client.execute(contractRequest);
System.out.println(JSONObject.toJSONString(contractResponse));

ContractItem contract = contractResponse.getItem();

// using standard account, get option contract
ContractModel model = new ContractModel("AAPL", SecType.OPT.name(),
    Currency.USD.name(), "20211126", 150D, Right.CALL.name());
contractRequest = ContractRequest.newRequest(model, "402901");
contractResponse = client.execute(contractRequest);
System.out.println("return standard contract:" + JSONObject.toJSONString(contractResponse));

// using standard account, get warrant contract
ContractModel contractModel = new ContractModel("13745", SecType.WAR.name());
contractModel.setStrike(719.38D);
contractModel.setRight(Right.CALL.name());
contractModel.setExpiry("20211223");
ContractRequest contractRequest = ContractRequest.newRequest(
    contractModel, "402901");
ContractResponse contractResponse = client.execute(contractRequest);
System.out.println("return standard contract:" + JSONObject.toJSONString(contractResponse));
```
获取多个股票合约代码示例:

```java
List<String> symbols = new ArrayList<>();
symbols.add("AAPL");
symbols.add("TSLA");
ContractsRequest contractsRequest = ContractsRequest.newRequest(
    new ContractsModel(symbols), "402901");
ContractsResponse contractsResponse = client.execute(contractsRequest);
System.out.println("return standard contracts:" + JSONObject.toJSONString(contractsResponse));
```


**返回示例**

单个合约
```json
{
	"code": 0,
	"item": {
		"currency": "USD",
		"identifier": "AAPL",
		"localSymbol": "AAPL",
		"longInitialMargin": 0.3,
		"longMaintenanceMargin": 0.3,
		"marginable": true,
		"market": "US",
		"multiplier": 1.0,
		"name": "Apple Inc",
		"secType": "STK",
		"shortFeeRate": 3.25,
		"shortInitialMargin": 0.4,
		"shortMaintenanceMargin": 0.4,
		"shortMargin": 0.4,
		"shortable": false,
		"shortableCount": 0,
		"status": 1,
		"symbol": "AAPL",
		"tradeable": true,
		"tradingClass": "AAPL"
	},
	"message": "success",
	"success": true,
	"timestamp": 1637684344859
}
```

多个合约
```
{
	"code": 0,
	"data": [{
		"contractId": 265598,
		"currency": "USD",
		"exchange": "SMART",
		"identifier": "AAPL",
		"localSymbol": "AAPL",
		"marginable": false,
		"market": "US",
		"minTick": 0.01,
		"name": "Apple",
		"primaryExchange": "NASDAQ",
		"secType": "STK",
		"shortFeeRate": 0.25,
		"shortable": true,
		"shortableCount": 10000000,
		"status": 1,
		"symbol": "AAPL",
		"tradeable": true,
		"tradingClass": "NMS"
	}, {
		"contractId": 76792991,
		"currency": "USD",
		"exchange": "SMART",
		"identifier": "TSLA",
		"localSymbol": "TSLA",
		"longInitialMargin": 0.4,
		"longMaintenanceMargin": 0.4,
		"marginable": true,
		"market": "US",
		"minTick": 0.01,
		"name": "Tesla Motors",
		"primaryExchange": "NASDAQ",
		"secType": "STK",
		"shortFeeRate": 0.25,
		"shortInitialMargin": 0.3,
		"shortMargin": 0.3,
		"shortable": true,
		"shortableCount": 10000000,
		"status": 1,
		"symbol": "TSLA",
		"tradeable": true,
		"tradingClass": "NMS"
	}],
	"message": "success",
	"success": true,
	"timestamp": 1640251997046
}
```



### 获取期权/窝轮/牛熊证合约列表

**输入参数：**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
symbols |array|Yes|股票代码列表，仅支持一个symbol
sec_type|string     |  Yes  | 合约类型 目前支持: (OPT 期权/ WAR 港股窝轮/ IOPT 港股牛熊证) 
expiry|String  |  | 到期日(yyyyMMdd), 如果是OPT必须有值 
lang|string|No|语言支持: zh_CN,zh_TW,en_US, 默认: en_US


**返回结果：**

名称|类型|说明
--- | --- | ---
symbol|string|股票代码
name | string  | 合约名称
exchange | string  | 交易所
market | string | 市场
secType | string  | 合约类型
currency | string  | 币种
expiry | string  | 到期日(期权、窝轮、牛熊证、期货)， 20171117 
right | string  | 期权方向(期权、窝轮、牛熊证), PUT/CALL
strike | string  | 行权价
multiplier | double | 每手数量(期权、窝轮、牛熊证、期货)


**请求示例:**
```java
List<String> symbols = new ArrayList<>();
symbols.add("00700");
QuoteContractResponse response = client.execute(QuoteContractRequest.newRequest(symbols, SecType.WAR, "20211223"));
if (response.isSuccess()) {
  System.out.println(response.getContractItems());
} else {
  System.out.println("response error:" + response.getMessage());
}
```

**响应示例：**
```json
{
	"code": 0,
	"data": [{
		"items": [{
			"currency": "HKD",
			"exchange": "SEHK",
			"expiry": "20211223",
			"market": "HK",
			"multiplier": 50000.0,
			"name": "MSTENCT@EC2112B.C",
			"right": "CALL",
			"secType": "WAR",
			"strike": "719.38",
			"symbol": "13745"
		}, {
			"currency": "HKD",
			"exchange": "SEHK",
			"expiry": "20211223",
			"market": "HK",
			"multiplier": 5000.0,
			"name": "JPTENCT@EC2112A.C",
			"right": "CALL",
			"secType": "WAR",
			"strike": "900.5",
			"symbol": "13680"
		}],
		"secType": "WAR",
		"symbol": "00700"
	}],
	"message": "success",
	"sign": "bxQhZiWMsT9aSVTNtt2SXVeeh5w8Ypug/6UY3nL9N7LFKB1YxBVpQoKDJ4JloFojyb/CPCGT0fCXTxboDBTZvnA4stjbh1YqbNlz2lNqmHhpxYUKMdE+w2hFKVvoYMlMPCmsY5NqSQ3S/fsSzZrJyxBRPzZ+d+0qb7VSYw9yhho=",
	"success": true,
	"timestamp": 1637686550209
}
```


### 获取期货合约

**输入参数：** 

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
contract_code|string|yes|合约的symbol，如 CN1901
lang|string|no|语言参数，对返回值中的"name"字段有影响，取值范围为："zh_CN", "en_US"


**返回结果：** 

名称|类型|说明
--- | --- | ---
type|string|期货合约对应的交易品种， 如 CL
trade|boolean|是否可交易
continuous|boolean|是否连续合约
name|string|合约的名字，有简体和英文名，根据参数lang来返回
currency|string|交易的货币
ibCode|string|交易合约代码，下单时使用。如：CL
contractCode|string|合约代码，如，CL1901
contractMonth|string|合约的交割月份
lastTradingDate|string|交最后交易日
firstNoticeDate|string|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
lastBiddingCloseTime|long|竞价截止时间
multiplier |double | 合约乘数
exchangeCode| string| 交易所代码
minTick| double | 最小报价单位 


**请求示例:**
```java
FutureContractResponse response = client.execute(FutureContractByConCodeRequest.newRequest("CN1901"));
System.out.println(response.getFutureContractItem());
```

**响应示例：**
```json
{
  "code": 0,
  "serverTime": 1545049140384,
  "message": "success",
  "data": {
		"type": "CL",
		"trade": true,
		"continuous": false,
		"name": "WTI原油1901",
		"currency": "USD",
		"ibCode":"CL",
		"contractCode": "CL1901",
		"contractMonth": "201901",
		"firstNoticeDate": "20181221",
		"lastTradingDate": "20181219",
		"lastBiddingCloseTime": 0,
		"exchangeCode":"NYMEX",
		"multiplier":1000,
		"minTick":0.01
	}
}
```

### 获取交易所下的可交易合约（期货）

**输入参数：**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
exchange_code|string|yes|交易所代码
lang|string|no|语言参数，对返回值中的"name"字段有影响，取值范围为："zh_CN", "en_US"


**返回结果：**

名称|类型|说明
--- | --- | ---
type|string|期货合约对应的交易品种， 如 CL
trade|boolean|是否可交易
continuous|boolean|是否连续合约
name|string|合约的名字，有简体和英文名，根据参数lang来返回
currency|string|交易的货币
ibCode|string|交易合约代码，下单时使用。如：CL
contractCode|string|合约代码，如， CL1901
contractMonth|string|合约的交割月份
lastTradingDate|string|最后交易日
firstNoticeDate|string|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
lastBiddingCloseTime|long|竞价截止时间
multiplier |double | 合约乘数
exchangeCode| string| 交易所代码
minTick| double | 最小报价单位 


**请求示例:**
```java
FutureBatchContractResponse response = client.execute(FutureContractByExchCodeRequest.newRequest("CME"));
System.out.println(response.getFutureContractItems());
```

**响应示例：**
```json
{
   "code": 0,
   "serverTime": 1545049140384,
   "message": "success",
   "data": [{
      "type": "CL",
      "trade": true,
      "continuous": false,
      "name": "WTI原油1901",
      "currency": "USD",
      "ibCode":"CL",
      "contractCode": "CL1901",
      "contractMonth": "201901",
      "firstNoticeDate": "20181221",
      "lastTradingDate": "20181219",
      "lastBiddingCloseTime": 0,
      "exchangeCode":"NYMEX",
      "multiplier":1000,
      "minTick":0.01
   }]
```

### 查询指定品种的连续合约（期货）

**输入参数：**

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
type|string|yes|期货合约对应的交易品种， 如 CL
lang|string|yes|语言参数，对返回值中的"name"字段有影响，取值范围为："zh_CN", "en_US"

**返回结果：**

名称|类型|说明
 --- | --- | ---
type|string|期货合约对应的交易品种， 如 CL
trade|boolean|是否可交易
continuous|boolean|是否连续合约
name|string|合约的名字，有简体和英文名，根据参数lang来返回
currency|string|交易的货币
ibCode|string|交易合约代码，下单时使用。如：CL
contractCode|string|合约代码，如， CL1901
contractMonth|string|合约的交割月份
lastTradingDate|string|最后交易日
firstNoticeDate|string|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
lastBiddingCloseTime|long|竞价截止时间
multiplier |double | 合约乘数
exchangeCode| string| 交易所代码
minTick| double | 最小报价单位 


**请求示例:**
```java
FutureContractResponse cl = client.execute(FutureContinuousContractRequest.newRequest("CL"));
System.out.println(cl.getFutureContractItem());
```

**响应示例：**
```json
{
   "code": 0,
   "serverTime": 1545049140384,
   "message": "success",
   "data": {
      "type": "CL",
      "trade": true,
      "continuous": false,
      "name": "WTI原油1901",
      "currency": "USD",
      "ibCode":"CL",
      "contractCode": "CL1901",
      "contractMonth": "201901",
      "firstNoticeDate": "20181221",
      "lastTradingDate": "20181219",
      "lastBiddingCloseTime": 0,
      "exchangeCode":"NYMEX",
      "multiplier":1000,
      "minTick":0.01
   }
}
```

### 查询指定品种的当前合约（期货）

**输入参数：** 

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
type|string|yes|期货合约对应的交易品种， 如 CL
lang|string|no|语言参数，对返回值中的"name"字段有影响，取值范围为："zh_CN", "en_US"

**返回** 

名称|类型|说明
 --- | --- | ---
type|string|期货合约对应的交易品种， 如 CL
trade|boolean|是否可交易
continuous|boolean|是否连续合约
name|string|合约的名字，有简体和英文名，根据参数lang来返回
currency|string|交易的货币
ibCode|string|交易合约代码，下单时使用。如：CL
contractCode|string|合约代码，如， CL1901
contractMonth|string|合约的交割月份
lastTradingDate|string|最后交易日
firstNoticeDate|string|第一通知日，合约在第一通知日后无法开多仓。已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓。
lastBiddingCloseTime|long|竞价截止时间
multiplier |double | 合约乘数
exchangeCode| string| 交易所代码
minTick| double | 最小报价单位 

**请求示例**
```java
FutureContractResponse response = client.execute(FutureCurrentContractRequest.newRequest("CL"));
System.out.println(response.getFutureContractItem());
```

**返回示例**
```json
{
   "code": 0,
   "serverTime": 1545049140384,
   "message": "success",
   "data": {
      "type": "CL",
      "trade": true,
      "continuous": false,
      "name": "WTI原油1901",
      "currency": "USD",
      "ibCode":"CL",
      "contractCode": "CL1901",
      "contractMonth": "201901",
      "firstNoticeDate": "20181221",
      "lastTradingDate": "20181219",
      "lastBiddingCloseTime": 0,
      "exchangeCode":"NYMEX",
      "multiplier":1000,
      "minTick":0.01
   }
}
```


### 查询指定期货合约的交易时间 FutureTradingDateRequest

**参数** 

参数 | 类型 | 是否必填 | 描述
--- | --- | --- | ---
contract_code|string|yes|期货合约代码，如CL1901
trading_date|long|yes|交易日时间戳

**返回** 

名称|类型|说明
 --- | --- | ---
tradingTimes|array|交易时间
biddingTimes|array|竞价时间
timeSection|string|所在交易时区


**请求示例**
```java
FutureTradingDateResponse response = client.execute(FutureTradingDateRequest.newRequest("CN1901", System.currentTimeMillis()));
System.out.println(response.getFutureTradingDateItem());

```

**响应示例**
```json
{
    "code": 0,
    "timestamp": 1545049282852,
    "message": "success",
    "data": {
        "tradingTimes": [
            {
                "start": 1544778000000,
                "end": 1544820300000
            },
            {
                "start": 1545008400000,
                "end": 1545035400000
            }
        ],
        "timeSection": "Singapore",
        "biddingTimes": [
            {
                "start": 1544777400000,
                "end": 1544778000000
            },
            {
                "start": 1545007500000,
                "end": 1545008400000
            },
            {
                "start": 1545035400000,
                "end": 1545035700000
            }
        ]
    }
}
```
