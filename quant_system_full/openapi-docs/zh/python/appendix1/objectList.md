---
title: 对象列表
---
##  对象列表

### 请求客户端
|  对象名  | 描述 |
|  ----  | ---- |
|PushClient|订阅类。websocket长链接推送使用该类处理, 如实时行情推送、资产/订单/持仓变动推送|
|QuoteClient|行情类。行情有关接口使用该类处理, 如请求k线、实时价格|
|TradeClient|交易类。交易有关接口使用该类处理, 如下单、改单|

### 账户类
**综合/模拟账户**
|  对象名  |引用路径| 描述 |
|  ----  |----| ---- |
| [PortfolioAccount](/zh/python/appendix1/object.html#portfolioaccount-资产-综合-模拟账户) | tigeropen.trade.domain.prime_account.PortfolioAccount |账户资产信息|
| [Segment](/zh/python/appendix1/object.html#segment-分品种资产-综合-模拟账户) |tigeropen.trade.domain.prime_account.Segment|按交易品种划分的资产(期货/股票)|
| [CurrencyAsset](/zh/python/appendix1/object.html#currencyasset-分币种资产-综合-模拟账户) | tigeropen.trade.domain.prime_account.CurrencyAsset|现金资产信息|

**环球账户**
|  对象名  |引用路径| 描述 |
|  ----  |----| ---- |
|[PortfolioAccount](/zh/python/appendix1/object.html#portfolioaccount-资产-环球账户)| tigeropen.trade.domain.account.PortfolioAccount |账户资产信息|
|[Account](/zh/python/appendix1/object.html#account-汇总资产-环球账户)| tigeropen.trade.domain.account.Account |汇总的账户信息|
|[CommoditySegment](/zh/python/appendix1/object.html#commoditysegment-期货资产-环球账户)| tigeropen.trade.domain.account.CommoditySegment |期货资产信息|
|[SecuritySegment](/zh/python/appendix1/object.html#securitysegment-股票资产-环球账户)| tigeropen.trade.domain.account.SecuritySegment |股票资产信息|
|[MarketValue](/zh/python/appendix1/object.html#marketvalue-分币种资产-环球账户)|tigeropen.trade.domain.account.MarketValue|市值对象|

### 交易类
|  对象名  |引用路径| 描述 |
|  ----  |----| ---- |
|[Position](/zh/python/appendix1/object.html#position-持仓)| tigeropen.trade.domain.position.Position |持仓对象|
|[Order](/zh/python/appendix1/object.html#order-订单)| tigeropen.trade.domain.order.Order |订单对象|
|[OrderLeg](/zh/python/appendix1/object.html#orderleg-附加订单)| tigeropen.trade.domain.order.OrderLeg | 附加订单对象 |
|[AlgoParams](/zh/python/appendix1/object.html#algoparams-算法订单参数)| tigeropen.trade.domain.order.AlgoParams | 算法订单(VWAP/TWAP)参数 |
|[Contract](/zh/python/appendix1/object.html#contract-合约)| tigeropen.trade.domain.contract.Contract |合约对象|

### 行情类
|  对象名  |引用路径| 描述 |
|  ----  |----| ---- |
| [MarketStatus](/zh/python/appendix1/object.html#marketstatus-市场状态) |tigeropen.quote.domain.market_status.MarketStatus| 市场状态对象 |
| [OptionFilter](/zh/python/appendix1/object.html#optionfilter-期权链过滤器) |tigeropen.quote.domain.filter.OptionFilter|期权过滤Filter|

