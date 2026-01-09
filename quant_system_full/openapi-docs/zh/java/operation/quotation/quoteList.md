# 行情接口列表

**行情类请求通过`TigerHttpClient`对象发送, `TigerHttpClient`类为Http客户端，负责发送行情类与交易类请求，发送请求前需先初始化`TigerHttpClient`。具体初始化方法及使用方法请参考请求对象/接口的使用示例**

## 通用

接口|功能描述
----|----
[TigerHttpRequest(ApiServiceType.GRAB_QUOTE_PERMISSION)](/zh/java/operation/quotation/common.html)|抢占行情权限/获取权限列表

### 证券类

接口|功能描述
----|----
[QuoteMarketRequest](/zh/java/operation/quotation/stock.html#获取市场状态-quotemarketrequest)|获取市场状态
[QuoteSymbolRequest](/zh/java/operation/quotation/stock.html#获取股票代号列表-quotesymbolrequest)|股票代号列表
[QuoteSymbolNameRequest](/zh/java/operation/quotation/stock.html#获取股票代号列表和名称-quotesymbolnamerequest)|股票代号列表和名称
[QuoteDelayRequest](/zh/java/operation/quotation/stock.html#延时行情-quotedelayrequest)|延时行情
[QuoteTimelineRequest](/zh/java/operation/quotation/stock.html#获取分时数据-quotetimelinerequest)|分时数据
[QuoteRealTimeQuoteRequest](/zh/java/operation/quotation/stock.html#获取实时行情-quoterealtimequoterequest)|实时行情
[QuoteKlineRequest](/zh/java/operation/quotation/stock.html#获取k线数据-quoteklinerequest)|K线数据
[QuoteTradeTickRequest](/zh/java/operation/quotation/stock.html#获取逐笔成交-quotetradetickrequest)|逐笔成交
[QuoteStockTradeRequest](/zh/java/operation/quotation/stock.html#获取股票交易信息-quotestocktraderequest)|股票交易信息

### 期货

我们暂未支持期货行情，预计期货行情接口将于本季度开放。请持续关注

### 期权

接口请求对象|功能描述
----|----
[OptionExpirationQueryRequest](/zh/java/operation/quotation/option.html#optionexpirationqueryrequest-获取期权过期日)|获取期权过期日
[OptionChainQueryV3Request](/zh/java/operation/quotation/option.html#optionchainqueryv3request-获取期权链)|获取期权链
[OptionBriefQueryRequest](/zh/java/operation/quotation/option.html#optionbriefqueryrequest-获取期权行情摘要)|获取期权行情摘要
[OptionKlineQueryRequest](/zh/java/operation/quotation/option.html#optionklinequeryrequest-获取期权k线)|获取期权K线
[OptionTradeTickQueryRequest](/zh/java/operation/quotation/option.html#optiontradetickqueryrequest-获取期权逐笔成交)|获取期权逐笔成交数据
[OptionFundamentals](http://localhost:8080/desktop/cdn/openapi-test/zh/java/operation/quotation/option.html#optiontradetickqueryrequest-%E8%8E%B7%E5%8F%96%E6%9C%9F%E6%9D%83%E9%80%90%E7%AC%94%E6%88%90%E4%BA%A4)|期权指标计算工具