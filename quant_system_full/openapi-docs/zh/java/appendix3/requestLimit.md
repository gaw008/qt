---
title: 请求频率限制
---
## 频率限制
为保护服务器，防止恶意攻击，所有需要向老虎OpenAPI服务器发送请求的接口，都会有频率限制。

API中的接口分为低频接口和高频接口，低频接口限制为10次/分钟，高频接口为120次/每分钟，超过频率的限制的请求会被服务器拒绝。计数周期为任意60秒的滑动窗口

### 低频接口列表（10次/分钟）
|  接口方法  | 
|  ----  | 
|TigerHttpRequest(ApiServiceType.GRAB_QUOTE_PERMISSION)	抢占行情权限/获取权限列表|    
|QuoteMarketRequest 市场状态|    
|QuoteSymbolRequest 股票代号|
|QuoteSymbolNameRequest 股票代号名称|  
|QuoteKlineRequest K线| 
|OptionKlineQueryRequest 期权K线|    
|OptionChainQueryV3Request 期权链| 
|OptionExpirationQueryRequest 期权过期日| 
<!--
|期货k线(暂不开放）|    
|期货交易所(暂不开放)|
-->

### 高频接口列表（120次/分钟）
|  接口方法  | 
|  ----  | 
|TigerHttpRequest(ApiServiceType.ORDER_NO) 获取订单号|   
|TradeOrderRequest 交易下单| 
|TigerHttpRequest(ApiServiceType.MODIFY_ORDER) 修改订单|
|TigerHttpRequest(ApiServiceType.CANCEL_ORDER) 取消订单|
|TigerHttpRequest(ApiServiceType.ORDERS) 查询订单 |
|TigerHttpRequest(ApiServiceType.ACTIVE_ORDERS) 查询待成交订单|
|TigerHttpRequest(ApiServiceType.INACTIVE_ORDERS) 已撤销订单|
|TigerHttpRequest(ApiServiceType.FILLED_ORDERS) 查询已成交订单 |
|QuoteTimelineRequest 获取分时数据|
|QuoteRealTimeQuoteRequest 实时行情|
|QuoteTradeTickRequest 逐笔成交|
|OptionBriefQueryRequest 期权行情摘要|
|OptionTradeTickQueryRequest 期权逐笔成交|
<!--
|期货逐笔成交(暂不开放）|
|期货实时行情(暂不开放)|
-->


### 其他接口（60次/分钟）
不在上述列表里的其他接口，默认访问频率限制为 60次/min