---
title: 请求频率限制
---
## 频率限制
为保护服务器，防止恶意攻击，所有需要向老虎OpenAPI服务器发送请求的接口，都会有频率限制。

API中的接口分为低频接口和高频接口，低频接口限制为10次/分钟，高频接口为120次/每分钟，超过频率的限制的请求会被服务器拒绝。计数周期为任意60秒的滑动窗口

### 低频接口列表（10次/分钟）
|  接口方法  | 
|  ----  | 
|行情抢占(grab_quote_permission)|   
|行情权限列表(get_quote_permission)| 
|市场状态(get_market_status)|    
|股票代号(get_symbols)|
|股票代号名称(get_symbol_names)   |
|股票行情(get_stock_details)|    
|K线(get_bars)| 
|期权K线(get_option_bars)|    
|期权链(get_option_chain)| 
|期权过期日(get_option_expirations)|  
|期货k线(暂不开放)|    
|期货交易所(get_future_exchanges)|

### 高频接口列表（120次/分钟）
|  接口方法  | 
|  ----  | 
|获取订单号(create_order)|   
|创建订单(place_order)| 
|修改订单(modify_order) |
|取消订单(cancel_order) |
|查询订单(get_orders)   |
|查询未成交订单(get_open_orders)  |
|查询已撤销订单(get_cancelled_orders)  |
|查询已成交订单(get_filled_orders)  |
|分时(get_timeline)   |
|实时行情(get_stock_briefs)  |
|逐笔成交(get_trade_ticks)|
|期权行情摘要(get_option_briefs)   |
|期权逐笔成交(get_option_trade_ticks)|
|期货合约(get_future_current_contract)|


### 其他接口（60次/分钟）
不在上述列表里的其他接口，默认访问频率限制为 60次/min