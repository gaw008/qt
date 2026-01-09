## 行情接口列表
行情类接口负责查询行情相关的功能，目前支持的功能以及对应的行情接口如下表：

注：调用行情接口之前需要先初始化[`QuoteClient`](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/quote/quote_client.py) ，初始化行情对象的示例请见[快速入门](/zh/python/quickStart/basicFunction.html#查询行情)
 

### 通用
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [grab_quote_permission](/zh/python/operation/quotation/common.html#grab-quote-permission-行情权限抢占)  | 行情权限抢占 |
| [get_quote_permission](/zh/python/operation/quotation/common.html#get-quote-permission-查询行情权限) | 行情权限查询 |

### 股票行情
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_market_status](/zh/python/operation/quotation/stock.html#get-market-status-获取市场状态) | 获取市场交易状态 |
| [get_symbols](/zh/python/operation/quotation/stock.html#get-symbols-获取所有证券代码列表)  |  获取所有证券的代码列表 |
| [get_symbol_names](/zh/python/operation/quotation/stock.html#get-symbol-names-获取代码及名称列表)  | 获取代码及名称列表 |
| [get_timeline](/zh/python/operation/quotation/stock.html#get-timeline-获取最新一日的分时数据)  | 获取最新一日的分时数据 |
| [get_stock_briefs](/zh/python/operation/quotation/stock.html#get-stock-briefs-获取股票实时行情)  | 获取股票实时行情 | <!-- | [get_stock_details](/zh/python/operation/quotation/stock.html#get-stock-details-获取股票详情) | 获取股票详情 |-->
| [get_stock_delay_briefs](/zh/python/operation/quotation/stock.html#get-stock-delay-briefs-获取股票延迟行情)| 获取股票延迟行情 |
| [get_bars](/zh/python/operation/quotation/stock.html#get-bars-获取个股k线数据)  | 获取个股K线数据 |
| [get_trade_ticks](/zh/python/operation/quotation/stock.html#get-trade-ticks-获取逐笔成交数据)  | 获取逐笔成交数据 |
| [get_depth_quote](/zh/python/operation/quotation/stock.html#get-depth-quote-获取深度行情) | 获取深度行情 |
| [get_short_interest](/zh/python/operation/quotation/stock.html#get-short-interest-获取美股的做空数据-目前未上线) | 获取美股的做空数据\<目前未上线\>|
| [get_trade_metas](/zh/python/operation/quotation/stock.html#get-trade-metas-获取股票交易需要的信息) | 获取股票交易需要的信息(每股手数, 价格变动单位等) |

### 期货行情（请求行情接口暂未开放）
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_future_exchanges](/zh/python/operation/quotation/future.html#get-future-exchanges-获取期货交易所列表) | 获取期货交易所列表 |
| [get_future_contracts](/zh/python/operation/quotation/future.html#get-future-contracts-获取交易所下的可交易合约)  | 获取交易所下的可交易合约 |
| [get_current_future_contract](/zh/python/operation/quotation/future.html#get-current-future-contract-查询指定品种的当前合约) | 查询指定品种的当前合约 |
| [get_future_trading_times](/zh/python/operation/quotation/future.html#get-future-trading-times-查询指定合约的交易时间) | 查询指定合约的交易时间 |

### 期权行情
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_option_expirations](/zh/python/operation/quotation/option.html#get-option-expirations-获取美股期权到期日) | 获取美股期权过期日 |
| [get_option_chain](/zh/python/operation/quotation/option.html#get-option-chain-获取期权链)  | 获取期权链 |
| [get_option_briefs](/zh/python/operation/quotation/option.html#get-option-briefs-获取期权最新行情) | 获取期权最新行情 |
| [get_option_bars](/zh/python/operation/quotation/option.html#get-option-bars-获取期权日k数据)  | 获取期权日K数据 |
| [get_option_trade_ticks](/zh/python/operation/quotation/option.html#get-option-trade-ticks-获取期权的逐笔成交数据)  | 获取期权逐笔成交数据 |