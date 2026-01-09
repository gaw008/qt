---
title: 对象
---
# PortfolioAccount 资产(综合/模拟账户)
`tigeropen.trade.domain.prime_account.PortfolioAccount` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/prime_account.py)  

**说明**  

账户资产对象，适用于综合/模拟账户。包含账户的总资产、盈亏、持仓市值、现金、可用资金、保证金、杠杆等相关信息。

**对象属性**

属性名|类型|描述
----|----|----
account|str|对应的账户 id
update_timestamp|int|更新时间, 毫秒为单位的13位数字时间戳
segments|tigeropen.trade.domain.prime_account.Segment|按照交易品种区分的账户信息。内容是一个dict，分别有两个key，'S'表示证券，'C' 表示期货，value均为 [Segment 对象](/zh/python/appendix1/object.html#segment-分品种资产-综合-模拟账户)。

# Segment 分品种资产(综合/模拟账户)
`tigeropen.trade.domain.prime_account.Segment` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/prime_account.py)

**说明**

将资产按照股票/期货交易品种划分，每部分为一个Segment。

**对象属性**

属性名|类型|描述
----|----|----
currency|str|币种, 如 USD, HKD
capability|str|账户类型, 保证金账户: RegTMargin, 现金账户: Cash。保证金账户支持融资融券功能，T+0交易次数不受限制，最大购买力于日内最高4倍，隔日最高2倍。
category|str|交易品种分类 C: (Commodities 期货), S: (Securities 股票)
cash_balance|float|现金额。现金额就是当前所有币种的现金余额之和。如果您当前帐户产生了融资或借款，需要注意的是利息一般是按天算，按月结，具体以融资天数计算。每日累计，下个月初5号左右统一扣除，所以在扣除利息之前，用户看到的现金余额是没有扣除利息的，如果扣除利息前现金余额为零，可能扣除后会产生欠款，即现金余额变为负值。
cash_available_for_trade|float| 可用资金。可用资金用来检查是否可以开仓或打新。开仓是指做多买入股票、做空融券卖出股票等交易行为。需要注意的是可用资金不等于可用现金，是用总资产、期权持仓市值、冻结资金和初始保证金等指标算出来的一个值，当可用资金大于0时，代表该账户可以开仓，可用资金\*4 为账户的最大可用购买力。算法，可用资金=总资产-美股期权市值-当前总持仓的初始保证金-冻结资金。其中初始保证金的算法是∑（持仓个股市值\*当前股票的开仓保证金比例）。举例：小虎当前账户总资产为10000美元，持有1000美元美股期权，持有总市值为2000美元的苹果，苹果的初始保证金比例当前为45%，没有冻结资金，小虎的可用资金=10000-1000-2000*45%=8100美元
cash_available_for_withdrawal|float|当前账号内可以出金的现金金额
buying_power|float| 最大购买力。 最大购买力是账户最大的可用购买金额。可以用最大购买力的值来估算账户最多可以买多少金额的股票，但是由于每只股票的最多可加的杠杆倍数不一样，所以实际购买单支股票时可用的购买力要根据具体股票保证金比例来计算。算法，最大购买力=4*可用资金。举例：小虎的可用资金是10万美元，那么小虎的最大购买力是40万美元，假设当前苹果股价是250美元一股，苹果的初始保证金比例是30%，小虎最多可以买100000/30%=33.34万美金的苹果，假设苹果的初始保证金比例为25%，小虎最多可以买100000/25%=40万美金的苹果。保证金账户日内最多有四倍于资金（未被占用做保证金的资金）的购买力 隔夜最多有两倍的购买力
gross_position_value|float|证券总价值，是账户持仓证券的总市值之和，即全部持仓的总市值。算法，持仓证券总市值之和；备注：所有持仓市值都会按主币种计算；举例1，若小虎同时持有市值3000美元苹果（也就是做多），持有市值-1000美元的谷歌（也就是做空），小虎证券总价值=3000美元苹果+（-1000美元谷歌）=2000美元；举例2，小虎持有市值1万美元的苹果，市值5000美元的苹果做多期权，小虎证券总价值=10000+5000=15000美元
equity_with_loan|float| 含贷款价值总权益，即ELV，ELV是用来计算开仓和平仓的数据指标；算法，现金账户=现金余额，保证金账户=现金余额+证券总市值-美股期权市值；ELV = 总资产 - 美股期权
net_liquidation|float|总资产(净清算值)。总资产就是我们账户的净清算现金余额和证券总市值之和，通常用来表示目前账户中有多少资产。算法，总资产=证券总市值+现金余额+应计分红-应计利息；举例：小虎账户有1000美元现金，持仓价值1000美元的苹果（即做多苹果），没有待发放的股息和融资利息和未扣除的利息，那么小虎的总资产=现金1000+持仓价值1000=2000美元，若小虎账户有1000美元现金，做空1000美元的苹果，此时证券总市值-1000美元，现金2000美元，用户总资产=现金2000+（持仓市值-1000），账户总资产共计1000美元。
init_margin|float| 初始保证金。当前所有持仓合约所需的初始保证金要求之和。初次执行交易时，只有当含贷款价值总权益大于初始保证金才允许开仓。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。
maintain_margin|float| 维持保证金。当前所有持仓合约所需的维持保证金要求之和。持有头寸时，当含贷款价值总权益小于维持保证金会引发强平。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。
overnight_margin|float| 隔夜保证金。隔夜保证金是在收盘前15分钟开始检查账户所需的保证金，为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。老虎国际的隔夜保证金比例均在50%以上。如果账户含货款价值总权益低于隔夜保证金，账户在收盘前15分钟存在被强制平仓的风险。算法，∑（持仓个股隔夜时段的维持保证金）。个股的维持保证金率用户能够在“个股详情页-报价区-点击融资融券标识"查询。举例：小虎账户总资产为10万美元，苹果开仓保证金比例是40%，日内维持保证金是20%，日内全仓买入苹果共25万美元，到收盘前15分钟（隔夜时段）维持保证金比例将被提高到50%，此时的用户隔夜保证金为：25万*50%=12.5万，用户的含贷款价值总权益为10万小于12.5万，用户将会被平仓部分股票。
unrealized_pl|float| 持仓盈亏。定义，持仓个股、衍生品的未实现盈亏金额；算法，当前价\*股数-持仓成本
realized_pl|float| 已实现盈亏
excess_liquidation|float| 当前剩余流动性。当前剩余流动性是衡量当前账户潜在的被平仓风险的指标，当前剩余流动性越低账户被平仓的风险越高，当小于0时会被强制平掉部分持仓。具体的算法为：当前剩余流动性=含货款价值总权益(equity_with_loan)-账户维持保证金(maintain_margin) 。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。举例：(1)、客户总资产1万美金，买入苹果12000美金（假设苹果开仓和维持保证金比例为50%）。当前剩余流动性=总资产10000-账户维持保证金6000=4000。(2)、随着股票的下跌，假设股票市值跌到8000，这个时候当前剩余流动性=总资产 （-2000+8000）-账户维持保证金4000=2000。(3)、此时用户又买入了1000美金的美股期权，那么账户的当前剩余流动性还剩下2000-1000=1000。若您的账户被强制平仓，则会以市价单在强制平仓时进行成交，强平的股票对象由券商自行决定，请您注意风控值和杠杆等指标。|
overnight_liquidation|float|隔夜剩余流动性。隔夜剩余流动性是指用 含货款价值总权益(equity_with_loan)-隔夜保证金(overnight_margin) 算出来的值。为满足监管机构的保证金要求，我们会在临近收盘前15分钟提升初始保证金与维持保证金要求至最低50%。如果账户的隔夜剩余流动性低于0，在收盘前15分钟起账户存在被强行平掉部分持仓的风险。若您的账户被强制平仓，则会以市价单在强制平仓时进行成交，强平的股票对象由券商自行决定，请您注意风控值和杠杆等指标。
leverage|float| 杠杆。杠杆是衡量账户风险程度的重要指标，可以帮助用户快速了解账户融资比例和风险程度；算法，杠杆=证券市值绝对值之和/总资产；备注1，保证金账户日内最大杠杆4倍，隔夜2倍；备注2，老虎考虑到历史波动、流动性和风险等因素，并不是每只股票都能4倍杠杆买入，一般做多保证金比例范围在25%-100%之间，保证金比例等于25%的股票，可以理解为4倍杠杆买入，保证金比例等于100%的股票，可以理解为0倍杠杆买入，即全部用现金买入。需要留意做空保证金可能会大于100%。备注3，个股保证金比例用户能够在“个股详情页-报价区-点击融资融券标识”查询；举例，小虎账户总资产10万美元，想买苹果，苹果当前个股做多初始保证金比例50%（1/50%=2倍杠杆），小虎最多只能买入20万市值的苹果股票；小虎想做空谷歌，谷歌做空保证金比例200%，小虎最多能做空10/200%=5万谷歌股票；小虎想买入微软，微软初始保证金100%（1/100%=1倍杠杆），小虎最多只能买入10万美元的微软股票
currency_assets|dict|按照交易币种区分的账户资产信息，是以币种为 key 的 dict, 值为 CurrencyAsset 对象

# CurrencyAsset 分币种资产(综合/模拟账户) 
`tigeropen.trade.domain.prime_account.CurrencyAsset` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/prime_account.py)  

**说明**  

按币种区分的资产。  

**对象属性**

属性名|类型|描述
----|----|----
currency|str|当前的货币币种，常用货币包括： USD-美元，HKD-港币，SGD-新加坡币，CNH-人民币
cash_balance|float|可以交易的现金，加上已锁定部分的现金（如已购买但还未成交的股票，还包括其他一些情形也会有锁定现金情况）
cash_available_for_trade|float| 当前账号内可以交易的现金金额
gross_position_value|float|证券或期货总价值
stock_market_value|float|股票的市值, category为C（期货类型）时，不会有股票市值
futures_market_value|float| 期货的市值，category为S（股票类型）时，不会有期货市值
option_market_value|float|期权的市值
unrealized_pl|float|账号内浮动盈亏
realized_pl|float|账号内已实现盈亏

---

# PortfolioAccount 资产(环球账户)
`tigeropen.trade.domain.account.PortfolioAccount` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/account.py)

**说明**  

账户资产对象，适用于环球账户。包含账户的总资产、盈亏、持仓市值、现金、可用资金、保证金、杠杆等相关信息

**对象属性**

属性名|类型|描述
----|----|----
account|str|对应的账户 id
summary|tigeropen.trade.domain.account.Account|账户汇总信息，对segments的统计
segments|dict|按照交易品种区分的账户信息。分别有两个key，'S'对应证券，value为一个[SecuritySegment对象](/zh/python/appendix1/object.html#securitysegment-股票资产-环球账户)，'C' 表示期货，value为一个[CommoditySegment对象](/zh/python/appendix1/object.html#commoditysegment-期货资产-环球账户)。
market_values|dict|按币种区分的账户信息。其中的 key: 'USD' 表示美元， 'HKD' 表示港币; value 是一个 [MarketValue 对象](/zh/python/appendix1/object.html#marketvalue-对象-环球账户)



# Account 汇总资产(环球账户)  
`tigeropen.trade.domain.account.Account` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/account.py)  

**说明**

账户各交易品种资产汇总信息。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|accrued_cash| float|当前月份的累积应付利息，按照日频更新|
|accrued_dividend| float |累计分红. 指的是所有已执行但仍未支付的分红累加值|
|available_funds| float | 可用资金（可用于交易）。 计算方法为 equity_with_loan - initial_margin_requirement|
|∆ buying_power| float |购买力: 预估您还可以购入多少美元的股票资产。保证金账户日内最多有四倍于资金（未被占用做保证金的资金）的购买力。隔夜最多有两倍的购买力|
|cash| float |现金量|
|currency| str |币种, 含义参考枚举参数-币种|
|cushion| float |剩余流动性占总资产的比例，计算方法为: excess_liquidity/net_liquidation|
|∆ day_trades_remaining|int|当日剩余日内交易次数， -1 表示无限制
|equity_with_loan| float |含借贷值股权(含贷款价值资产) 。 证券 Segment: 现金价值 + 股票价值 。 期货 Segment: 现金价值 - 维持保证金|
|excess_liquidity| float |剩余流动性- 证券 Segment: 计算方法: equity_with_loan - maintenance_margin_requirement- 期货 Segment: 计算方法: net_liquidation - maintenance_margin_requirement
|∆ gross_position_value| float |证券总价值: 做多股票的价值+做空股票价值+做多期权价值+做空期权价值。
|initial_margin_requirement| float |初始保证金
|maintenance_margin_requirement| float |维持保证金
|realized_pnl| float |本日已实现盈亏
|unrealized_pnl| float |浮动盈亏
|net_liquidation| float |总资产(净清算价值)。 证券 Segment: 现金价值 + 股票价值 + 股票期权价值。 期货 Segment: 现金价值 + 盯市盈亏
|∆ regt_equity| float |仅针对证券Segment，即根据 Regulation T 法案计算的 equity with loan（含借贷股权值）
|∆ regt_margin| float |仅针对证券Segment， 即根据 Regulation T 法案计算的 initial margin requirements（初始保证金）
|∆ sma| float |仅针对证券Segment。隔夜风控值，每个交易日收盘前10分钟左右对账户持仓的隔夜风险进行检查，隔夜风控值需要大于0，否则会在收盘前对账户部分头寸强制平仓。如果交易日盘中出现隔夜风控值低于0，而时间未到收盘前10分钟，账户不会出发强平。
|timestamp|int|更新时间

# SecuritySegment 股票资产(环球账户) 
`tigeropen.trade.domain.account.SecuritySegment` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/account.py)  

**说明**  

股票资产信息。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|accrued_cash| float |当前月份的累积应付利息，按照日频更新。
|accrued_dividend| float |累计分红. 指的是所有已执行但仍未支付的分红累加值
|available_funds| float |可用资金（可用于交易）。 计算方法为 equity_with_loan - initial_margin_requirement
|cash| float |现金
|equity_with_loan| float |含借贷值股权(含贷款价值资产)。计算方法： 现金价值 + 股票价值 
|excess_liquidity| float |剩余流动性。计算方法: equity_with_loan - maintenance_margin_requirement
|gross_position_value| float |证券总价值: 做多股票的价值+做空股票价值+做多期权价值+做空期权价值。
|initial_margin_requirement| float |初始保证金
|maintenance_margin_requirement|float维持保证金
|leverage| float |仅用于证券 Segment gross_position_value / net_liquidation
|net_liquidation| float |总资产(净清算价值)。 计算方法： 现金价值 + 股票价值 + 股票期权价值
|∆ regt_equity| float |仅针对证券Segment，即根据 Regulation T 法案计算的 equity with loan（含借贷股权值）
|∆ regt_margin| float |仅针对证券Segment， 即根据 Regulation T 法案计算的 initial margin requirements（初始保证金）
|∆ sma| float |仅针对证券Segment。隔夜风控值，每个交易日收盘前10分钟左右对账户持仓的隔夜风险进行检查，隔夜风控值需要大于0，否则会在收盘前对账户部分头寸强制平仓。如果交易日盘中出现隔夜风控值低于0，而时间未到收盘前10分钟，账户不会出发强平。
|timestamp|int|更新时间


# CommoditySegment 期货资产(环球账户)
`tigeropen.trade.domain.account.CommoditySegment` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/account.py)  

**说明** 

期货资产信息。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|accrued_cash| float |当前月份的累积应付利息，按照日频更新。
|accrued_dividend| float |累计分红. 指的是所有已执行但仍未支付的分红累加值
|available_funds| float |可用资金（可用于交易）。 计算方法为 equity_with_loan - initial_margin_requirement
|cash| float |现金
|equity_with_loan| float |含借贷值股权(含贷款价值资产)计算方法：现金价值 - 维持保证金
|excess_liquidity| float |剩余流动性。计算方法: net_liquidation - maintenance_margin_requirement
|initial_margin_requirement| float |初始保证金
|maintenance_margin_requirement| float | 维持保证金
|net_liquidation|float|总资产(净清算价值)。计算方法：现金价值 + 盯市盈亏
|timestamp| int |更新时间


# MarketValue 分币种资产(环球账户) 
`tigeropen.trade.domain.account.MarketValue` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/account.py) 

**说明**  

按币种区分的资产信息。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|currency| str|货币单位|
|net_liquidation| float |总资产(净清算价值)|
|cash_balance| float |现金|
|stock_market_value| float |股票市值|
|option_market_value| float |期权市值|
|warrant_value| float |窝轮市值|
|futures_pnl| float |盯市盈亏|
|unrealized_pnl| float |未实现盈亏|
|realized_pnl| float |已实现盈亏|
|exchange_rate| float |对账户主币种的汇率|
|net_dividend| float |应付股息与应收股息的净值|
|timestamp|int|更新时间|

---

# Position 持仓 
`tigeropen.trade.domain.position.Position` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/position.py)  

**说明**  

账户持仓信息。包括持仓的合约标的、持仓数量、成本、盈亏等信息。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|account|str|对应的账户ID
|contract|tigeropen.trade.domain.contract.Contract|合约对象
|quantity|int|合约数量
|average_cost|float|含佣金的平均成本
|market_price|float|市价
|market_value|float|市值
|realized_pnl|float|已实现盈亏
|unrealized_pnl|float|浮动盈亏

---

# Order 订单
`tigeropen.trade.domain.order.Order` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/order.py)  

**说明**  

订单对象。查询订单会返回该对象，下单改单的参数也需要使用该对象。

**对象属性**

|属性名|类型|描述|
|----|----|----|
|account|str|订单所属的账户 
|id|long|全局订单 id
|order_id|int|账户自增订单号，已废弃
|parent_id|long|母订单id，目前只用于 TigerTrade App端的附加订单中
|order_time|int|下单时间，毫秒单位13位数字时间戳
|reason|str|下单失败时，会返回失败原因的描述
|trade_time|int|最新成交时间，毫秒单位13位数字时间戳
|action|str|交易方向， 'BUY' / 'SELL'
|quantity|int|下单数量
|filled|int|成交数量
|avg_fill_price|float|包含佣金的平均成交价
|commission|float|包含佣金、印花税、证监会费等系列费用
|realized_pnl| float |实现盈亏
|trail_stop_price| float |跟踪止损价格
|limit_price| float |限价单价格
|aux_price| float |在止损单中，表示出发止损单的价格， 在移动止损单中， 表示跟踪的价差
|trailing_percent| float |跟踪止损单-百分比，取值范围为0-100
|percent_offset| float | <该字段未使用>
|order_type|str|订单类型, 'MKT'市价单/'LMT'限价单/'STP'止损单/'STP_LMT'止损限价单/'TRAIL'跟踪止损单
|time_in_force|str|有效期,'DAY'日内有效/'GTC'撤销前有效
|outside_rth|bool|是否支持盘前盘后交易，美股专属。
|contract|Contract|[tigeropen.trade.domain.contract.Contract](/zh/python/appendix1/object.html#contract-合约) 合约对象
|status|OrderStatus|[tigeropen.common.consts.OrderStatus](/zh/python/appendix2/#订单状态) 枚举， 表示订单状态
|remaining|int|未成交的数量

## 构建方法：

通过 SDK 中的 [tigeropen.common.util.order_utils](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/common/util/order_utils.py) 在本地生成订单对象：
order_utils 仅提供了常用的参数，如果需要额外的参数，可生成订单对象后，修改其属性即可
```python
from tigeropen.common.util.contract_utils import stock_contract
from tigeropen.common.util.order_utils import (market_order,        # 市价单
                                            limit_order,         # 限价单
                                            stop_order,          # 止损单
                                            stop_limit_order,    # 限价止损单
                                            trail_order,         # 移动止损单
                                            order_leg)           # 附加订单
                        
contract = stock_contract('AAPL', currency='USD')
order = limit_order('your account', contract, 'BUY', 100, 150.5)
order.time_in_force = 'GTC' # 设置订单属性
     
# 后续操作...                                    
```

### market_order 市价单

```python
market_order(account, contract, action, quantity)
```

**参数**

参数名|类型|描述
----|----|----
account|str|下单账户， 可以使用综合账户、环球账户或模拟账户
contract|tigeropen.trade.domain.contract.Contract|要交易的[合约对象](/zh/python/appendix1/object.html#contract-合约)
action|str|交易方向， 'BUY' / 'SELL'
quantity|int|下单数量，必须为大于0的整数

**返回**

`Order` 对象

### limit_order 限价单

`
limit_order(account, contract, action, quantity, limit_price)
`

**参数**

参数名|类型|描述
----|----|----
account|str|下单账户， 可以使用综合账户、环球账户或模拟账户
contract|tigeropen.trade.domain.contract.Contract|要交易的[合约对象](/zh/python/appendix1/object.html#contract-合约)
action|str|交易方向， 'BUY' / 'SELL'
quantity|int|下单数量，必须为大于0的整数
limit_price|float| 限价的价格，必须大于0

**返回**

`Order` 对象

### stop_order 止损单

`
stop_order(account, contract, action, quantity, aux_price)
`

**参数**

参数名|类型|描述
----|----|----
account|str|下单账户， 可以使用综合账户、环球账户或模拟账户
contract|tigeropen.trade.domain.contract.Contract|要交易的[合约对象](/zh/python/appendix1/object.html#contract-合约)
action|str|交易方向， 'BUY' / 'SELL'
quantity|int|下单数量，必须为大于0的整数
aux_price|float|触发止损单的价格，必须大于0

**返回**

`Order` 对象


### stop\_limit\_order 限价止损单

`
stop_limit_order(account, contract, action, quantity, limit_price, aux_price)
`

**参数**

参数名|类型|描述
----|----|----
account|str|下单账户， 可以使用综合账户、环球账户或模拟账户
contract|tigeropen.trade.domain.contract.Contract|要交易的[合约对象](/zh/python/appendix1/object.html#contract-合约)
action|str|交易方向， 'BUY' / 'SELL'
quantity|int|下单数量，必须为大于0的整数
limit_price|float| 限价的价格，必须大于0
aux_price|float|触发止损单的价格，必须大于0

**返回**

`Order` 对象

### trail_order 移动止损单

`
trail_order(account, contract, action, quantity, trailing_percent=None, aux_price=None)
`

**参数**

参数名|类型|描述
----|----|----
account|str|下单账户， 可以使用综合账户、环球账户或模拟账户
contract|tigeropen.trade.domain.contract.Contract|要交易的[合约对象](/zh/python/appendix1/object.html#contract-合约)
action|str|交易方向， 'BUY' / 'SELL'
quantity|int|下单数量，必须为大于0的整数
trailing_percent|float|跟踪止损单-百分比，取值范围为0-100, aux_price和trailing_percent两者互斥
aux_price|float| 价差， 与 trailing_percent 互斥

**返回**

`Order` 对象

### order_leg 附加订单

`
order_leg(leg_type, price, time_in_force='DAY', outside_rth=None)
`

**参数**

参数名|类型|描述
----|----|----
leg_type|str|附加订单类型. 'PROFIT' 止盈单类型,  'LOSS' 止损单类型
price|float|附加订单价格
time_in_force|str| 附加订单有效期. 'DAY'（当日有效）和'GTC'（取消前有效 Good-Til-Canceled).
outside_rth|bool|附加订单是否允许盘前盘后交易(美股专属). True 允许, False 不允许

**返回**

`OrderLeg` 对象 tigeropen.trade.domain.order.OrderLeg

# Transaction 成交记录
`tigeropen.trade.domain.order.Transaction`

**说明**
订单的成交记录

**对象属性**
对象属性|类型|描述
----|----|----
account|str|账户
order_id|int|订单id
contract|Contract|[合约对象](/zh/python/appendix1/object.html#contract-合约)
id|int|成交记录id
action|str|订单方向
filled_quantity|int|成交数量
filled_price|float|成交价格
filled_amount|float|成交金额
transacted_at|str|成交时间

# OrderLeg 附加订单
`tigeropen.trade.domain.order.OrderLeg`

**说明**  

下主订单同时携带的附加订单

**对象属性**

对象属性|类型|描述
----|----|----
leg_type|str|附加订单类型. 'PROFIT' 止盈单类型,  'LOSS' 止损单类型
price|float|附加订单价格.
time_in_force|str| 附加订单有效期. 'DAY'（当日有效）和'GTC'（取消前有效 Good-Til-Canceled).
outside_rth|bool|附加订单是否允许盘前盘后交易(美股专属). True 允许, False 不允许.


# AlgoParams 算法订单参数  
`tigeropen.trade.domain.order.AlgoParams`

**说明**  

算法订单(VWAP/TWAP)参数对象

**对象属性**

对象属性|类型|描述
----|----|----
start_time|str或int|生效开始时间(时间字符串或时间戳，TWAP和VWAP专用)，如 '2020-11-19 23:00:00' 或 1640159945678
end_time|str或int|失效时间(时间字符串或时间戳 TWAP和VWAP专用)
no_take_liq|bool|是否尽可能减少交易次数(VWAP订单专用)
allow_past_end_time|bool|是否允许失效时间后继续完成成交(TWAP和VWAP专用)
participation_rate|float|参与率(VWAP专用,0.01-0.5)


# Contract 合约
`tigeropen.trade.domain.contract.Contract` [source](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/domain/contract.py)  

**说明**  

合约是指交易的买卖对象或者标的物（比如一只股票，或者一个期权），合约是由交易所统一制定的。比如购买老虎证券的股票，可以通过TIGR这个字母代号和市场信息（即market=’US‘，美国市场）来唯一标识。类似的在购买期权或者期货产品时，可能会需要用到其他一些标识字段。通过合约信息，我们在下单或者获取行情时就可以唯一的确定一个标的物了。在Open API Python SDK中，合约信息通过 tigeropen.trade.domain.contract.Contract 对象来保存。Contract 对象可传入构造 Order 订单对象的工具函数中创建 Order 对象，用于下单

常见的合约包括股票合约，期权合约，期货合约等，大部分合约包括如下几个要素:

* 标的代码(symbol)，一般美股、英股等合约代码都是英文字母，港股、A股等合约代码是数字，比如老虎证券的symbol是TIGR。
* 合约类型(security type)，常见合约类型包括：STK（股票），OPT（期权），FUT（期货），CASH（外汇），比如老虎证券股票的合约类型是STK。
* 货币类型(currency)，常见货币包括 USD（美元），HKD（港币）。
* 交易所(exchange)，STK类型的合约一般不会用到交易所字段，订单会自动路由，期货合约都用到交易所字段。

绝大多数股票，差价合约，指数或外汇对可以通过这四个属性来唯一确定。由于其性质，更复杂的合约（如期权和期货）需要一些额外的信息。以下是几种常见类型合约，以及其由哪些要素构成：

股票：

```python
from tigeropen.common.util.contract_utils import stock_contract
contract = stock_contract(symbol='TIGR', currency='USD')
contract1 = stock_contract(symbol='00700', currency='HKD')
```

期权

```python
from tigeropen.common.util.contract_utils import option_contract
contract = option_contract(identifier='AAPL  190118P00160000')
```

期货

```python
from tigeropen.common.util.contract_utils import future_contract
contract = future_contract(symbol='CL', currency='USD', expiry='20190328', multiplier=1.0, exchange='SGX')
```

具体字段及构造方法见下文

**对象属性**

|属性名|类型|描述|
|----|------|----|
|symbol|str|合约代码 
|contract_id|int| 合约id
|sec_type|str|证券类型
|name|str|合约名称
|currency|str|币种
|exchange|str|交易所
|expiry|str|到期日(适用于期权/期货）
|strike|float|行权价(适用于期权)
|multiplier|float|每手数量
|put_call|str|期权方向, 看涨看跌
|local_symbol|str||  
|short_margin|float|做空保证金比例
|short_fee_rate|float|做空费率
|shortable|int|做空池剩余
|long_initial_margin|float|做多初始保证金
|long_maintenance_margin|float|做多维持保证金
|contract_month|str|合约月份， 如202201，表示2022年1月
|identifier|str|合约标识符
|primary_exchange|str|股票上市交易所
|market|str|市场
|min_tick|float|最小报价单位
|trading_class|str|合约的交易级别名称
|status|str|状态
|continuous|bool|期货专有，是否为连续合约
|trade|bool|期货专有，是否可交易
|last_trading_date|str|期货专有，最后交易日，如 '20211220'，表示2021年12月20日
|first_notice_date|str|期货专有，第一通知日，合约在第一通知日后无法开多仓. 已有的多仓会在第一通知日之前（通常为前三个交易日）被强制平仓，如 '20211222'，表示2021年12月22日
|last_bidding_close_time|int|期货专有，竞价截止时间戳

注意: print时只会显示部分属性

## 构建方法

Contract 对象可通过以下工具函数来生成:

获取合约信息 `get_contract`/`get_contracts`

**参数**:

| 参数 | 是否必填 | 描述 |
|:--|:--|:--|
| symbol | Yes | 合约代码 如 00700/AAPL |
| sec_type | Yes | 合约类型 如 SecurityType.STK/SecurityType.OPT |
| currency | No | 币种 如 Currency.USD/Currency.HKD |
| exchange | No | 交易所 如 SMART/SEHK |
| expiry | No | 到期日 交易品种是期权时必传 yyyyMMdd |
| strike | No | 行权价 交易品种是期权时必传 |
| put_call | No | CALL/PUT 交易品种是期权时必传 |
| secret_key | No | 机构交易员密钥，机构用户专有，需要在client_config中配置

**返回**

get_contract 返回 Contract 对象; get_contracts 返回 Contract 对象列表. 对象属性参见上文的说明部分

**示例**
```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

# 获取股票合约
contract = trade_client.get_contract('FB', sec_type=SecurityType.STK)
contracts = trade_client.get_contracts(['AAPL', 'TSLA'], sec_type=SecurityType.STK)

# 获取期货合约
fut_contract = trade_client.get_contract('CL', sec_type=SecurityType.FUT)

# 获取期权合约
opt_contract = trade_client.get_contract('SPY', sec_type=SecurityType.OPT, expiry='20231215', strike=435.0, put_call='CALL')
```

---
# MarketStatus 市场状态
`tigeropen.quote.domain.market_status.MarketStatus`

**说明**

市场交易状态

**对象属性**  

对象属性|类型|描述
----|----|----
market|str|市场。（US:美股，CN:沪深，HK:港股）
trading_status|str|市场交易状态码。 未开盘 NOT_YET_OPEN; 盘前交易 PRE_HOUR_TRADING; 交易中 TRADING; 午间休市 MIDDLE_CLOSE; 盘后交易 POST_HOUR_TRADING; 已收盘 CLOSING; 提前休市 EARLY_CLOSED; 休市 MARKET_CLOSED;
status|str|市场状态描述(未开盘，交易中，休市等）
open_time|datetime.datetime|最近开盘时间


# OptionFilter 期权链过滤器
`tigeropen.quote.domain.filter.OptionFilter`  

**说明** 

期权过滤参数对象

**对象属性**

| 参数  | 类型  | 是否必填   | 描述 |  
| :--- | :--- | :--- | :--- |
|implied_volatility|float| No |隐含波动率, 反映市场预期的未来股价波动情况, 隐含波动率越高, 说明预期股价波动越剧烈 |
|in_the_money  |bool|   No |是否价内|
|open_interest |int  |No   |未平仓量, 每个交易日完结时市场参与者手上尚未平仓的合约数. 反映市场的深度和流动性 |
|delta   |float|  No|   delta, 反映正股价格变化对期权价格变化对影响. 股价每变化1元, 期权价格大约变化 delta. 取值 -1.0 ~ 1.0 |
|gamma   |float|  No|   gamma, 反映正股价格变化对于delta的影响. 股价每变化1元, delta变化gamma |
|theta   |float|  No|   theta, 反映时间变化对期权价格变化的影响. 时间每减少一天, 期权价格大约变化 theta |
|vega |float|  No|   vega, 反映波动率对期权价格变化的影响. 波动率每变化1%, 期权价格大约变化 vega  |
|rho  |float|  No|   rho, 反映无风险利率对期权价格变化的影响. 无风险利率每变化1%, 期权价格大约变化 rho | 
