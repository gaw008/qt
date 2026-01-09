---
title: 查询账户信息
---
### get\_managed\_accounts 获取管理的账号列表
`
TradeClient.get_managed_accounts(account=None)
`

**说明**

获取本tiger_id关联的资金账号

**参数**

参数名|类型|描述
----|----|----
account|str|账号id，选填，不传则返回所有关联的 account

**返回**

`AccountProfile`（tigeropen.trade.domain.profile.AccountProfile）对象构成的列表

属性如下：

参数名|类型|描述
----|----|----
account|str|交易账户
capability|str| 账户类型(CASH:现金账户, RegTMargin: Reg T 保证金账户, PMGRN: 投资组合保证金)
status|str|账户状态(New, Funded, Open, Pending, Abandoned, Rejected, Closed, Unknown)

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

accounts = trade_client.get_managed_accounts()
```

**返回示例**
```python
[AccountProfile({'account': 'DU575569', 'capability': None, 'status': 'Funded'})]
```


### get_assets 获取账户资产信息

`
TradeClient.get_assets(account=None, sub_accounts=None, segment=False, market_value=False)
`

**说明**

获取账户资产信息，返回结构主要适用于环球账户，综合/模拟账号虽然也可使用此接口，但有较多字段为空，请使用 [get_prime_assets](/zh/python/operation/trade/accountInfo.html#get-prime-assets-获取综合-模拟账户资产信息) 查询综合/模拟账户资产

**参数**

参数名|类型|描述
----|----|----
account|str|账户id
sub_accounts|list\<str...\>|子账户列表，默认为None
segment|bool|是否返回按照品种（证券、期货）分类的数据，默认 False，为True时，返回一个dict，C表示期货， S表示股票
market_value|bool|是否返回按照币种（美元、港币、人民币）分类的数据，默认为 False
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置

**返回**

`list`

list 中的每个元素是一个 [PortfolioAccount 对象](/zh/python/appendix1/object.html#portfolioaccount-资产-环球账户)。如果只有一个account，list 中只有一个元素。
PortfolioAccount(tigeropen.trade.domain.account.PortfolioAccount) 对象的结构如下。

**Account、SecuritySegment、CommoditySegment 中的信息请参考[对象信息](/zh/python/appendix1/object.html#portfolioaccount-资产-综合-模拟账户)**

```
PortfolioAccount 对象
├── account：账户id
├── summary：当前账户的汇总统计信息。里面的值是一个 Account 对象
├── segments：分品种的账户信息,是一个dict
│   ├── 'S' 表示证券账户，value 是 SecuritySegment 对象
│   └── 'C' 表示期货账户，value 是 CommoditySegment 对象
├── market_value：分币种的账户统计信息，是一个dict
│   ├── 'USD' 表示美元，value 是 MarketValue 对象
│   ├── 'HKD' 表示港币，value 是 MarketValue 对象
└─  └── 'CNH' 表示人民币，value 是 MarketValue 对象
```

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

portfolio_account = trade_client.get_assets(segment=True, market_value=True)
print(portfolio_account)

[PortfolioAccount({'account': 'DU575569', 
                    'summary': Account({'accrued_cash': 0, 'accrued_dividend': 0, 'available_funds': 948.69, 
                                        'buying_power': 948.69, 'cash': 948.81, 'currency': 'USD', 'cushion': 0.5944, 
                                        'day_trades_remaining': 3, 'equity_with_loan': 1255.69, 'excess_liquidity': 948.81, 
                                        'gross_position_value': 647.53, 'initial_margin_requirement': 307, 
                                        'maintenance_margin_requirement': 307, 'net_liquidation': 1596.34, 
                                        'realized_pnl': 0, 'regt_equity': 1255.81, 
                                        'regt_margin': 153.5, 'sma': 3512.56, 
                                        'timestamp': 1561529631, 'unrealized_pnl': -885.36}), 

                    'segments': defaultdict(<class 'tigeropen.trade.domain.account.Account'>, 
                            {'C': CommoditySegment({'accrued_cash': 0, 'accrued_dividend': 0, 
                                                    'available_funds': 0, 'cash': 0, 'equity_with_loan': 0, 'excess_liquidity': 0, 
                                                    'initial_margin_requirement': 0, 'maintenance_margin_requirement': 0, 
                                                    'net_liquidation': 0, 'timestamp': 1544393719}), 
                            'S': SecuritySegment({'accrued_cash': 0, 'accrued_dividend': 0, 
                                                'available_funds': 120.73, 'cash': 120.73, 'equity_with_loan': 1292.04, 
                                                'excess_liquidity': 120.73, 'gross_position_value': 1171.31, 
                                                'initial_margin_requirement': 1171.31, 'leverage': 0.91, 
                                                'maintenance_margin_requirement': 1171.31, 'net_liquidation': 1292.04, 
                                                'regt_equity': 1292.04, 'regt_margin': 585.66, 'sma': 1973.39, 'timestamp': 1545206069})
                            }), 

                    'market_values': defaultdict(<class 'tigeropen.trade.domain.account.MarketValue'>, 
                            {'CNH': MarketValue({'currency': 'CNH', 'net_liquidation': 0, 'cash_balance': 0, 
                                '               stock_market_value': 0, 'option_market_value': 0, 
                                                'warrant_value': 0, 'futures_pnl': 0, 'unrealized_pnl': 0, 
                                                'realized_pnl': 0, 'exchange_rate': 0.14506, 
                                                'net_dividend': 0, 'timestamp': 1544078822}), 
                            'HKD': MarketValue({'currency': 'HKD', 'net_liquidation': 0, 'cash_balance': 0, 
                                                'stock_market_value': 0, 'option_market_value': 0, 
                                                'warrant_value': 0, 'futures_pnl': 0, 'unrealized_pnl': 0, 
                                                'realized_pnl': 0, 'exchange_rate': 0.12743, 
                                                'net_dividend': 0, 'timestamp': 1550158606}), 
                            'USD': MarketValue({'currency': 'USD', 'net_liquidation': 1596.34, 
                                                'cash_balance': 948.81, 'stock_market_value': 307, 
                                                'option_market_value': 340.53, 'warrant_value': 0, 
                                                'futures_pnl': 0, 'unrealized_pnl': -885.36, 
                                                'realized_pnl': 0, 'exchange_rate': 1, 
                                                'net_dividend': 0, 'timestamp': 1561519773})}
                            )}
                    )]
```

### get\_prime\_assets 获取综合/模拟账户资产信息

`
TradeClient.get_prime_assets(account=None)
`

**说明**

获取资产信息，适用于综合/模拟账户

**参数**

参数名|类型|描述
----|----|----
account|str|账户id, 如不指定, 则使用 client_config 中的默认 account

**返回**

`list`

list 中的每个元素是一个 [PortfolioAccount 对象](/zh/python/appendix1/object.html#portfolioaccount-资产-综合-模拟账户)。如果只有一个account，list 中只有一个元素。
PortfolioAccount 对象的结构如下。

**PortfolioAccount、Segment 中的字段详细解释请参考[对象信息](/zh/python/appendix1/object.html#segment-分品种资产-综合-模拟账户)**

```
PortfolioAccount 对象
├── account：账户id
├── update_timestamp: 更新时间, 毫秒单位的时间戳
├── segments：分品种的账户信息,是以证券类别为 key 的 dict, 值为 Segment 对象
│   ├── 'S' 表示证券账户，value 是 Segment 对象
│   │    ├── currency: 币种, 如 USD, HKD
│   │    ├── capability: 账户类型, 保证金账户: RegTMargin, 现金账户: Cash。
│   │    ├── category': 交易品种分类 C: (Commodities 期货), S: (Securities 股票)
│   │    ├── cash_balance': 现金额。
│   │    ├── cash_available_for_trade': 可用资金。
│   │    ├── cash_available_for_withdrawal': 当前账号内可以出金的现金金额
│   │    ├── gross_position_value': 证券总价值
│   │    ├── equity_with_loan': 含贷款价值总权益
│   │    ├── net_liquidation': 总资产;净清算值
│   │    ├── init_margin': 初始保证金
│   │    ├── maintain_margin': 维持保证金
│   │    ├── overnight_margin': 隔夜保证金
│   │    ├── unrealized_pl': 浮动盈亏
│   │    ├── realized_pl': 已实现盈亏
│   │    ├── excess_liquidation': 当前剩余流动性
│   │    ├── overnight_liquidation': 隔夜剩余流动性
│   │    ├── buying_power': 购买力
│   │    ├── leverage': 当前使用的杠杆倍数
│   │    ├── currency_assets：按照交易币种区分的账户资产信息，是以币种为 key 的 dict
│   │    │   ├── 'USD' 表示美元，value 是 CurrencyAsset 对象
│   │    │   │   ├── currency': 当前的货币币种，常用货币包括： USD-美元，HKD-港币，SGD-新加坡币，CNH-人民币
│   │    │   │   ├── cash_balance': 可以交易的现金，加上已锁定部分的现金（如已购买但还未成交的股票，还包括其他一些情形也会有锁定现金情况）
│   │    │   │   ├── cash_available_for_trade': 当前账号内可以交易的现金金额
│   │    │   │   ├── gross_position_value': 总价值
│   │    │   │   ├── stock_market_value': 股票的市值, category为C（期货类型）时，不会有股票市值
│   │    │   │   ├── futures_market_value': 期货的市值，category为S（股票类型）时，不会有期货市值
│   │    │   │   ├── option_market_value': 期权的市值
│   │    │   │   ├── unrealized_pl': 账号内浮动盈亏
│   │    │   │   ├── realized_pl': 账号内已实现盈亏
│   │    │   ├── 'HKD' 表示港币，value 是 CurrencyAsset 对象
│   │    └─  └── 'CNH' 表示人民币，value 是 CurrencyAsset 对象
│   └── 'C' 表示期货账户，value 是 Segment 对象

```

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥, 个人开发者无需填写')
trade_client = TradeClient(client_config)

portfolio_account = trade_client.get_prime_assets()
print(portfolio_account)

>>> PortfolioAccount({'account': '1234567', 
                    'update_timestamp': 1638949616442, 
                    'segments': 
                        {'S': Segment({'currency': 'USD', 'capability': 'RegTMargin', 'category': 'S', 'cash_balance': 111978.7160247, 'cash_available_for_trade': 123905.775195, 'cash_available_for_withdrawal': 123905.775195, 'gross_position_value': 22113.5652986, 'equity_with_loan': 134092.2813233, 'net_liquidation': 135457.2802984, 'init_margin': 9992.3764097, 'maintain_margin': 8832.4423281, 'overnight_margin': 11607.5876493, 'unrealized_pl': -1121.0821891, 'realized_pl': -3256.0, 'excess_liquidation': 125259.8389952, 'overnight_liquidation': 122484.693674, 'buying_power': 495623.1007801, 'leverage': 0.164693, 
                                        'currency_assets': 
                                            {'USD': CurrencyAsset({'currency': 'USD', 'cash_balance': 123844.77, 'cash_available_for_trade': 123792.77, 'gross_position_value': 9460.69, 'stock_market_value': 9460.69, 'futures_market_value': inf, 'option_market_value': 0.0, 'unrealized_pl': 4183.4433333, 'realized_pl': -3256.0}), 
                                            'HKD': CurrencyAsset({'currency': 'HKD', 'cash_balance': -92554.07, 'cash_available_for_trade': -93664.15, 'gross_position_value': 98691.2, 'stock_market_value': 98691.2, 'futures_market_value': inf, 'option_market_value': 0.0, 'unrealized_pl': -41374.784536, 'realized_pl': 0.0}), 
                                            'CNH': CurrencyAsset({'currency': 'CNH', 'cash_balance': 0.0, 'cash_available_for_trade': 0.0, 'gross_position_value': inf, 'stock_market_value': inf, 'futures_market_value': inf, 'option_market_value': inf, 'unrealized_pl': inf, 'realized_pl': inf})}}), 
                        'C': Segment({'currency': 'USD', 'capability': 'RegTMargin', 'category': 'C', 'cash_balance': 3483681.32, 'cash_available_for_trade': 3481701.32, 'cash_available_for_withdrawal': 3481701.32, 'gross_position_value': 1000000.0, 'equity_with_loan': 3481881.32, 'net_liquidation': 3483681.32, 'init_margin': 1980.0, 'maintain_margin': 1800.0, 'overnight_margin': 1800.0, 'unrealized_pl': 932722.41, 'realized_pl': -30.7, 'excess_liquidation': 3481881.32, 'overnight_liquidation': 3481881.32, 'buying_power': 0.0, 'leverage': 0.0, 
                                        'currency_assets': 
                                            {'USD': CurrencyAsset({'currency': 'USD', 'cash_balance': 3483681.32, 'cash_available_for_trade': 3483681.32, 'gross_position_value': 1000000.0, 'stock_market_value': inf, 'futures_market_value': 1000000.0, 'option_market_value': inf, 'unrealized_pl': 932722.41, 'realized_pl': -30.7}), 
                                            'HKD': CurrencyAsset({'currency': 'HKD', 'cash_balance': 0.0, 'cash_available_for_trade': 0.0, 'gross_position_value': inf, 'stock_market_value': inf, 'futures_market_value': inf, 'option_market_value': inf, 'unrealized_pl': inf, 'realized_pl': inf}), 
                                            'CNH': CurrencyAsset({'currency': 'CNH', 'cash_balance': 0.0, 'cash_available_for_trade': 0.0, 'gross_position_value': inf, 'stock_market_value': inf, 'futures_market_value': inf, 'option_market_value': inf, 'unrealized_pl': inf, 'realized_pl': inf})}})
                        }
                        }
                    )

```


### get_positions 获取持仓数据

`
TradeClient.get_positions(account=None, sec_type=SecurityType.STK, currency=Currency.ALL, market=Market.ALL, symbol=None, sub_accounts=None)
`

**说明**

获取账户的持仓信息

**参数**  

参数名|类型|描述
----|----|----
account|str|账户id，如不指定, 则使用 client_config 中的默认 account
sec_type|SecurityType|交易品种，包括 STK/OPT/FUT 等, 默认 STK, 可以从 [tigeropen.common.consts.SecurityType](/zh/python/appendix2/#合约类型) 下导入
currency|Currency|币种，包括 ALL/USD/HKD/CNH 等, 默认 ALL, 可以从 [tigeropen.common.consts.Currency](/zh/python/appendix2/#货币类型) 下导入
market|Market|市场，包括 ALL/US/HK/CN 等, 默认 ALL, 可以从 [tigeropen.common.consts.Market](/zh/python/appendix2/#市场) 下导入
symbol|str|证券代码
sub_account|list\<str...\>|子账户列表
secret_key|str|机构交易员密钥，机构用户专有，需要在client_config中配置，个人开发者无需关注

**返回**

`list`

结构如下：

每个元素是一个 [Position 对象](/zh/python/appendix1/object.html#position-持仓)。Position(`tigeropen.trade.domain.position.Position`)对象有如下的属性：

参数名|类型|描述
----|----|----
account|str|所属账户
contract|Contract|合约对象，[tigeropen.trade.domain.contract.Contract](/zh/python/appendix1/object.html#contract-合约)
quantity|int|持仓数量
average_cost|float|持仓成本
market_price|float|最新价格
market_value|float|市值
realized_pnl|float|已实现盈亏
unrealized_pnl|float|浮动盈亏

**示例**

```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

accounts = trade_client.get_positions(sec_type=SecurityType.STK, currency=Currency.ALL, market=Market.ALL)
```

**返回示例**
```python
[contract: BABA/STK/USD, quantity: 1, average_cost: 178.99, market_price: 176.77,
contract: BIDU/STK/USD, quantity: 3, average_cost: 265.4633, market_price: 153.45,
contract: SPY/STK/USD, quantity: 7, average_cost: 284.9243, market_price: 284.97]
```
