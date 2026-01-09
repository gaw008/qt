---
title: 获取合约
---
### 合约说明
合约是指交易的买卖对象或者标的物（比如一只股票，或者一个期权），合约是由交易所统一制定的。比如购买老虎证券的股票，可以通过TIGR这个字母代号和市场信息（即market=’US‘，美国市场）来唯一标识。类似的在购买期权或者期货产品时，可能会需要用到其他一些标识字段。通过合约信息，我们在下单或者获取行情时就可以唯一的确定一个标的物了。

常见的合约包括股票合约，期权合约，期货合约等。

大部分合约包括如下几个要素:

* 标的代码(symbol)，一般美股、英股等合约代码都是英文字母，港股、A股等合约代码是数字，比如老虎证券的symbol是TIGR。
* 合约类型(security type)，常见合约类型包括：STK（股票），OPT（期权），FUT（期货），CASH（外汇），比如老虎证券股票的合约类型是STK。
* 货币类型(currency)，常见货币包括 USD（美元），HKD（港币）。
* 交易所(exchange)，STK类型的合约一般不会用到交易所字段，订单会自动路由，期货合约都用到交易所字段。

绝大多数股票，差价合约，指数或外汇对可以通过这四个属性来唯一确定。

由于其性质，更复杂的合约（如期权和期货）需要一些额外的信息。

### get_contract 获取单个合约信息
`
TradeClient.get_contract(symbol, sec_type=SecurityType.STK, currency=None, exchange=None, expiry=None, strike=None,
                 put_call=None)
`

**说明**

查询交易所需的单个合约信息

**参数**
参数名|类型|描述
----|----|----
symbol|str|股票代码，如 'AAPL'
sec_type| SecurityType | 证券类型，tigeropen.common.consts.SecurityType 枚举，如 SecurityType.STK
currency| Currency | 币种，tigeropen.common.consts.Currency 枚举，如 Currency.USD
exchange| str | 交易所，非必填，如 'CBOE'
expiry|str|合约到期日（适用于期货/期权），格式 yyyyMMdd，如 ‘20220130’
strike |float|行权价（适用于期权）
put_call|str|看涨看跌（适用于期权），'PUT' 看跌， 'CALL' 看涨

**返回**

`tigeropen.trade.domain.contract.Contract` 合约对象, 参见[对象介绍](/zh/python/appendix1/object.html#contract-合约)。常用属性如下

参数名|类型|描述
----|----|----
currency|str|币种
symbol|str|合约代码
exchange|str|交易所
short_fee_rate|float|卖空参考利率
shortable|int|卖空池余额
short_margin|float|卖空保证金
long_initial_margin|float|做多初始保证金
long_maintenance_margin|float|做多维持保证金


**示例**
```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

contract = trade_client.get_contract('AAPL', sec_type=SecurityType.STK)
print(contract)
```

**返回示例**
```python
AAPL/STK/USD
```

### get_contracts 获取多个合约信息

`
TradeClient.get_contracts(symbol, sec_type=SecurityType.STK, currency=None, exchange=None):
`

**说明**

查询交易所需的多个合约信息，以列表的形式返回

**参数**

参数名|类型|描述
----|----|----
symbol|str|股票代码，如 'AAPL'
sec_type| SecurityType | 证券类型，tigeropen.common.consts.SecurityType 枚举，如 SecurityType.STK
currency| Currency | 币种，tigeropen.common.consts.Currency 枚举，如 Currency.USD
exchange| str | 交易所，非必填，如 'CBOE'

**返回** 

`list` 

列表中每一项为合约对象（tigeropen.trade.domain.contract.Contract），参见[对象介绍](/zh/python/appendix1/object.html#contract-合约)。常用属性如下

参数名|类型|描述
----|----|----
currency|str|币种
symbol|str|合约代码
exchange|str|交易所
short_fee_rate|float|卖空参考利率
shortable|int|卖空池余额
short_margin|float|卖空保证金
long_initial_margin|float|做多初始保证金
long_maintenance_margin|float|做多维持保证金

**示例**
```python
from tigeropen.trade.trade_client import TradeClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account', secret_key='机构交易员专有密钥')
trade_client = TradeClient(client_config)

contracts = trade_client.get_contracts('AAPL', sec_type=SecurityType.STK)
print(contracts)
```
**返回示例**
```python
[AAPL/STK/USD, JD/STK/USD]
```

### 本地生成合约对象
股票：

```python
from tigeropen.common.util.contract_utils import stock_contract
contract = stock_contract(symbol='TIGR', currency='USD')
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
