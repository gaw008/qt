---
title: 交易
---

### 常见下单失败错误及排查方法

如果您遇到了下单失败方面的问题，请首先参考下文常见错误列表排查。若以下回答不能解决您的问题，请[联系我们](/zh/python/feedback/#官方量化qq群)排查故障

为了更快为您定位问题，建议您准备好以下信息
- 用户id
- order_id（若有），可通过get_orders()查询
- 完整错误信息及报错的代码（若在QQ群等公众平台反馈问题，请务必注意在发布前屏蔽您的隐私信息）
- 下单时段
- 账户类别（环球、综合、模拟）
- 市场及标的
- 订单类型（市价、限价等）


### 下单失败排查方法

若程序抛出异常，请对照下文常见错误消息检查问题，若不能解决问题，请联系我们。若程序无报错，请根据以下方法检查被拒单的原因：

订单状态是异步更新的，因此，如果下单失败，请使用get_order再次查询订单，订单内会标记失败原因。可以根据以下示例，在create_order, place_order, get_order 三个环节之后分别查看Order对象，比较其差异。

**1. create order 创建订单环节**

没有全局id，只有账户相关的order_id，status 为 NEW
```python
Order({'account': 'U523', 'id': None, 'order_id': 297, 'parent_id': None, 'order_time': None, 'reason': None, 'trade_time': None, 'action': 'BUY', 'quantity': 100, 'filled': 0, 'avg_fill_price': 0, 'commission': None, 'realized_pnl': None, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': None, 'outside_rth': None, 'contract': ES/None/USD, 'status': 'NEW', 'remaining': 100}) 
```

**2. place order 发送订单**

生成全局id，status 仍为 NEW。palce order 之后的 Order 对象中只增加了 id。如果需要获取最新的订单状态，需要使用 get_order 进行查询。
```python
Order({'account': 'U523', 'id': 154758864761483264, 'order_id': 297, 'parent_id': None, 'order_time': None, 'reason': None, 'trade_time': None, 'action': 'BUY', 'quantity': 100, 'filled': 0, 'avg_fill_price': 0, 'commission': None, 'realized_pnl': None, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': None, 'outside_rth': None, 'contract': ES/None/USD, 'status': 'NEW', 'remaining': 100})
```

**3. get order 获取订单状态**

status 变更为 REJECTED，且reason中增加了订单被拒绝的原因
```python
Order({'account': 'U523', 'id': 154758864761483264, 'order_id': 297, 'parent_id': 0, 'order_time': 1550115294556, 'reason': '201:Order rejected - Reason: YOUR ORDER IS NOT ACCEPTED. MINIMUM OF 2000 USD (OR EQUIVALENT IN OTHER CURRENCIES) REQUIRED IN ORDER TO PURCHASE ON MARGIN, SELL SHORT, TRADE CURRENCY OR FUTURE', 'trade_time': 1550115294694, 'action': 'BUY', 'quantity': 100, 'filled': 0, 'avg_fill_price': 0, 'commission': 0, 'realized_pnl': 0, 'trail_stop_price': None, 'limit_price': 0.1, 'aux_price': None, 'trailing_percent': None, 'percent_offset': None, 'order_type': 'LMT', 'time_in_force': 'DAY', 'outside_rth': True, 'contract': ES, 'status': 'REJECTED', 'remaining': 100})
```

以下为常见错误类型：

**1.错误提示：standard account response error(bad_request:Orders cannot be place at this moment)**

**错误原因**

当前时段不可下单

**解决方法**

遇到此错误时，请首先检查您下单的时间，并与对应市场交易时间对照。注意：市场交易时间可能受到交易所所在地区公共节假日的影响，请结合交易所官网公告进行判断。

- 若您使用的是**模拟账户**：

    模拟账户目前不支持部分市场及品类的盘前盘后及预挂单交易。请暂时修改程序尝试在正常交易时段进行下单，同时可以向我们[反馈问题](/zh/python/feedback/#官方量化qq群)

- 若您使用的是**综合账户**：

    综合账户不支持部分订单类型的盘前、盘后交易及预挂单交易。关于支持的订单类型，请见支持的[订单类型列表](/zh/python/FAQ/trade.html#支持交易的标的列表)

---

**2.错误提示：standard account response error(BAD_REQUEST:You cannot place market or stop order during pre-market and after-hours trading)**

**错误原因**

盘前盘后交易时不可下市价单和止损单

**解决方法**

检查下单时间，请注意对应的时区，并根据交易时间更改订单类型。关于支持的订单类型及交易时段，请见支持的[订单类型列表](/zh/java/FAQ/trade.html#支持交易的订单类型)

---

**3.错误提示：The order quantity you entered exceeds your currently available position**

**错误原因**
  
下卖单数量超过您当前可用仓位
  
**解决方法**

若需要从做多调整为做空，请先调用接口检查仓位并下单平仓，在平仓后再下卖单进行做空

---

**4.错误提示：standard account response error(bad_request:We don’t support trading of this stock now(Error Code 4))**

**错误原因**

传入了不可交易的股票代码

**解决方法**
    
可以使用手机APP确认标的的交易状态，或通过行情接口检查股票交易状态后再进行下单操作

### 支持交易的订单类型

#### 美国
**正股, ETFs**
订单类型 |时效| 盘前盘后 |预挂单
---- | ---- | ---- | ---- 
限价单|	DAY/GTC|✓|✓
市价单|DAY/GTC|×|×
止损限价单|	DAY/GTC|×|✓
止损单|DAY/GTC|×|✓
条件订单|DAY/GTC|×|×
附加订单|DAY/GTC|×|×

**期权**	
订单类型|时效|预挂单
---- | ---- | ----  
限价单|DAY/GTC|✓
市价单|DAY|×
止损限价单|DAY/GTC|✓

**期货**
订单类型 |时效|预挂单
---- | ---- | ---- 
限价单|	DAY/GTC|✓
市价单|DAY/GTC|×
止损限价单|	DAY/GTC|✓
止损单|DAY/GTC|✓
条件订单|DAY/GTC|×
附加订单|DAY/GTC|×

#### 香港
*收盘后有安全时间，安全时间内不允许预挂单*

**正股，ETFs**
订单类型|时效|预挂单
---- | ---- | ---- 
限价单|DAY/GTC|	✓
市价单|	DAY/GTC|	×
止损限价单|	DAY/GTC|	✓
止损单|	DAY/GTC|	✓
条件订单|	DAY/GTC|	×
附加订单|	DAY/GTC|	×
盘前竞价单|	DAY	| ×

**期权**
订单类型|时效|预挂单
---- | ---- | ---- 		
限价单|DAY/GTC|	✓
市价单|	DAY|×
止损限价单|	DAY/GTC|✓
止损单|	DAY/GTC	|✓

**涡轮、牛熊证**
订单类型|时效|预挂单
---- | ---- | ---- 	
限价单|DAY/GTC|✓

#### 新加坡	
*收盘后有安全时间，安全时间内不允许预挂单*

**正股，ETFs**
订单类型|时效|预挂单
---- | ---- | ---- 	
限价单|DAY/GTC|	✓
市价单|DAY/GTC|	×
止损限价单|	DAY/GTC|✓
止损单|	DAY/GTC|×
条件订单|DAY/GTC|×
附加订单|DAY/GTC|×
盘前竞价单|DAY|×