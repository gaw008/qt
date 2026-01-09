---
title: 通用
---
### grab\_quote\_permission 行情权限抢占

`QuoteClient.grab_quote_permission(market=Market.ALL, lang=None)`

**说明**

抢占行情权限，在多个设备共享一个账号时，只有主设备上会返回行情信息。所以每次切换设备时需执行行情权限抢占，将当前设备设置为主设备。若不切换设备则无需调用

**参数**

无

**返回**

`dict`

dict数据格式如下：

| KEY | VALUE |
|:--|:--|
| name | 行情权限名称 |
| expire_at | 权限过期时间(-1为长期有效) |

**示例**
```python  
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

permissions = quote_client.grab_quote_permission()
```

**返回示例**
```python
[{'name': 'usStockQuote', 'expire_at': 1698767999000}, {'name': 'usStockQuoteLv2Arca', 'expire_at': 1698767999000}, {'name': 'usStockQuoteLv2Totalview', 'expire_at': 1698767999000}, {'name': 'hkStockQuoteLv2', 'expire_at': 1698767999000}, {'name': 'usOptionQuote', 'expire_at': 1698767999000}]
```

### get\_quote\_permission 查询行情权限

**说明**
查询当前所拥有的行情权限

**参数**
无

**返回** 
同[grab_quote_permission](/zh/python/operation/quotation/common.html#grab-quote-permission-行情权限抢占)

**示例**
```python  
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='私钥路径', tiger_id='your tiger id', account='your account')
quote_client = QuoteClient(client_config)

permissions = quote_client.get_quote_permission()
```