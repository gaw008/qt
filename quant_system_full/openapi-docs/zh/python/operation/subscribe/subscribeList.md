## 订阅接口及响应方法列表
订阅类接口负责推送数据相关的功能，目前支持的功能，对应的推送接口，以及响应方法如下表：

**注意**
1. 以下全部为异步API，需要指定一个方法响应返回的结果，具体使用方法详见快速入门
2. 使用以下接口前需要先初始化[`PushClient`](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/push/push_client.py) 。使用示例见[快速入门](/zh/python/quickStart/basicFunction.html#订阅行情)  
 

### 账户变动
|  接口方法  | 功能 | 回调方法 |
|  ----  | ----  | ---- |
| [subscribe_asset](/zh/python/operation/subscribe/accountChange.html#%E8%B5%84%E4%BA%A7%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E4%B8%8E%E5%8F%96%E6%B6%88)  | 订阅资产变化 |PushClient.asset_changed|
| [unsubscribe_asset](/zh/python/operation/subscribe/accountChange.html#%E8%B5%84%E4%BA%A7%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E4%B8%8E%E5%8F%96%E6%B6%88) | 取消订阅资产变化 ||
| [subscribe_position](/zh/python/operation/subscribe/accountChange.html#%E6%8C%81%E4%BB%93%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E4%B8%8E%E5%8F%96%E6%B6%88)  | 订阅仓位的变化 |PushClient.position_changed|
| [unsubscribe_position](/zh/python/operation/subscribe/accountChange.html#%E6%8C%81%E4%BB%93%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E4%B8%8E%E5%8F%96%E6%B6%88) | 取消订阅仓位的变化 ||
| [subscribe_order](/zh/python/operation/subscribe/accountChange.html#%E8%AE%A2%E5%8D%95%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E5%92%8C%E5%8F%96%E6%B6%88) | 订阅订单变化的订阅 |PushClient.order_changed|
| [unsubscribe_order](/zh/python/operation/subscribe/accountChange.html#%E8%AE%A2%E5%8D%95%E5%8F%98%E5%8C%96%E7%9A%84%E8%AE%A2%E9%98%85%E5%92%8C%E5%8F%96%E6%B6%88)  | 取消订阅订单变化 ||

### 订阅行情
|  接口方法  | 功能 | 回调方法| 
|  ----  | ----  |----| 
| [subscribe_quote](/zh/python/operation/subscribe/quotation.html#%E8%AE%A2%E9%98%85%E8%A1%8C%E6%83%85) | 订阅行情 |PushClient.quote_changed| 
| [subscribe_depth_quote](/zh/python/operation/subscribe/quotation.html#订阅深度行情) | 订阅深度行情 |PushClient.quote_changed| 
| [subscribe_option](/zh/python/operation/subscribe/quotation.html#订阅期权行情) | 订阅期权行情 |PushClient.quote_changed |
| [query_subscribed_quote](/zh/python/operation/subscribe/quotation.html#查询已订阅的标的列表) | 查询已订阅的标的列表|PushClient.subscribed_symbols|
| [unsubscribe_quote](/zh/python/operation/subscribe/quotation.html#取消订阅) | 取消订阅行情 || 
| [unsubscribe_depth_quote](/zh/python/operation/subscribe/quotation.html#取消订阅) | 取消订阅深度行情 |
    
### 其他事件响应方法
|  事件  | 回调方法 |
|  ----  | ----  |
| 连接成功 | PushClient.connect_callback |
| 断开连接 | PushClient.disconnect_callback |
| 连接出错 | PushClient.error_callback |