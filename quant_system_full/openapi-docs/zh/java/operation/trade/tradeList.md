## 交易类接口列表

**交易类请求也通过`TigerHttpClient`对象发送, `TigerHttpClient`类为Http客户端，负责发送行情类与交易类请求，发送请求前需先初始化`TigerHttpClient`。具体初始化方法及使用方法请参考请求对象/接口的使用示例**

### 查询账户信息
接口 | 功能
---- | ----
[TigerHttpRequest(ApiServiceType.ACCOUNTS)](/zh/java/operation/trade/accountInfo.html#账户列表-tigerhttprequest-apiservicetype-accounts)| 账户列表
[TigerHttpRequest(ApiServiceType.POSITIONS)](/zh/java/operation/trade/accountInfo.html#账户持仓-tigerhttprequest-apiservicetype-positions)| 账户持仓
[TigerHttpRequest(ApiServiceType.ASSETS)](/zh/java/operation/trade/accountInfo.html#环球账户资产-tigerhttprequest-apiservicetype-assets)|环球账户资产
[PrimeAssetRequest](/zh/java/operation/trade/accountInfo.html#综合-模拟账号获取资产-primeassetrequest)|综合/模拟账号获取资产

### 获取合约
接口 | 功能
---- | ----
[ContractRequest/ContractsRequest](/zh/java/operation/trade/getContract.html)| 获取合约

### 获取订单信息

接口 | 功能
---- | ----
[TigerHttpRequest(ApiServiceType.PREVIEW_ORDER)](/zh/java/operation/trade/orderInfo.html#预览订单-仅环球账户-tigerhttprequest-apiservicetype-preview-order)| 预览订单
[TigerHttpRequest(ApiServiceType.ORDERS)](/zh/java/operation/trade/orderInfo.html#获取订单-tigerhttprequest-apiservicetype-orders)|获取订单,订单列表
[TigerHttpRequest(ApiServiceType.ORDER_TRANSACTIONS)](/zh/java/operation/trade/orderInfo.html#获取成交记录-tigerhttprequest-apiservicetype-order-transactions)|订单成交记录列表
[TigerHttpRequest(ApiServiceType.ACTIVE_ORDERS)](/zh/java/operation/trade/orderInfo.html#获取待成交订单列表-tigerhttprequest-apiservicetype-active-orders) | 待成交订单列表 
[TigerHttpRequest(ApiServiceType.FILLED_ORDERS)](/zh/java/operation/trade/orderInfo.html#获取已成交订单列表-tigerhttprequest-apiservicetype-filled-orders) | 已成交订单列表
[TigerHttpRequest(ApiServiceType.INACTIVE_ORDERS)](/zh/java/operation/trade/orderInfo.html#获取已撤销订单列表-tigerhttprequest-apiservicetype-inactive-orders)| 已撤销订单列表

### 交易下单
接口 | 功能
---- | ----
[TradeOrderRequest](/zh/java/operation/trade/placeOrder.html#创建订单-tradeorderrequest)|交易下单

### 取消或修改订单

接口 | 功能
---- | ----
[TigerHttpRequest(ApiServiceType.MODIFY_ORDER)](/zh/java/operation/trade/cancelAndModify.html#修改订单-tigerhttprequest-apiservicetype-modify-order)|修改订单
[TigerHttpRequest(ApiServiceType.CANCEL_ORDER)](/zh/java/operation/trade/cancelAndModify.html#取消订单-tigerhttprequest-apiservicetype-cancel-order)|取消订单