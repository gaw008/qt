## 交易接口列表
交易类接口包括查询账户，下单等功能，目前支持的功能以及对应的交易接口如下表：

**注意**

使用以下接口前需要先初始化[`TradeClient`](https://github.com/tigerfintech/openapi-python-sdk/blob/master/tigeropen/trade/trade_client.py)，初始化示例见[快速入门](/zh/python/quickStart/basicFunction.html#交易下单)

### 查询账户信息
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_managed_accounts](/zh/python/operation/trade/accountInfo.html#get-managed-accounts-%E8%8E%B7%E5%8F%96%E7%AE%A1%E7%90%86%E7%9A%84%E8%B4%A6%E5%8F%B7%E5%88%97%E8%A1%A8)  | 获取管理的账号列表 |
| [get_assets](/zh/python/operation/trade/accountInfo.html#get-assets-%E8%8E%B7%E5%8F%96%E8%B4%A6%E6%88%B7%E8%B5%84%E4%BA%A7%E4%BF%A1%E6%81%AF) |  获取账户资产信息 |
| [get_prime_assets](/zh/python/operation/trade/accountInfo.html#get-prime-assets-%E8%8E%B7%E5%8F%96%E7%BB%BC%E5%90%88-%E6%A8%A1%E6%8B%9F%E8%B4%A6%E6%88%B7%E8%B5%84%E4%BA%A7%E4%BF%A1%E6%81%AF) | 获取综合/模拟账户资产信息 |
| [get_positions](/zh/python/operation/trade/accountInfo.html#get-positions-%E8%8E%B7%E5%8F%96%E6%8C%81%E4%BB%93%E6%95%B0%E6%8D%AE)  | 获取持仓数据 |

### 获取合约
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_contracts](/zh/python/operation/trade/getContract.html#get-contracts-%E8%8E%B7%E5%8F%96%E5%A4%9A%E4%B8%AA%E5%90%88%E7%BA%A6%E4%BF%A1%E6%81%AF)| 获取多个合约对象 |
| [get_contract](/zh/python/operation/trade/getContract.html#get-contract-%E8%8E%B7%E5%8F%96%E5%8D%95%E4%B8%AA%E5%90%88%E7%BA%A6%E4%BF%A1%E6%81%AF)| 获取单个合约对象 |

### 获取订单信息
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [get_order](/zh/python/operation/trade/orderInfo.html#get-order-%E8%8E%B7%E5%8F%96%E6%8C%87%E5%AE%9A%E8%AE%A2%E5%8D%95) | 获取单个订单 |
| [get_orders](/zh/python/operation/trade/orderInfo.html#get-orders-%E8%8E%B7%E5%8F%96%E8%AE%A2%E5%8D%95%E5%88%97%E8%A1%A8) | 获取订单列表 |
| [get_open_orders](/zh/python/operation/trade/orderInfo.html#get-open-orders-%E8%8E%B7%E5%8F%96%E5%BE%85%E6%88%90%E4%BA%A4%E7%9A%84%E8%AE%A2%E5%8D%95%E5%88%97%E8%A1%A8) | 获取待成交的订单列表 |
| [get_cancelled_orders](/zh/python/operation/trade/orderInfo.html#get-cancelled-orders-%E8%8E%B7%E5%8F%96%E5%B7%B2%E6%92%A4%E9%94%80%E7%9A%84%E8%AE%A2%E5%8D%95%E5%88%97%E8%A1%A8) | 获取已取消的订单列表 |
| [get_filled_orders](/zh/python/operation/trade/orderInfo.html#get-filled-orders-获取已成交的订单列表) | 获取已成交的订单列表 |
|[get_transactions](/zh/python/operation/trade/orderInfo.html#get-transactions-获取订单成交记录)|获取订单成交记录(仅适用于综合账户)|

### 下单
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [place_order](/zh/python/operation/trade/placeOrder.html#place-oder-%E4%B8%8B%E5%8D%95)  | 提交订单 |

### 修改或取消订单
|  接口方法  | 功能 | 
|  ----  | ----  | 
| [cancel_order](/zh/python/operation/trade/cancelAndModify.html#cancel-order-%E5%8F%96%E6%B6%88%E8%AE%A2%E5%8D%95)| 取消订单 |
| [modify_order](/zh/python/operation/trade/cancelAndModify.html#modify-order-%E4%BF%AE%E6%94%B9%E8%AE%A2%E5%8D%95) | 修改订单 |