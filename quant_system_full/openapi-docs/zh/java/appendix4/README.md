---
title: 附录4：底层实现
---
### 整体结构

**注意：只需要使用SDK调用API的用户可以跳过本节内容**

API接口按照功能划分主要分为交易接口和行情接口，按照调用形式划分可以分为REST API接口和长连接推送接口。其中REST API接口包括了交易接口和部分行情接口，订单状态、行情实时报价、深度行情报价等数据对实时性要求较高，是通过长连接推送接口实现，长连接推送使用的协议是WebSocket和Stomp，REST API接口和推送接口在调用时都需要进行身份认证和接口签名

### 请求地址

环境 | 请求地址|Content-Type| 请求方法
--- | --- | --- | ---
正式环境 | https://openapi.itiger.com/gateway | application/json | POST
>开发者信息注册成功后会自动生成模拟账号，模拟账号交易也是在通过正式环境的链接访问，但是底层交易环境是相互隔离的

### 公共请求参数

参数 | 类型 | 是否必填 | 最大长度 | 描述 | 示例值
--- | --- | --- | --- | --- | ---
tiger_id    | string | 是 | 10   | 分配给开发者的ID | 20150129
method      | string | 是 | 128  | 接口名称，详见:[method说明](#method) | place_order
charset     | string | 是 | 10   | 请求使用的编码格式，目前支持：UTF-8 | UTF-8
sign_type   | string | 是 | 10   | 商户生成签名字符串所使用的签名算法类型，目前支持RSA | RSA
timestamp   | string | 是 | 19   | 请求时间，格式"yyyy-mm-dd HH:mm:ss"  | 2018-05-09 11:30:00
version     | string | 是 | 3    | 接口版本，目前版本包括：1.0，2.0。具体版本请查看接口文档 | 1.0
biz_content | string | 是 | -    | 请求参数的集合，除公共参数外所有业务参数都放在这个参数中传递，具体参照各接口接入文档 |
sign        | string | 是 | 344  | 请求参数的签名串，详见:[签名规则](./sign.md) | 详见示例
account_type| string | 否 | -    | 账户类型，token认证时必传，类型包括:[GLOBAL,STANDARD] 
access_token| string | 否 | -    | token认证时必传，用于验证用户身份，可以通过登录接口获取
trade_token | string | 否 | -    | token认证时选填，下单相关接口必传，可以通过交易密码接口获取


### 公共响应参数

参数 | 类型 | 是否必填 | 描述 | 示例值
--- | --- | --- | --- | ---
code            | string | 是 | 网关返回码,详见: code字段说明 | 40001
message         | string | 是 | 网关返回码描述,详见: code字段说明 | post param is empty
data            | string | 否 | 业务返回内容，详见具体的API接口文档 | -
timestamp       | string | 是 | 网关返回的时间戳 | 1525849042762


### 请求示例
```json
{
  "tiger_id":"1",
  "charset":"UTF-8",
  "sign_type":"RSA",
  "version":"1.0",
  "timestamp":"2018-09-01 10:00:00",
  "method":"order_no",
  "biz_content":"{\"account\":\"DU575569\"}",
  "sign":"QwM4MCdffJ5WK59f+dbFvKMn5Qqw2A5GTA8g0XIAp/Fsvb5fbZUwYzxjznx0jO7VO9Npbzd+ywR6VrMz4liblTMPGDvDnPJP0rGUVF+xbj/3MBr3vFZ25XheyjfHIpP6f+qhNkn9KdFsviohZAWeplkYjV+OyxwMQmpnkP/vll4="
}
```

### 响应示例
```json
{
	"code": 0,
	"message": "",
	"timestamp": 1525849042762,
	"data": {
			"orderId":10000164
	}
}
```

### 回调接口


回调接口目前支持两种方式，一种是通过SDK中提供的订阅和推送接口来实现，另外一种方式是通过http回调地址来实现。http回调地址可以在开发者注册信息页面进行添加和更新，支持域名配置，但不支持IP和端口形式的地址配置(在开发者信息页面支持填写ip和port形式地址，但是不会实际推送回调消息)。http回调接口可以用来获取订单、资产和持仓的最新变动，但不能获取行情最新的价格变动和深度买卖盘数据。SDK中的订阅接口可以收到全部类型的回调消息，如无特殊要求，推荐优先使用SDK提供的推送接口。

> http回调结果推送如果超时或者失败，会尝试重复推送三次，如果三次推送都超时或者失败，则不会继续推送该条消息，但不会影响后续消息的推送。
>
> 如果域名被检测出格式非法，或者无法正常接收和处理请求，该域名有可能被添加到黑名单里，后续不会再继续往该域名地址推送消息


回调接口消息格式为:

```json
{
	"subject":"Asset",
	"data": {
		"type":"asset",
		"usdCash":854879.28,
		"account":"DU575567",
		"timestamp":1528719139003
	}
}
```

* 回调的消息类型,即上面示例中的subject字段，主要包括以下几种：OrderStatus(订单状态变化消息)，Asset（资产变动消息），Position（持仓变动消息），Quote（行情价格变动消息），QuoteDepth（深度买卖盘价格变动消息）
* 每种消息类型对应的字段内容，可以参考订阅部分文档