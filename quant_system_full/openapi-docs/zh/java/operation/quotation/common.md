---
title: 通用
---
### 抢占行情权限/获取权限列表 TigerHttpRequest(ApiServiceType.GRAB_QUOTE_PERMISSION)

**说明**

抢占行情权限，在多个设备共享一个账号时，只有主设备上会返回行情信息。所以每次切换设备时需执行行情权限抢占，将当前设备设置为主设备。若不切换设备则无需调用

**参数**

无

**返回**

字段名称|类型|说明
---|---|---
name|string|权限名称
expireAt|long|过期时间

**示例**

```java
TigerHttpRequest request = new TigerHttpRequest(ApiServiceType.GRAB_QUOTE_PERMISSION);
String bizContent = AccountParamBuilder.instance()
        .buildJson();
request.setBizContent(bizContent);
TigerHttpResponse response = client.execute(request);
```

**返回示例**
```json
{
  "code": 0,
  "message": "success",
  "timestamp": 1525938835697,
  "data": [{"name":"usQuoteBasic","expireAt":1621931026000}]
}
```