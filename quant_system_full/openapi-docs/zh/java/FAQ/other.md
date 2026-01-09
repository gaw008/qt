---
title: 其他
---
#### Q1: Client error？
```json
{"code":3,"message":"client error(java.security.spec.InvalidKeySpecException: java.security.InvalidKeyException: IOException : algid parse error, not a sequence)","success":false,"timestamp":0}
```
出现这个错误可能原因是私钥格式不是pkcs8格式，按照文档说明里操作时，要使用控制台打印出的私钥

#### Q2: 签名错误？

返回结果为：
```json
{"code":1000,"data":"","message":"common param error(sign check error)","timestamp":1527732508206}
```
出现这个错误时，可能是把openapi的公钥配置成了用户自己的公钥，或者是私钥配置不正确，请仔细检查下配置。

时间戳的转换

#### Q3: API 中返回的时间戳？

默认为毫秒时间戳， 参考以下方法转成本地时间：

```java
>>>from datetime import datetime
>>>from pytz import timezone

>>>time_stamp = 1546635600000
>>>tz = timezone('Asia/Chongqing')

>>>datetime.fromtimestamp(time_stamp, tz)

datetime.datetime(2019, 1, 5, 5, 0, tzinfo=<DstTzInfo 'Asia/Chongqing' CST+8:00:00 STD>)
```
#### Q4: API 中美股股票或期权查询？
所涉及到的时间戳（long类型）都应为美国东部时间的时间戳

---