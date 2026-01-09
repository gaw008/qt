---
title: 其他
---
### Q1: 如何使用模拟盘？

登记开发者信息后，会自动生成一个模拟盘账户， 可以进入 Tiger trade APP 『交易』tab 下， 点击顶部区域切换查看实盘与模拟盘。

模拟盘的下单方法与实盘相同， 在place order 时 Account 参数传入模拟盘 Account， 而不是实盘 Account。


### Q2: sandbox、 模拟、 实盘都是什么，有什么关系？

模拟与实盘交易都需要使用真实环境进行。

sandbox 是一套独立的开发环境，用于某些特殊情景下的开发调试。 需要使用独立的私钥与Tigerid。

推荐用户优先使用模拟账户对程序进行调试。

### Q3: 有其他问题如何问题反馈？

使用上的疑问， 请添加官方QQ群咨询， 群号码：441334668。
收到错误返回后，为了尽快定位问题，给您反馈，请向提供以下信息(请优先使用chrome或者firefox等浏览器)。

* 请求信息，包括接口名称和请求参数
* 返回的错误信息

### 常见报错
**Q4: 报错如下**
```json
{"code":40013,"data":"","message":"invalid signature","timestamp":1527732508206}
```
错误原因: 可能是误修改了sdk内置的openapi公钥(tiger_public_key)，或者是私钥配置不正确，请仔细检查配置

**Q5: 报错如下**
```
File "....../tigeropen/tiger_open_client.py", line ..., in execute
self.__config.charset)
File "....../tigeropen/common/util/web_utils.py", line ..., in do_post
raise RequestException('[' + THREAD_LOCAL.uuid + ']post connect failed. ' + str(e))
tigeropen.common.exceptions.RequestException: post connect failed. 
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1056)

或
response sign varify faild. Varification Failed
```
错误原因: 可能是误修改了sdk内置的openapi公钥, 或者生成的密钥有问题, 可以尝试用密钥生成网站生成密钥对，版本选pkcs#1

**Q6: 报错如下**
```
rsa/key.py:492 _load_pkcs1_der
ValueError: Unable to read this file, version 136 != 0
```
python sdk 仅支持 pkcs#1版本的私钥, 不支持 pkcs#8, 请确认生成密钥是为pkcs#1版本

**Q7: 报错如下**
```
account is not authrized to the api user
```
错误原因: 一般是账户填写有误，请检查TigerOpenClientConfig.account是否正确

**Q8: 报错如下**
```json
{"code":1000,"message":"common param error(public key error)", "timestamp":1574386825800}
```
错误原因: 一般是 tiger_id 有误或者公钥未上传, 或者 is_sandbox设置成了True


**Q9: 报错如下**
```text
permission denied(Current user and device do not have permissions in the US market)
```
错误原因: 一般是没有API行情权限(请注意, API行情权限不同于app行情权限), 请到app购买. 如果已经购买, 
请调用 QuoteClient.grab_quote_permission() 抢占行情权限.


### 时间戳的转换

API 中返回的默认为毫秒时间戳， 参考以下方法转成本地时间：

```python
from datetime import datetime
from pytz import timezone

time_stamp = 1546635600
tz = timezone('Asia/Chongqing')

datetime.fromtimestamp(time_stamp, tz)
datetime.datetime(2019, 1, 5, 5, 0, tzinfo=<DstTzInfo 'Asia/Chongqing' CST+8:00:00 STD>)

```

---

### MarketStatus 等对象的使用方法

以 get_market_status 举例。 这个 API 返回了一个 MarketStatus 的列表。

```python
a_list = [MarketStatus({'market': 'US', 'status': '盘前交易', 'open_time': datetime.datetime(2019, 1, 7, 9, 30, tzinfo=<DstTzInfo 'US/Eastern' EST-1 day, 19:00:00 STD>)})]
us_market_status = a_list[0]
us_market_status.market
'US'
```