---
title: 配置 Open API
---
#### Q: openapi 域名是多少？ 长连接PushClient的端口是什么?  
域名: openapi.itiger.com
长连接端口: 8883. 
一般无需关注, 如需要排查网络问题时可用到

#### Q: python sdk 中用户信息应该填哪些?  
1. private_key 私钥路径 

2. tiger_id 开发者id 

3. account 账户, 不论实盘还是模拟，使用哪个账户就将哪个账户填在此处

有两种方式生成用户信息配置:  

第一种:  
用工具函数生成
```python
       from tigeropen.tiger_open_config import get_client_config
       client_config = get_client_config(private_key_path='your private key path',
              tiger_id='you tiger id', account='your account')
```
第二种:  
直接写配置类
```python
       from tigeropen.tiger_open_config import TigerOpenClientConfig
       client_config = TigerOpenClientConfig()
       client_config.private_key = read_private_key('your private key path')
       client_config.tiger_id = 'your tiger id'
       client_config.account = 'your account'
```