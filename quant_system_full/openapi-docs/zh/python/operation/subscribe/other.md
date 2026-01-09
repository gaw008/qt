---
title: 其他事件响应方法
---
### 连接成功
`
PushClient.connect_callback
`

**说明**

连接成功的回调

**参数**

无

**返回**

无

**示例**
```python
from tigeropen.push.push_client import PushClient
from tigeropen.tiger_open_config import get_client_config
client_config = get_client_config(private_key_path='private key path', tiger_id='your tiger id', account='your account')

# 初始化PushClient
protocol, host, port = client_config.socket_host_port
push_client = PushClient(host, port, use_ssl=(protocol == 'ssl'))


def connect_callback():
    """连接建立成功的回调"""
    print('connected')
    
push_client.connect_callback = connect_callback

```


### 断开连接
`
PushClient.disconnect_callback
`

**说明**

连接断开的回调

**参数**

无

**返回**

无

**示例**
```python
def disconnect_callback():
    """断线重连"""
    for t in range(1, 200):
        try:
            print('disconnected, reconnecting...')
            push_client.connect(client_config.tiger_id, client_config.private_key)
        except:
            print('connect failed, retry')
            time.sleep(t)
        else:
            print('reconnect success')
            break
            
# 初始化push_client步骤略，同上
push_client.disconnect_callback = disconnect_callback   
   
```

### 连接出错
```
PushClient.error_callback
```

**说明**

连接出错的回调

**参数**

错误信息接收参数

**返回**

无

**示例**
```python
def error_callback(content):
	"""错误回调
	:param content: 错误信息
	"""
	print(content)

# 初始化push_client步骤略，同上  
push_client.error_callback = error_callback
```
