---
title: 对象列表
---
## 对象列表

本部分包含部分重要的通用对象的补充解释，包含的对象列表如下：

客户端
对象|基本功能
----|----
[TigerHttpClient](/zh/java/appendix1/object.html)|Http客户端，负责发送交易类和行情类请求
[WebSocketClient](/zh/java/appendix1/object.html#websocketclient-websocket客户端)|Websocket客户端，负责处理订阅请求


## 常用对象
**常量**  
引用路径 `com.tigerbrokers.stock.openapi.client.constant` [source](https://github.com/tigerfintech/openapi-java-sdk/tree/master/src/main/java/com/tigerbrokers/stock/openapi/client/constant)

名称 | 描述 
--- | --- 
ApiServiceType | API请求方法
OrderChangeKey | 订单推送字段
PositionChangeKey | 持仓推送字段
AssetChangeKey | 资产推送字段

**HTTP请求相关对象**  
引用路径 `com.tigerbrokers.stock.openapi.client.https` [source](https://github.com/tigerfintech/openapi-java-sdk/tree/master/src/main/java/com/tigerbrokers/stock/openapi/client/https)

名称 | 描述
--- | ---
client.TigerHttpClient | http请求客户端
domain | 返回的业务数据对象
request | 封装的请求对象
response | 封装的返回对象

**长连接相关**  
引用路径 `com.tigerbrokers.stock.openapi.client.socket` [source](https://github.com/tigerfintech/openapi-java-sdk/tree/master/src/main/java/com/tigerbrokers/stock/openapi/client/socket)

名称 | 描述
--- | ---
ApiComposeCallback | 回调接口类
WebSocketClient | 长连接客户端

**枚举，数据结构**  
引用路径 `com.tigerbrokers.stock.openapi.client.struct` [source](https://github.com/tigerfintech/openapi-java-sdk/tree/master/src/main/java/com/tigerbrokers/stock/openapi/client/struct)

**工具类**  
引用路径 `com.tigerbrokers.stock.openapi.client.util` [source](https://github.com/tigerfintech/openapi-java-sdk/tree/master/src/main/java/com/tigerbrokers/stock/openapi/client/util)