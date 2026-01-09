## 订阅接口列表

**订阅接口使用注意事项**

订阅接口通过`WebSocketClient`对象实现，首先初始化`WebSocketClient`，再使用client中的方法调用相应的服务。关于WebSocketClient类的具体描述见对象列表。订阅推送相关的请求均为异步处理，故需要用户自定义回调函数，与中间函数进行绑定。某个事件发生，或有最新信息更新被服务器推送时，程序会自动调用用户自定义的回调函数并传入返回接口返回的数据，由用户自定义的回调函数来处理数据

订阅接口通用调用方法：

订阅接口通用调用方法步骤如下：
1. 实现回调接口
   
   实现`ApiComposeCallback`类的对应回调方法，并在初始化时传入`WebSocketClient`对象。由于订阅类为异步API，必须实现回调方法用于响应接口返回的数据。具体的回调接口及实现示例请参考各个接口的详细说明页面

2. 构造认证类

    构造认证类`ApiAuthentication`，并在初始化`WebSocketClient`时传入，用于身份验证
    示例如下：
    ```java
    ClientConfig clientConfig = TigerOpenClientConfig.getDefaultClientConfig();
    ApiAuthentication authentication = ApiAuthentication.build(clientConfig.tigerId, clientConfig.privateKey);
    ```
3. 生成client
   
   初始化`WebSocketClient`, 方法详见下页示例

4. 连接接口

    调用WebSocketClient.connect()，可实现`ApiComposeCallback`类中的回调方法来响应连接的结果

5. 调用服务

    通过下方列表中的`WebSocketClient`的方法来调用相应的对应服务，如`WebSocketClient.subscribe()`
   
6. 关闭服务
    通过`WebSocketClient.disconnect()`如果没有后续操作可以停止服务,之后将不会再收到异步消息推送。主动发起disconnect()会同时取消已订阅的行情

**支持的订阅接口的完整列表如下：**

### 账户变动
接口|功能描述|回调接口
----|----|-----
[subscribe](/zh/java/operation/subscribe/accountChange.html)|订阅账户变动|subscribeEnd, orderStatusChange, positionChange, assetChange
[cancelSubscribe](/zh/java/operation/subscribe/accountChange.html#void-cancelsubscribe-subject-subject-取消订阅)|取消订阅账户变动|cancelSubscribeEnd

### 行情订阅
接口|功能描述|回调接口
----|----|----
[subscribeQuote](/zh/java/operation/subscribe/quotation.html#subscribequote-set-string-symbols-quotekeytype-quotekeytype-行情订阅)|订阅行情数据|quoteChange
[cancelSubscribeQuote](/zh/java/operation/subscribe/quotation.html#subscribequote-set-string-symbols-quotekeytype-quotekeytype-行情订阅)|取消订阅行情|cancelSubscribeEnd
[subscribeOption](/zh/java/operation/subscribe/quotation.html#subscribeoption-set-string-symbols-订阅期权行情)|订阅期权行情| optionChange
[cancelSubscribeOption](/zh/java/operation/subscribe/quotation.html#subscribeoption-set-string-symbols-订阅期权行情)|取消订阅期权行情|cancelSubscribeEnd
[subscribeDepthQuote](/zh/java/operation/subscribe/quotation.html#subscribedepthquote-set-string-symbols-订阅深度行情)|订阅深度行情|depthQuoteChange
[cancelSubscribeDepthQuote](/zh/java/operation/subscribe/quotation.html#subscribedepthquote-set-string-symbols-订阅深度行情)|取消订阅深度行情|cancelSubscribeEnd
[cancelSubscribe](/zh/java/operation/subscribe/quotation.html#getsubscribedsymbols-查询已订阅标的)|取消订阅|cancelSubscribeEnd

### 其他事件响应方法
回调接口|功能描述
----|----
[void connectAck()](/zh/java/operation/subscribe/other.html)|连接成功
[void connectionClosed()](/zh/java/operation/subscribe/other.html)|连接已关闭
[void error(String errorMsg)](/zh/java/operation/subscribe/other.html)|异常回调
[void error(int id, int errorCode, String errorMsg)](/zh/java/operation/subscribe/other.html#异常回调)|异常回调