---
title: 对象
---
### TigerHttpClient HTTP客户端
`com.tigerbrokers.stock.openapi.client.https.client.TigerHttpClient`

**初始化方法：**
1. 构造含有个人账户及密钥的用户配置对象ClientConfig
```java
public class TigerOpenClientConfig {
  static {
    ClientConfig clientConfig = ClientConfig.DEFAULT_CONFIG;
    clientConfig.tigerId = "your tiger id";
    clientConfig.defaultAccount = "your account"; 
    clientConfig.privateKey = "you private key string";

  }
  public static ClientConfig getDefaultClientConfig() {
    return ClientConfig.DEFAULT_CONFIG;
  }
}
```

2. 使用`com.tigerbrokers.stock.openapi.client.config.ClientConfig`初始化HttpClient
```java
private static TigerHttpClient client = new TigerHttpClient(TigerOpenClientConfig.getDefaultClientConfig());
```

### WebSocketClient Websocket客户端
`com.tigerbrokers.stock.openapi.client.socket.WebSocketClient`

**初始化方法：**
1. 构造认证类

使用`com.tigerbrokers.stock.openapi.client.config.ClientConfig`构造认证类`com.tigerbrokers.stock.openapi.client.socket.ApiAuthentication`，并在初始化WebSocketClient时传入，用于身份验证 

示例如下： 
```java
ClientConfig clientConfig = TigerOpenClientConfig.getDefaultClientConfig(); 
ApiAuthentication authentication = ApiAuthentication.build(clientConfig.tigerId, clientConfig.privateKey);
```

2. 构造Websocket客户端
```java
private static ClientConfig clientConfig = TigerOpenClientConfig.getDefaultClientConfig();
private static WebSocketClient client =
    WebSocketClient.getInstance().url(clientConfig.socketServerUrl)
      .authentication(ApiAuthentication.build(clientConfig.tigerId, clientConfig.privateKey))
      .apiComposeCallback(new DefaultApiComposeCallback());
```