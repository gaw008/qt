---
title: 准备工作
---
### 环境要求
- 操作系统要求：
  - Windows
  - MacOS
  - Linux
  - Solaris
    
- 编程语言版本要求：Java JDK 1.7及以上

### 安装Java Development Kit
为获得最好的支持，减少兼容性问题，推荐安装 JDK17

安装最新版本的JDK:

1. 访问Oracle官方网站下载页面： [https://www.oracle.com/java/technologies/downloads/](https://www.oracle.com/java/technologies/downloads/)
2. 注意需根据操作系统选择对应的安装包，下载安装包并按提示安装

### 获取Tiger Open API Java SDK
**方式1：使用Maven代码仓库（推荐）**

推荐您通过添加Maven依赖的方式来获取SDK。Java 开发者可以通过Maven仓库获取Java SDK，Maven仓库会不定期更新版本，目前最新版本如下：

请将以下代码添加到pom.xml文件中：

```xml
<dependency>
  <groupId>io.github.tigerbrokers</groupId>
  <artifactId>openapi-java-sdk</artifactId>
  <version>1.3.5</version>
</dependency>
```

**方式2：Github/Gitee代码仓库**

SDK项目的源代码同时在Github与Gitee代码仓库上发布

项目的github地址为:

[https://github.com/tigerbrokers/openapi-java-sdk](https://github.com/tigerbrokers/openapi-java-sdk)

项目的gitee地址:

[https://gitee.com/tigerbrokers/openapi-java-sdk](https://github.com/tigerbrokers/openapi-java-sdk)

您可选择从代码仓库中Clone/下载源代码并手动添加依赖

### 安装IDE
我们推荐使用IntelliJ IDEA作为集成开发环境

官方下载地址：
[https://www.jetbrains.com/idea/download/](https://www.jetbrains.com/idea/download/)

### 注册开发者信息
使用API之前请首先点击链接访问老虎量化官网开通权限并登记开发者身份：[https://www.itiger.com/openapi/info](https://www.itiger.com/openapi/info)，**推荐使用Chrome浏览器打开**

注意：开通Open API权限需要在老虎开户并满足**当前资产大于等于2000美元**的最低入金门槛，

注册信息时系统会要求开发人员和用户签署API授权协议，随后需要在此页面中完成开发者信息的登记，请点击更新并填入您的信息。登记开发者身份需要准备的信息有：

登记开发者身份需要准备的信息有：
|  信息   | 是否必填  | 说明 |
|  ----  | ----  | ----| 
| RSA公钥  | 是 | 使用RSA公钥，是为了确保用户的接口请求安全，通过RSA双向签名验证机制，防止接口请求被恶意篡改，关于RSA的生成，请参考快速入门-准备工作中“生成RSA密钥”部分|
| IP白名单  | 否 | 只有在白名单内的IP才可以访问API接口，多个IP间以 “;” 分隔|
| 回调URL  | 否 | 用户应用程序的回调地址，可以用于接收订单、持仓、资产的变更消息。用户也可以直接通过SDK提供的订阅接口接收回调消息|

注：注册信息时系统会要求开发人员和用户签署API授权协议

**注册成功后您可在此页面中获取以下信息：**

- **tigerId：** 开放平台为每一位开发者分配的唯一ID，用来标识一个开发者，所有API接口的调用都会用到tigerId。
- **account：** 用户的资金账号，在请求交易相关接口时需要用到资金账号。具体分为环球账号、综合账号与模拟账号，
  - **环球资金账号（Global）**：以大写字母U开头，如：U12300123，
  - **综合资金账号（Standard）**：为一串较短的数字（5到10位），如：51230321，
  - **模拟资金账号（Paper）**：17位数字，如：20191106192858300，

注册开发者信息成功后只会返回已成功入金的资金账号和模拟账号。如果用户的环球账号和综合账号都已成功入金，则都会返回。

### 生成RSA密钥
最后一步使用前的准备工作是生成RSA密钥。老虎Open API采用RSA进行身份验证。使用RSA加密，是为了确保用户的接口请求安全，通过RSA双向签名验证机制，防止接口请求被恶意篡改。用户在使用前自行生成RSA密钥，并将生成的RSA密钥对中的公钥提供给老虎开放平台，将私钥保存在用户本地，在调用接口时向服务器传入RSA私钥，用于内容签名，确保接口安全。

密钥可通过在线网站生成或本地生成, 接下来详细说明本地生成RSA密钥的方法：

RSA的生成依赖于OpenSSL，需要首先安装：
- MacOS操作系统下可以通过在终端（Terminal）中使用`brew`命令安装：`brew install openssl`
- Linux操作系统通过终端（Terminal）中使用`apt-get`命令安装：`sudo apt-get install openssl`
- Windows下的安装较为繁琐，请参考OpenSSL官方网站，或酌情使用在线生成工具。具体工具请搜索关键词 “RSA证书在线生成”，或者直接通过第三方下载exe版本的安装工具，直接运行([http://slproweb.com/products/Win32OpenSSL.html](http://slproweb.com/products/Win32OpenSSL.html))
  
完成OpenSSL的安装后，请使用下面的命令在终端（Terminal）或命令提示符中（cmd）生成公钥和私钥。其中私钥用户在本地存储，用于请求服务器时将内容签名。公钥需要在开放平台上传，用于验证用户的请求。使用以下命令生成的RSA密钥默认保存在用户当前目录

1. 首先生成RSA私钥（保存在当前目录），如果选择使用Python SDK，使用此命令生成的rsa_private_key.pem作为私钥即可

    ```$ openssl genrsa -out rsa_private_key.pem 1024```
    
2. Java 开发者需要将私钥转换成PKCS8格式, Python SDK 不需要执行这步操作
    
    ```$ openssl pkcs8 -topk8 -inform PEM -in rsa_private_key.pem -outform PEM -nocrypt```

3. 使用RSA私钥生成对应的RSA公钥

    ```$ openssl rsa -in rsa_private_key.pem -pubout -out rsa_public_key.pem```

生成的公钥与私钥示例：

>**出于资金安全的考虑，请勿直接使用本页面上或其他互联网页面上的RSA密钥**

RSA私钥
```
-----BEGIN PRIVATE KEY-----
MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAL96InBHMUph4vXC
CK1Y/lWwqlgOBcz4A2cpEVzFQd3l0wBsE4kBPlbXWeBJc+ixquS7OkV5b2pVmJJ3
WMhfX3M8UUJvgzsU812SPBO0Eu51AcY+lEXQPwsoZXvVQVj3Ql+eeCxqEunfu6rc
ZMKAgy9/q//wbrdPjt+u9A+nT7qtAgMBAAECgYBJaf9q3qZfJGgyc3xJB3XZJF7s
WH/DrlaS0xTz4k5En7knYp86YKZKuOvl7iBybuRqUBkPS6u9uJ2f1F7sJybC6MZ2
MPvoOKj4UYCjAQr0mwjYP69HJTRnNnwUqbtDXsuKJQ5tzS7q/3HYbO3M2xewqpgS
9Bbm8cwV6Mi2FwFCwQJBAPkULhWEfpMabT4gDYTXCiaUhUEZc0J5JfeL+xt20ZVS
phoQG4IVAo0TbqCrEDA91GxGk9KSCLS7hbRvYV7cRVECQQDEzDR+ZhIy/JTw5ubY
nraY796icld4X0g+b57QPmotx6jTEfsdhxku8BN1tK2j/VhbA1OEQ9hcZOTmAiIE
N7idAkEAg95sO5YnESiXl6GOprrWs/BE0GByBkpvkGy66CJy+XSFXh0TAz6uWBRm
qIeIjZHeieif0IbiNxVkx0+EpJ1H0QJAEFUeOitAcWjS95dCK1Iot1KY+IRizAOk
XEIpPQEhEMGUOkgwvgebSHD2PHuNOaHp9ku1X7G9wBVDhe9BYXY6ZQJBAOOviKaI
dMlEFFEdDDe8HZT89OFXTmIVNj3eyllSx91J25AsBCpUiLXfoN7z3t+fCpmOANPf
KqGNYjWwAjKiAFc=
-----END PRIVATE KEY-----
```
RSA公钥
```
-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEDAQUAA4GNADCBiQKBgQC/eiJwRzFKYeL1wgitWP5VsKpY
DgXM+ANnKRFcxUHd5dMCbBOJAT5W11ngSXPosarkuzpFeW9qVZiSd1jIX19zPFFC
b4M7FPNdkjwTtBLudQHGPpRF0D8LKFV71UFY90JfnngsahLp37uq3GTCgIMvf6v/
8G63T47frvQPp0+6rQIDAQAB
-----END PUBLIC KEY-----
```

注意： 

**1. 生成的密钥需为1024位**

**2. Java 与 Python SDK 使用的私钥格式不同，Java 使用的私钥为 PKCS#8 格式。若使用SDK遇到问题时，注意先检查私钥的格式是否正确**

生成RSA公私钥后，需将RSA公钥(默认文件名为rsa_public_key.pem)提交给开放平台，将PKCS#8格式的RSA私钥妥善保存，后续程序调用API时将会用到

### 购买行情（可选）

我们免费提供延迟行情接口，但实时行情与历史行情权限需要另外购买。Open API的行情权限独立与APP与PC端，如果您已经购买了APP或PC行情，也需要另外购买Open API的行情权限以获得实时数据。具体购买方法如下：

**个人客户**

在 Android版本的 **Tiger Trade APP - 我的 - 行情权限 - OpenAPI权限** 中进行购买 

**机构客户**

在 **机构中心-行情权限** 中进行购买