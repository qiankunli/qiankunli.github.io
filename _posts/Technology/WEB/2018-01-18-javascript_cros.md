---

layout: post
title: web 跨域问题
category: 技术
tags: WEB
keywords: javascript

---

## 简介

[Cross-Origin Resource Sharing (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

[Server-Side Access Control (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Server-Side_Access_Control)


## 什么是跨域？

Cookie 是一个Http State Management Mechanism，保存在客户端，由浏览器维护，表示应用状态。客户端得到Cookie 后， 后续请求都会自动将Cookie 携带至请求中。

同源策略：协议、主机、端口必须完全相同。如果两个 URL 的 protocol、host  和port 都相同的话，则这两个 URL 是同源。在页面中通过 `about:blank` 或 `javascript:` URL 执行的脚本会**继承**打开该 URL 的文档的源，因为这些类型的 URLs 没有包含源服务器的相关信息。

浏览器为什么要有同源策略？ 为了保证用户信息的安全，防止恶意的网站窃取数据。设想这样一种情况：A 网站是一家银行，用户登录以后，A 网站在用户的机器上设置了一个 Cookie，包含了一些隐私信息（比如存款总额）。用户离开 A 网站以后，又去访问 B 网站，如果没有同源限制，B 网站可以读取 A 网站的 Cookie，那么隐私信息就会泄漏。更可怕的是，Cookie 往往用来保存用户的登录状态，如果用户没有退出登录，其他网站就可以冒充用户，为所欲为。随着互联网的发展，同源政策越来越严格。目前，如果非同源，共有三种行为受到限制。

1. 无法获取非同源网页的 cookie、localstorage 和 indexedDB。
2. 无法访问非同源网页的 DOM （iframe）。
3. 无法向非同源地址发送 AJAX 请求 或 fetch 请求（可以发送，但浏览器拒绝接受响应）

也就是说，如果协议、域名或者端口有一个不同，都被当作是不同的域，请求跨域了，虽然请求发出去了，但浏览器会拦截响应。PS： 用户先后在地址栏输入的两个域的地址，比如a.com和b.com，浏览器肯定不能将a.com 的cookie 带到发给b.com 的请求上。但若是一个页面由多个部分组成：图片、javascript等，可能由同一家公司的不同域名提供服务，此时也被同源策略误伤到了。

跨域检查由浏览器自动完成，包括向请求添加Origin header、检查响应header等，javascript脚本无需关心。

![](/public/upload/apache/cros_1.png)


## 两种请求类型

1. simple request
2.	preflight request，“需预检的请求”要求必须首先使用 OPTIONS   方法发起一个预检请求到服务器，以获知服务器是否允许该实际请求。 

对于需要预检的请求，每次请求发一次options 和 实际请求（比如post）（url 都是一样的）。为了减少 耗时，可以通过配置 `Access-Control-Max-Age` 缓存 options 结果，减少options 的请求次数。 [使用 Access-Control-Max-Age 来缓存 CORS 配置](https://www.web-tinker.com/article/20961.html)

## 后端工作

实现CORS通信，客户端是浏览器自动实现（前端开发的js脚本无需改变），nginx + 服务器代表的后端主要做以下几件事：

1. 请求中的Origin header 记录了请求本身所在的域名， 服务端可据此判断是否响应该跨域请求
2. 处理跨域请求后，在响应中添加跨域相关header，以通过浏览器对相关header的检查
3. 服务端回应 preflight request 中的OPTIONS 请求

### nginx 实现

[Nginx通过CORS实现跨域](http://www.yunweipai.com/archives/9381.html)

在nginx server 配置下新增

	server {
		...
		add_header 'Access-Control-Allow-Origin' $http_origin;
		add_header 'Access-Control-Allow-Credentials' 'true';
		add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
		add_header 'Access-Control-Allow-Headers' 'Authorization,Content-Type,xx';
		...
	}

若该域名下已经配置了很多映射，应检查响应中新增这些header 对 其它请求的影响。 一个办法是尽量减少add_header 的作用范围

	server {
		location /xx{
			add_header ...
			add_header ...
		}
	}
	
若该域名下已存在跨域配置，则可以新增一个map配置

	map $uri $originHeader {
        default '*';
        ~正则表达式 $http_origin;
    }
    server {
		add_header 'Access-Control-Allow-Origin' $originHeader;
		add_header ...
	}

### 代码实现

[Spring MVC通过CROS协议解决跨域问题](http://www.imooc.com/article/7719)

## Access-Control-Allow-Origin 设置为*还是$http_origin？

[跨域资源共享 CORS 详解](http://www.ruanyifeng.com/blog/2016/04/cors.html)

Access-Control-Allow-Credentials是一个布尔值，表示是否允许发送Cookie。**默认情况下，Cookie不包括在CORS请求之中**。设为true，即表示服务器明确许可，Cookie可以包含在请求中，一起发给服务器。**如果要发送Cookie，Access-Control-Allow-Origin就不能设为星号，必须指定明确的、与请求网页一致的域名（这样浏览器才知道发送哪些cookie）**。（从实践上看，这段话正确性存疑， 但应晓得有这回事）

## 小结

浏览器 + 后端 有一套策略（同源策略），实现起来基于一堆header。我们可以通过 设置header 来影响浏览器的行为。