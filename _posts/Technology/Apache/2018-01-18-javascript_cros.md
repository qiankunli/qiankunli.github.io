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

[跨域资源共享 CORS 详解](http://www.ruanyifeng.com/blog/2016/04/cors.html)

[跨域资源共享（CORS）](https://www.alibabacloud.com/help/zh/doc-detail/31928.htm) 基本要点：

1. 同源策略
2. CORS 是一个由浏览器共同遵循的一套控制策略——同源策略，**通过HTTP的Header来进行交互**
3. 如果header 不符合策略要求，浏览器可以拒绝javascript脚本 的http请求和拦截http 响应
4. 跨域检查由浏览器自动完成，包括向请求添加Origin header、检查响应header等，javascript脚本无需关心


![](/public/upload/apache/cros_1.png)


## 两种请求类型

1.	preflight request
2. simple request

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