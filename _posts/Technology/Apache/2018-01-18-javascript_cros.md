---

layout: post
title: web 跨域问题
category: 技术
tags: WEB
keywords: javascript

---

## 简介（未完成）

[Cross-Origin Resource Sharing (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

[Server-Side Access Control (CORS)](https://developer.mozilla.org/en-US/docs/Web/HTTP/Server-Side_Access_Control)


什么是跨域？

为什么跨域要特殊处理一下？

如何实现？

1. 请求 通过header 告诉 服务器，请求是从哪个域跨过来的


征得服务器允许

1.	preflight request
2. simple request



## 实现

不推荐使用nginx的方式，这样项目的部署便多了一层依赖。

### nginx 实现

[Nginx通过CORS实现跨域](http://www.yunweipai.com/archives/9381.html)

### 代码实现

[Spring MVC通过CROS协议解决跨域问题](http://www.imooc.com/article/7719)