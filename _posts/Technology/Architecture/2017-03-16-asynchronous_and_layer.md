---

layout: post
title: 异步、分层那些事儿
category: 技术
tags: Architecture
keywords: 异步

---

## 简介

笔者是一个java开发工程师，处于业务组，写了很多的crud代码。因此，对一些项目、架构保持很大的敬畏，一些开源项目的源码即便看的懂，也觉得很牛逼。十几个类，彼此之间，简单来看，是聚合、继承等关系，但又有一种说不清道不明的联系。不知怎么地，就可以弄出异步、监听等神奇的效果。

笔者这么多年的读书学习的经验表明，如果你认为一个东西很厉害，理解起来有一些tricky的点，那就说明还没有深入理解它。

笔者最近在看zookeeper client的源码，也曾看过我司的服务治理框架mainstay（可以实现rpc、服务发现、服务降级、异步调用等功能），它们涉及的东西很多：io、多线程、分层和异步等等。


## 异步

1. 异步的两种表现形式：future和callback，具体参见[Future](http://qiankunli.github.io/2016/07/08/future.html)和[回调](http://qiankunli.github.io/2016/07/08/callback.html)
2. 异步的实现有两种层面：系统级（中断及中断处理，本质就是硬件+os支持的callback）和业务级，业务级别有

	1. 比如zookeeper client，queue + 线程实现callback
	2. 比如一些rpc框架，底层netty callback + 全局map 实现future

3. 异步可以用在不同的地方：

	||计算|IO:BIO/NIO|
	|---|---|---|
	|单线程|当一个函数前后代码不变，就中间的逻辑经常变时，可以考虑提个callback出去||
	|多线程|一个queue和一个执行线程，执行线程执行任务，结果写入future返回。业务线程操作future|一个`<全局唯一id,Future>` + io线程。io线程发送请求，拿到结果后，根据全局唯一id找到future并写入结果|
	
	

	
## 分层

zookeeper、dubbo client端的基本套路

|层|功能|交互数据|
|---|---|---|
|使用(可选)|结合spring实现自动注入、自定义配置标签等||
|api层|提供crud监及异步操作接口|业务自定义model|
|业务层|根据协议数据，以框架业务model的形式存储有状态信息，实现协议要求的机制，比如服务降级（dubbo）、监听（zookeeper）等功能|框架业务model|
|transport层|传输数据model，屏蔽netty、nio和bio使用差异|框架基本数据model，协议的对象化，负责序列化和反序列化|
|socket||byte[]|

业务层的异步（线程加队列）和  通信层的异步（底层os机制），简单看，上层是否异步，依赖于底层，**但分层是根据职责的，将异步封装为同步的代码，放在上层或下层均可**。也就说，理论上，为了让代码符合直觉，每一层都可以做到同步。最上层是否提供异步操作，只看用户的需求。

## 分层在代码上的表现（未完成）

	




