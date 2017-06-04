---

layout: post
title: google guava的一些理解
category: 技术
tags: Java
keywords: JTA TCC 

---

## 简介（待完善）

严重推进的文档——官方wiki：`https://github.com/google/guava/wiki`，部分中文翻译问题比较大。

## Future

A traditional Future represents the result of an asynchronous computation: a computation that may or may not have finished producing a result yet. 

一个同步操作往往会有一个结果（立即返回，或阻塞一段后返回），异步操作会立即返回一个结果（不是业务上的结果），确切的说是一个结果的Holder，即Future，异步操作的发起方和执行方约定在这个Future中存放和获取结果。

## 系统的理下异步执行框架(未完成)

一些框架代码是在业务线程驱动下执行，而提供异步操作的框架则

1. 提供操作对象，及异步接口
2. 框架本身需要`xx.start()`或`xx = new xx()`触发执行


## RateLimiter

`java.util.Semaphore` 限制了并发访问的数量而不是使用速率。RateLimiter 从概念上来讲，速率限制器会在可配置的速率下分配许可证。如果必要的话，每个`acquire()` 会阻塞当前线程直到许可证可用后获取该许可证。一旦获取到许可证，**不需要再释放许可证**。

[Guava官方文档-RateLimiter类](http://ifeve.com/guava-ratelimiter/)

## 新用Function

	interface Function<F, T>{
		 T apply(@Nullable F input);
	}
	
这个接口可以作为参数

如果我们维护一个`map<Class, Function>`

	function transfer(Object bean){
		Object value = map.get(bean.class).apply(bean);
		ReflectionUtils.setField(field, bean,
                        value);
	}

就可以在这样一些场景，游刃有余了。

## guava-retrying

[rholder/guava-retrying](https://github.com/rholder/guava-retrying)是一位大牛基于guava写一个的重试框架，封装了重试的一些通用问题

1. 什么时候重试，是否立即重试
2. 重试几次、重试间隔

