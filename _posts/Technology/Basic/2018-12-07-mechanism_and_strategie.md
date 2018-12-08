---

layout: post
title: 从一个签名框架看待机制和策略
category: 技术
tags: Basic
keywords: abtest

---

## 简介（未完成）

在web 开发中，经常要对返回结果做签名，以防止客户端收到篡改后的结果。

很多key value

## 最初形态 工具类

抽象key value，提供一个SignatureUtils.signature(Map<String,String> kvs,String secret) 工具类


## 后续形态 框架类


	class Data implements Signatureable{
	
		@SignatureField
		private String key1
		
		void setSignatrue()
	} 
	
	
## 最终形态


key value 和 secret 如何拼接 是机制

value 为null 是不是计入签名 是策略。value 是 如何转化为 字符串也是策略

策略最好使用配置 、注解、元信息 等表示，机制的核心 还是 `SignatureUtils.signature(Map<String,String> kvs,String secret)`

换句话说，除了`SignatureUtils.signature(Map<String,String> kvs,String secret)` 或者说 如何将外部场景 转换为`SignatureUtils.signature(Map<String,String> kvs,String secret)` 的部分都是策略


机制应该尽可能的稳定

策略 应该尽可能的简单、专一

比如linux 网络协议栈的 几个表，NAT 是 linux 内部数据包流转的一个流程。