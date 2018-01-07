---

layout: post
title: compensable-transaction 源码分析
category: 技术
tags: Java
keywords: elasticsearch

---

## 前言(未完成)

compensable-transaction是一个类似于spring-tx的事务实现框架，并且比spring-tx 多了持久化事务数据的能力。

为什么要持久化事务数据？防止系统突然关闭待来的不一致性。

1. spring-tx 主要用于jdbc，jdbc 协议本身就符合对事务参与者的一些要求。因此，即便系统突然关闭，因为还没有执行commit请求，数据仍然是一致的。
2. 系统重启时，加载未完成的事务数据，执行相关的业务代码。

很多事务参与者，本身不符合xa协议规定的能力（比如单纯的rpc服务），其rollback 逻辑需要调用方在业务代码中自己指定，此时，compensable-transaction 因其可以持久化事务数据，并按配置对未完成事务采取重试机制，支持更多的应用场景。

经常分析源码有以下几个好处：

1. 很多东西是共通的，套路是一样的，可以加快理解源码的速度
2. 涉猎广泛，涉猎广泛就可以理论联系实际。比如我们说事务有四大特性，从一个程序员的角度说，概念背的再溜不如show me your code。比如事务的原子性，在mysql中体现为redo和undo log，在spring-tx 体现为try catch中的rollback。在compensable-transaction 中除了commit、rollback之外，还有系统重启之后的重试，**此时transaction就是redo/undo log。**

## 项目结构

compensable-transaction 的基本逻辑是，当事务失败或未完成时，对业务代码进行重试。

1. compensable-transaction-core 基本的事务抽闲，据此便可以完成编程式事务代码

	1. org.mengyun.compensable.transaction 包括基本抽象类
	2. org.mengyun.compensable.transaction.recovery 定义重试逻辑
	3. org.mengyun.compensable.transaction.repository 定义事务数据的存储介质
	4. org.mengyun.compensable.transaction.serializer 定义了事务数据的序列化方式
	5. org.mengyun.compensable.transaction.support 提供TransactionConfigurator 统一操作TransactionManager、TransactionRepository和RecoverConfig 三大抽象。
2. compensable-transaction-spring 提供TransactionConfigurator的具体实现RecoverConfiguration，根据用户对RecoverConfiguration 配置，spring容器启动时初始化TransactionManager和TransactionRepository，并调度执行RecoverConfig。
3. compensable-transaction-server 事务数据持久化后，可以通过后台界面查看。

compensable-transaction 暂未提供注解方式，以提供“声明式事务代码”的支持。

