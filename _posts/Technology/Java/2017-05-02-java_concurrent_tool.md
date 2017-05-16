---

layout: post
title: java concurrent 工具类
category: 技术
tags: Java
keywords: JAVA concurrent

---

## 前言（未完成）

## ConcurrentHashMap

[探索 ConcurrentHashMap 高并发性的实现机制](https://www.ibm.com/developerworks/cn/java/java-lo-concurrenthashmap/)

文中提到

1. 用分离锁实现多个线程间的并发写操作
2. 用 HashEntery 对象的不变性来降低读操作对加锁的需求
3. 用 Volatile 变量协调读写线程间的内存可见性

在java中，我们通过cas ==> aqs实现高性能的锁，进而通过减小锁的粒度、读写分离、或final等减少锁的使用。

所以并发变成下的高性能，不只着眼于锁。同时，锁默认顺带解决了内存可见性问题，不用锁时，就要直接处理内存可见性问题。

## ImmutableMap

