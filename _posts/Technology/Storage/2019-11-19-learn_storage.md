---

layout: post
title: 学习存储
category: 技术
tags: Storage
keywords: TIDB

---

## 前言

![](/public/upload/storage/storage.png)

当我一开始学习mysql 的实现，我跟着mysql 脉络去学习一个db 如何实现，学习tidb 时也是。然后再回头看， 发现两者很多问题是类似的，知识在这个时候开始分层了。再去看mysql 的博客，你会发现内容是混杂的，一方面是 实现一个db的通用思想、机制， 一方面是msyql的实现细节。当学习了多个数据库实现之后，通用思想、机制提炼出来， mysql/tidb 专属细节整理一下，上帝的归上帝、凯撒的归凯撒。

![](/public/upload/storage/learn_storage.png)

## 存储大势的发展：ACID’s Consistency vs. CAP’s Consistency

一开始数据库都是单机的，实现ACID 的特性相对简单，然后数据量开始变大，在分布式场景下可用性盖过了一致性（所谓的CAP，大部分最终选择了牺牲了部分一致性），此时一致性由上游根据业务需要来取舍。 但是ACID 的需求只是被转移却从未消失过，Avoiding lost updates, dirty reads, stale reads and enforcing app-specific integrity constraints are critical concerns for app developers，Solving these concerns directly **at the database layer** using the consistency provided by ACID transactions is a much simpler approach.

[A Primer on ACID Transactions: The Basics Every Cloud App Developer Must Know](https://blog.yugabyte.com/a-primer-on-acid-transactions)

Consistency in CAP is a **more fundamental** concept — it refers to the guarantee that all members of a distributed system have a shared understanding of the value of a single data element from a read standpoint. 

On the other hand, ACID’s consistency refers to data integrity guarantees that ensure the transition of the entire database from one valid state to another. Such a transition involves strict enforcement of integrity constraints such as data type adherence, null checks, relationships and more. Given that a single ACID transaction can touch multiple data elements where as CAP’s consistency refers to a single data element, ACID transactions are a stronger guarantee than CAP’s consistency.

###  What’s Needed For Implementing ACID?

1. Provisional Updates (Atomicity). Transactions involve multiple operations across multiple rows.  needs a mechanism to track the start, progress and end of every transaction along with the ability to make provisional updates across multiple nodes in some temporary space. Conflict detection, rollbacks, commits and space cleanups are also needed. Using Two-phase commit (2PC) protocol or one of its variations is the most common way to achieve atomicity. [Achieving Atomicity](https://blog.yugabyte.com/6-signs-you-might-be-misunderstanding-acid-transactions-in-distributed-databases/)
2. Strongly Consistent Core (Consistency)  单机时不成问题，分布式场景下，因为副本问题，The transaction manager will rely on the correctness of a single operation on a single row to enforce the broader ACID-level consistency of multiple operations over multiple rows.  [Achieving Consistency](https://blog.yugabyte.com/6-signs-you-might-be-misunderstanding-acid-transactions-in-distributed-databases/)
3. Transaction Ordering (Isolation), For a database to support the strictest serializable isolation level, a mechanism such as globally ordered timestamps is needed to sequentially arrange all the transactions. 必须为事务界定一个顺序 [Achieving Isolation](https://blog.yugabyte.com/6-signs-you-might-be-misunderstanding-acid-transactions-in-distributed-databases/)
4. Persistent Storage (Durability) [Achieving Durability](https://blog.yugabyte.com/6-signs-you-might-be-misunderstanding-acid-transactions-in-distributed-databases/)

上述的各种机制 在mysql、postgresql、tidb 中都有体现，实现一个机制有多种策略，有些策略只能单机用，有些策略可以推广到分布式上。分布式可以有coordinator ，也可以消灭coordinator， 通过不断地 探察本质，逐步逼近实现一个分布式ACID 的原子能力是什么？ 


