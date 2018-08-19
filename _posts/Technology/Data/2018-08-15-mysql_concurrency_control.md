---

layout: post
title: mysql 并发控制
category: 技术
tags: Data
keywords: mysql concurrency control

---

## 前言（未完成）

[InnoDB并发如此高，原因竟然在这？](https://juejin.im/entry/5b70db49e51d45663f46bc52) 有几条思路性的东西，值得总结

通过并发控制保证数据一致性的常见手段有：

* 锁（Locking）
* 数据多版本（Multi Versioning）

	
## 锁

提高并发的演进思路：

1. 普通锁，本质是串行 执行
2. 读写锁，可以实现读读并发
3. 数据多版本，可以实现读写并发。核心原理是：

	1. 写任务发生时，将数据克隆一份，以版本号区分；
	2. 写任务操作新克隆的数据，直至提交；
	3. 并发读任务可以继续读取旧版本的数据，不至于阻塞；

	有点aufs 写文件的意思
4. 插入意向锁，提高插入并发。但对于AUTO_INCREMENT 类型的列，则AUTO-INC lock 用以使插入串行


[挖坑，InnoDB的七种锁](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961451&idx=1&sn=1bac366be5ad2dc721f79c9cb8e65e34&chksm=bd2d0db78a5a84a101e05a02e337fe91c3fd179132bced897156e1f34f0d0ba7e48dc89a1b95&mpshare=1&scene=23&srcid=0819tg70Rq5dtSfmkhNSo3Yw%23rd)

文中提到 "事务会阻塞"，而不是我们常说的 "线程会阻塞"，这种 表述是不是意味着，执行事务的线程 如果发现事务阻塞了，就可以转而执行其它事务， 就像goroutine 那样？

[插入InnoDB自增列，居然是表锁？
](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961455&idx=1&sn=4c26a836cff889ff749a1756df010e0e&chksm=bd2d0db38a5a84a53db91e97c7be6295185abffa5d7d1e88fd6b8e1abb3716ee9748b88858e2&mpshare=1&scene=23&srcid=0819Cm3t80QS2jGTBwZx9hJO%23rd)

An AUTO-INC lock is a special table-level lock taken by transactions inserting into tables with AUTO_INCREMENT columns. In the simplest case, if one transaction is inserting values into the table, any other transactions must wait to do their own inserts into that table, so that rows inserted by the first transaction receive consecutive primary key values.

对于包含 AUTO_INCREMENT 类型列的表，如果一个事务正在往表中插入记录，所有其他事务的插入必须等待。或者说，此时的插入操作 需要获取表锁。**目的是：不管如何并发，AUTO_INCREMENT列的值必须是连续的。**

[InnoDB并发插入，居然使用意向锁？](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961461&idx=1&sn=b73293c71d8718256e162be6240797ef&chksm=bd2d0da98a5a84bfe23f0327694dbda2f96677aa91fcfc1c8a5b96c8a6701bccf2995725899a&mpshare=1&scene=23&srcid=0819IqEIwB53R4cmFfYIa1YY%23rd)

多个事务，在同一个索引，同一个范围区间插入记录时，如果插入的位置不冲突，不会阻塞彼此。Insert Intention Lock signals the intent to insert in such a way that multiple transactions inserting into the same index gap need not wait for each other if they are not inserting at the same position within the gap.

[InnoDB 的意向锁有什么作用？ - 文龙的回答 - 知乎](
https://www.zhihu.com/question/51513268/answer/147733422)

1. IS and IX locks allow access by multiple clients. They won't  necessarily conflict until they try to get real locks on the same rows.But a table lock (ALTER TABLE, DROP TABLE, LOCK TABLES) blocks both IS and IX, and vice-versa.
2. **提高加表锁效率**，意向锁是在添加行锁之前添加，操作row 时添加行锁 是应该的，为何要 多此一举 添加意向锁 ？为的是 提高 “向一个表添加表级X锁(ALTER TABLE, DROP TABLE, LOCK TABLES)” 锁时的效率。有了意向锁，只需要判断该意向锁与即将添加的表级锁是否兼容即可，无需遍历整个表判断是否有行锁的存在。


这就可以解释，为何涉及到 加字段 等操作，dba 总是要求 凌晨执行相关代码。因为在 `alter table xx` 期间，基本无法对数据库进行读写了。


这里有一个问题，普通锁、读写锁、数据多版本 有一个提高并发度 的线可以将它们串起来，那么这7种锁 如何将它们串起来。

感觉上是 调控 写与写操作的行为。

## 其它

普通锁

mysql 是按事务执行的，不是按sql 语句执行的




## 其它材料

[深入理解MySQL――锁、事务与并发控制 这才是正确的！](https://zhuanlan.zhihu.com/p/36060546)


[MySQL 加锁处理分析](http://hedengcheng.com/?p=771#_Toc374698316) （未读）

官方文章 [InnoDB Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html)