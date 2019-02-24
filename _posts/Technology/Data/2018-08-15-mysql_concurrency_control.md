---

layout: post
title: mysql 并发控制
category: 技术
tags: Data
keywords: mysql concurrency control

---

## 前言

数据库锁定机制简单来说，就是数据库为了保证数据的一致性，而使各种共享资源在被并发访问变得有序所设计的一种规则。

[InnoDB并发如此高，原因竟然在这？](https://juejin.im/entry/5b70db49e51d45663f46bc52)  系列 内容 大部分来自 官方文章 [InnoDB Locking](https://dev.mysql.com/doc/refman/8.0/en/innodb-locking.html)，同是加入了作者的一些提炼。

2019.02.24补充，[JVM3——java内存模型](http://qiankunli.github.io/2017/05/02/java_memory_model.html)将mysql 与 jvm 内存模型的相关规则做了对比。

## 隔离性与 锁

[4种事务的隔离级别，InnoDB如何巧妙实现？](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961498&idx=1&sn=058097f882ff9d32f5cdf7922644d083&chksm=bd2d0d468a5a845026b7d2c211330a6bc7e9ebdaa92f8060265f60ca0b166f8957cbf3b0182c&mpshare=1&scene=23&srcid=0829xvsK46xDhwpojd9EbY18%23rd)

||第一个事务|第二个事务|说明|对应的锁|
|---|---|---|---|---|
|脏度|插入 未提交|select * ...|第二个事务读到了我提交的数据 ||
|不可重复度|读1...读2|更改|同样的条件 ,   你读取过的数据 ,   再次读取出来发现值不一样了 |间隙锁|
|幻读|读1...读2|插入、删除|  第 1 次和第 2 次读出来的记录数不一样（所以有间隙锁这一套）|临键锁|

隔离性的进一步认识：多个用户的并发事务 访问同一个数据库时，一个用户的事务不应该为其它用户的事务干扰。这个干扰意涵就比较tricky了

1. 未提交事务对事务的影响。事务有一个特性：事务未提交，那么对数据库的更改 就不算“尘埃落定”，这个时候的数据 就不应该被别的事务感知到。否则，可以视为“干扰”了
2. 已提交事务对未提交事务的影响。比如事务A 要插入一条记录（记录包含 一个name字段，且设置为unique），事务A 先查询了一下发现没有name=lisi 记录，然后插入name=lisi 的记录。但在select 和 insert 之间 事务B 插入了name=lisi 的记录并提交，便会导致事务A 操作失败。不可重复读 和幻读 均可导致 该结果，此时也是一种“干扰”。

但第二种情况 似乎情有可原，因为失败了就失败了，事务A 的用户再发起一次操作就可以了。所以说，隔离性是一致性和并发性的权衡。

![](/public/upload/data/mysql_sql_lock.png)
	
## 锁

[挖坑，InnoDB的七种锁](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961451&idx=1&sn=1bac366be5ad2dc721f79c9cb8e65e34&chksm=bd2d0db78a5a84a101e05a02e337fe91c3fd179132bced897156e1f34f0d0ba7e48dc89a1b95&mpshare=1&scene=23&srcid=0819tg70Rq5dtSfmkhNSo3Yw%23rd)

学东西， 比较重要的不是怎么样，而是为什么要这样。 提高并发的演进思路：

1. 普通锁，本质是串行 执行
2. 读写锁，可以实现读读并发
3. 数据多版本，可以实现读写并发。核心原理是：

	1. 写任务发生时，将数据克隆一份，以版本号区分；
	2. 写任务操作新克隆的数据，直至提交；
	3. 并发读任务可以继续读取旧版本的数据，不至于阻塞；

	有点aufs 写文件的意思
4. 写操作分为插入、更改和删除
5. 对已有数据行的修改与删除，必须加强互斥锁X锁，那对于数据的插入，是否还需要加这么强的锁，来实施互斥呢？插入意向锁（间隙锁的一种，所以也是实施在索引上的），可以提高插入并发。
6. 但对于AUTO_INCREMENT 类型的列，则AUTO-INC lock 用以使插入串行。
7. 读读并行、写写串行都比较确定， 关键就是读写 如何协调，那么针对读写可能产生的问题 用对应的锁来解决（所以，数据库提高或降低 隔离级别，也就是数据库启用/禁用了这些锁）。

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

[InnoDB，select为啥会阻塞insert？](https://mp.weixin.qq.com/s?__biz=MjM5ODYxMDA5OQ==&mid=2651961471&idx=1&sn=da257b4f77ac464d5119b915b409ba9c&chksm=bd2d0da38a5a84b5fc1417667fe123f2fbd2d7610b89ace8e97e3b9f28b794ad147c1290ceea&mpshare=1&scene=23&srcid=0822Acfwzhugvrc1wp9x4o51%23rd)


InnoDB的细粒度(行)锁，是实现在索引记录上的，A record lock is a lock on an index record。如果查询没有命中索引，也将退化为表锁。

1. 记录锁，它封锁索引记录，例如：

		select * from t where id=1 for update;

	它会在id=1的索引记录上加锁，以阻止其他事务插入，更新，删除id=1的这一行。

2. 间隙锁，它封锁索引记录中的间隔

		select * from t where id between 8 and 15 for update;

	会封锁区间，以阻止其他事务id=10的记录插入。
	
	间隙锁的主要目的，就是为了防止其他事务在间隔中插入数据，以导致“不可重复读”。如果把事务的隔离级别降级为读提交(Read Committed, RC)，Gap locking can be disabled explicitly.
	
3. 临键锁，是记录锁与间隙锁的组合，它的封锁范围，既包含索引记录，又包含索引区间。更具体的，临键锁会封锁索引记录本身，以及索引记录之前的区间。

	如果一个会话占有了索引记录R的共享/排他锁，其他会话不能立刻在R之前的区间插入新的索引记录。
	
	t(id PK, name KEY, sex, flag);
	
	表中有四条记录：

		1, shenjian, m, A
		3, zhangsan, m, A
		5, lisi, m, A
		9, wangwu, f, B

	PK上潜在的临键锁为：

		(-infinity, 1]
		(1, 3]
		(3, 5]
		(5, 9]
		(9, +infinity]

	临键锁的主要目的，也是为了避免幻读(Phantom Read)。如果把事务的隔离级别降级为RC，临键锁则也会失效。
	

	
## 线程阻塞 还是事务 阻塞

文中提到 "事务会阻塞"，而不是我们常说的 "线程会阻塞"，这种 表述是不是意味着，执行事务的线程 如果发现事务阻塞了，就可以转而执行其它事务， 就像goroutine 那样？ 从 [MySQL锁阻塞分析，mysql锁阻塞](http://www.bkjia.com/sjkqy/874857.html) 可以看到，就实现上来说， 事务阻塞也就意味着 执行事务的线程阻塞。进而可以推断，并发读写比较多时，会导致大量的数据库线程在同一时间处于阻塞状态，进而拖慢 数据库执行 任务队列中事务的速度。

	$ show engine innodb status

	------------
	TRANSACTIONS
	------------
	Trx id counter 4131
	Purge done for trx's n:o < 4119 undo n:o < 0 state: running but idle
	History list length 126
	LIST OF TRANSACTIONS FOR EACH SESSION:
	---TRANSACTION 0, not started
	MySQL thread id 2, OS thread handle 0x7f953ffff700, query id 115 localhost root init
	show engine innodb status
	---TRANSACTION 4130, ACTIVE 41 sec starting index read
	mysql tables in use 1, locked 1
	LOCK WAIT 2 lock struct(s), heap size 360, 1 row lock(s)
	MySQL thread id 4, OS thread handle 0x7f953ff9d700, query id 112 localhost root updating
	delete from emp where empno=7788
	------- TRX HAS BEEN WAITING 41 SEC FOR THIS LOCK TO BE GRANTED:   ## 等待了41s
	RECORD LOCKS space id 16 page no 3 n bits 88 index `PRIMARY` of table `test`.`emp` trx id 4130 lock_mode X locks rec but not gap waiting
	Record lock, heap no 9 PHYSICAL RECORD: n_fields 10; compact format; info bits 0  ## 线程4在等待往test.emp中的主键上加X锁，page num=3
	 0: len 4; hex 80001e6c; asc    l;;
	 1: len 6; hex 000000001018; asc       ;;
	 2: len 7; hex 91000001420084; asc     B  ;;
	 3: len 5; hex 53434f5454; asc SCOTT;;
	 4: len 7; hex 414e414c595354; asc ANALYST;;
	 5: len 4; hex 80001d8e; asc     ;;
	 6: len 4; hex 208794f0; asc     ;;
	 7: len 4; hex 80000bb8; asc     ;;
	 8: SQL NULL;
	 9: len 4; hex 80000014; asc     ;;
	
	------------------
	---TRANSACTION 4129, ACTIVE 45 sec starting index read
	mysql tables in use 1, locked 1
	LOCK WAIT 2 lock struct(s), heap size 360, 1 row lock(s)
	MySQL thread id 7, OS thread handle 0x7f953ff6c700, query id 111 localhost root updating
	update emp set sal=3500 where empno=7788
	------- TRX HAS BEEN WAITING 45 SEC FOR THIS LOCK TO BE GRANTED:   ## 等待了45s
	RECORD LOCKS space id 16 page no 3 n bits 88 index `PRIMARY` of table `test`.`emp` trx id 4129 lock_mode X locks rec but not gap waiting
	Record lock, heap no 9 PHYSICAL RECORD: n_fields 10; compact format; info bits 0  ## 线程7在等待往test.emp中的主键上加X锁，page num=3
	 0: len 4; hex 80001e6c; asc    l;;
	 1: len 6; hex 000000001018; asc       ;;
	 2: len 7; hex 91000001420084; asc     B  ;;
	 3: len 5; hex 53434f5454; asc SCOTT;;
	 4: len 7; hex 414e414c595354; asc ANALYST;;
	 5: len 4; hex 80001d8e; asc     ;;
	 6: len 4; hex 208794f0; asc     ;;
	 7: len 4; hex 80000bb8; asc     ;;
	 8: SQL NULL;
	 9: len 4; hex 80000014; asc     ;;
	
	------------------
	---TRANSACTION 4128, ACTIVE 51 sec
	2 lock struct(s), heap size 360, 1 row lock(s)
	MySQL thread id 3, OS thread handle 0x7f953ffce700, query id 110 localhost root clean

从`show engine innodb status` 输出可以看到， 一个事务id 通常 对应一个 thread id。


## 其它材料

[深入理解MySQL――锁、事务与并发控制 这才是正确的！](https://zhuanlan.zhihu.com/p/36060546)

[MySQL 加锁处理分析](http://hedengcheng.com/?p=771#_Toc374698316) 

喜欢请关注个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)
