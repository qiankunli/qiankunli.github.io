---

layout: post
title: 学习存储
category: 技术
tags: Storage
keywords: TIDB

---

## 前言


许式伟：存储系统需要花费绝大部分精力在各种异常情况的处理上，甚至你应该认为，这些庞杂的、多样的错误分支处理，才是存储系统的 “正常业务逻辑”。

1. 单机时代的文件系统，机器断电、程序故障、系统重启等常规的异常，文件系统必须可以正确地应对，甚至对于磁盘扇区损坏，文件系统也需要考虑尽量将损失降到最低。
2. 到了互联网时代，有了 C/S 或 B/S 结构，存储系统又有了新指标：可用性。
3. 什么样的数据会有最大的存储规模？非结构化数据。这类数据的组织形式通常以用户体验友好为目标，而不是机器友好为目标。所以数据本身也自然不是以机器易于理解的结构化形式来组织。图片、音视频、Office 文档等多媒体文件，就是比较典型的非结构化数据。
4. 非结构化数据的存储方式，最理想的绝对不是分布式文件系统。文件系统只是桌面操作系统为了方便用户手工管理数据而设计的产物。服务端操作系统发展的初期，人们简单沿用了桌面操作系统的整套体系框架。非结构化数据最佳的存储方式，还是键值存储（KV Storage）。用于存储非结构化数据的键值存储，有一个特殊的名字，叫对象存储（Object Storage）。对象存储的出现，是服务端体系架构和桌面操作系统分道扬镳的开始。
5. 既然对象存储是一个键值存储，就意味着我们可以通过对 Key 做 Hash，或者对 Key 按 Key Range 做分区，都能够让请求快速定位到特定某一台存储机器上，从而转化为单机问题。这也是为什么在数据库之后，会冒出来那么多 NoSQL 数据库。NoSQL 数据库的名字其实并不恰当，它们更多的不是去 SQL，而是去关系（我们知道数据库更完整的称呼是关系型数据库）。有关系意味着有多个索引，也就是有多个 Key，而**这对数据库转为分布式存储系统来说非常不利**。

    ```go
    // 对象存储 AWS S3 访问接口
    func PutObject(bucket, key string, object io.Reader) (err error)
    func GetObject(bucket, key string) (object io.ReadCloser, err error)
    ```

当我一开始学习mysql 的实现，我跟着mysql 脉络去学习一个db 如何实现，学习tidb 时也是。然后再回头看， 发现两者很多问题是类似的，知识在这个时候开始分层了。再去看mysql 的博客，你会发现内容是混杂的，一方面是 实现一个db的通用思想、机制， 一方面是msyql的实现细节。当学习了多个数据库实现之后，通用思想、机制提炼出来， mysql/tidb 专属细节整理一下，上帝的归上帝、凯撒的归凯撒。

![](/public/upload/storage/learn_storage.png)

## 分类

[从单机到分布式数据库存储系统的演进](https://mp.weixin.qq.com/s/dMk34u_9R2w1piU-yHHQ9Q) 各种存储会提供什么形式的接口来供外部访问数据？
1. 块存储，底层语义，基于（一个或多个固定大小的） block 编程；在 Linux 的 IO 软件栈中，要直接使用块存储的话就要基于 LBA 编程，因此接口较为简单朴素，再加上块存储本身处于整个存储软件栈的底层，这导致块存储使用起来并不十分友好，但具有超低的时延和超高的吞吐。
    1. 想要读写访问数据，就必须使用与存储相匹配的协议（SCSI、SATA、SAS、FCP、FCoE、iSCSI……）。
3. 文件系统，“文件”这个概念的出现是因为“块”对人类用户来说实在是过于难以使用、难以管理了，绝大多数传统的文件存储都是基于块存储之上去实现的，有专门组织块结构来构成文件的块的表（比如FAT），在表中再加入其他控制信息，就能很方便地扩展出更多的高级功能，比如除了文件占用的块地址信息外，在表中再加上文件的逻辑位置就形成了目录，加上文件的访问标志就形成了权限，我们还可以再加上文件的名称、创建时间、所有者、修改者等一系列的元数据信息，来构成其他应用形式。
    1. 文件存储的访问不像块存储那样有五花八门的协议，其POSIX接口（Portable Operating System Interface，POSIX）已经成为了事实标准,诸如 Open、Write、Read 等许多操作数据的接口都能在文件系统中被找到。**人们把定义文件分配表应该如何实现、储存哪些信息、提供什么功能的标准称为文件系统（File System）**，FAT32、NTFS、exFAT、ext2/3/4、XFS、BTRFS 等都是很常用的文件系统。
2. 对象存储。非结构化数据，使用对象存储系统来处理视频、图片、音频等非结构化的数据。
    1. 这里的“对象”可以理解为一个元数据及与其配对的一个逻辑数据块的组合，元数据提供了对象所包含的上下文信息，比如数据的类型、大小、权限、创建人、创建时间，等等，数据块则存储了对象的具体内容。**你也可以简单地理解为数据和元数据这两样东西共同构成了一个对象**。
    1. 常见的操作都是通过PUT实现上传和GET 实现下载等；它的表现形式，你可以认为后端存储空间无限大，你只需要使用PUT、GET方式实现上传下载即可，无需关心后端存储 ；可扩展性强，使用简单，但上传的文件，无法在对象存储中对其进行修改编辑，如果有需要，下载到本地，然后再上传，无法为服务器提供块级别的存储；产品举例，百度网盘，HDFS、FastDFS、swift、公有云的：ASW S3，腾讯云的COS，阿里云的OSS等；


## 关于一致性

分布式的存储系统他们实现数据复制的方法是完全一样的，该原理能查到的最早的出处是 1978 年 Lamport 的一篇论文《The Implementation of Reliable Distributed Multiprocess Systems》。不管在mysql，sqlserver关系型数据库里面，还是在redis，mongo等非关系型存储引擎里，他们都在用数据复制的状态机原理。

### 存储大势的发展：ACID’s Consistency vs. CAP’s Consistency

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

## 数据平台

[爱奇艺基础数据平台演进](https://mp.weixin.qq.com/s/zKZ8jJ1pzNlZ2s3jpeFvJA)

[字节跳动数据库的过去、现状与未来](https://mp.weixin.qq.com/s/4Bvo0EBo_xtKdVcqhGynCQ)
[字节跳动用九年时间打造出了怎样的数据平台](https://mp.weixin.qq.com/s/kOPmedd4N2OxRP-JEk1c2g)

[业务数据治理体系化思考与实践](https://mp.weixin.qq.com/s/AlxIMPnKcWSyhOSe8-OfTQ)早期以问题驱动治理、凭经验治理为主。主要包括以下几个方面。
1. 治理认知差异大，思路不统一。治理不彻底、治理经验缺乏沉淀
2. 治理方法不标准
3. 治理效率低、效果差
4. 数据管治缺乏体系化，缺乏整体顶层治理方案设计，问题越来越复杂，单点难解决，不同问题的优先级无法确定。治理不符合MECE原则：每个治理方向由哪些问题组成，哪些最重要，哪些的ROI最高，哪些问题和治理动作可以合并，同一问题在数仓不同主题、不同分层的衡量标准和治理方法应该有哪些差异，都需要在体系化治理中进行考虑。

## 其它

[图数据库存储技术及实践](https://mp.weixin.qq.com/s/NCvw1P5AXWIsDdi2tc5J7Q)

[图数据库的发展脉络与技术演进](https://mp.weixin.qq.com/s/8MHQT6J-ah3KElx5ZtmHwA) 未读。