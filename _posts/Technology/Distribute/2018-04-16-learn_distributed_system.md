---

layout: post
title: 总纲——如何学习分布式系统
category: 技术
tags: Distribute
keywords: 分布式系统

---

## 简介

* TOC
{:toc}

所谓分布式，指的是计算节点之间不共享内存，需要通过网络通信的方式交换数据。本文主要聊一下 分布式系统（尤其指分布式计算系统）的 基本组成和理论问题。

一个有意思的事情是现在已经不提倡使用Master-Slave 来指代这种主从关系了，毕竟 Slave 有奴隶的意思，在美国这种严禁种族歧视的国度，这种表述有点政治不正确了，所以目前大部分的系统都改成 Leader-Follower 了。

## 分布式的难点

《大数据经典论文解读》作为一个分布式的数据系统，它就需要满足三个特性，也就是可靠性、可扩展性和可维护性。
1. 作为一个数据系统，我们需要**可靠性**。如果只记录一份数据，那么当硬件故障的时候就会遇到丢数据的问题，所以我们需要对数据做复制。而**数据复制**之后，以哪一份数据为准，又给我们带来了主从架构、多主架构以及无主架构的选择。然后，在最常见的主从架构里，我们根据复制过程，可以有同步复制和异步复制之分。同步复制的节点可以作为高可用切换的 Backup Master，而异步复制的节点只适合作为只读的 Shadow Master。
2. 第二个重要的特性是**可扩展性**。在“大数据”的场景下，单个节点存不下所有数据，于是就有了**数据分区**。常见的分区方式有两种，第一种是通过区间进行分片，典型的代表就是 Bigtable，第二种是通过哈希进行分区，在大型分布式系统中常用的是一致性 Hash，典型的代表是 Cassandra。
3. 最后一点就是整个系统的**可维护性**。我们需要考虑**容错**，在硬件出现故障的时候系统仍然能够运作。我们还需要考虑**恢复**，也就是当系统出现故障的时候，仍能快速恢复到可以使用的状态。而为了确保我们不会因为部分网络的中断导致作出错误的判断，我们就需要利用共识算法，来确保系统中能够对哪个节点正在正常服务作出判断。这也就引出了**CAP** 这个所谓的“不可能三角”。

而分布式系统的核心问题就是 CAP 这个不可能三角，我们需要在一致性、可用性和分区容错性之间做权衡和选择。因此，我们选择的主从架构、复制策略、分片策略，以及容错和恢复方案，都是根据我们实际的应用场景下对于 CAP 进行的权衡和选择。

**单节点的存储引擎**：然而，即使是上万台的分布式集群，最终还是要落到每一台单个服务器上完成数据的读写。那么在存储引擎上，关键的技术点主要包括三个部分。
1. 第一个是**事务**。在写入数据的时候，我们需要保障写入的数据是原子的、完整的。在传统的数据库领域，我们有 **ACID** 这样的事务特性，也就是原子性（Atomic）、一致性（Consistency）、隔离性（Isolation）以及持久性（Durability）。而在大数据领域，很多时候因为分布式的存在，我们常常会退化到一个叫做 **BASE** 的模型。BASE 代表着基本可用（Basically Available）、软状态（Soft State）以及最终一致性（Eventually Consistent）。不过无论是 ACID 还是 BASE，在单机上，我们都会使用预写日志（WAL）、快照（Snapshot）和检查点（Checkpoints）以及写时复制（Copy-on-Write）这些技术，来保障数据在单个节点的写入是原子的。而只要写入的数据记录是在单个分片上，我们就可以保障数据写入的事务性，所以我们很容易可以做到单行事务，或者是进一步的实体组（Entity Group）层面的事务。
2. 第二个是底层的数据是**如何写入和存储**的。这个既要考虑到计算机硬件的特性，比如数据的顺序读写比随机读写快，在内存上读写比硬盘上快；也要考虑到我们在算法和数据结构中的时空复杂度，比如 Hash 表的时间复杂度是 O(1)，B+ 树的时间复杂度是 O(logN)。这样，通过结合硬件性能、数据结构和算法特性，我们会看到分布式数据库最常使用的，其实是基于 LSM 树（Log-Structured Merge Tree）的 MemTable+SSTable 的解决方案。
3. 第三个则是数据的序列化问题。出于存储空间和兼容性的考虑，我们会选用 Thrift 这样的二进制序列化方案。而为了在分析数据的时候尽量减少硬盘吞吐量，我们则要研究 Parquet 或者 ORCFile 这样的列存储格式。然后，为了在 CPU、网络和硬盘的使用上取得平衡，我们又会选择 Snappy 或者 LZO 这样的快速压缩算法。

分布式问题，往往脱胎于少量经典论文的算法证明；单节点的存储引擎，也是一个自计算机诞生起就被反复研究的问题，这两者其实往往是经典论文的再现。但是在上千个服务器上的计算引擎应该怎么做，则是一个巨大的工程实践问题，我们没有太多可以借鉴的经验。这也是为什么计算引擎的迭代和变化是最大的。不过随着 Dataflow 论文的发表，我们可以看到整个大数据的处理引擎，逐渐收敛成了一个统一的模型，大数据领域发展也算有了一个里程碑。

## 从哪里入手

有没有一个 知识体系/架构、描述方式，能将分布式的各类知识都汇总起来？当你碰到一个知识点，可以place it in context

![](/public/upload/distribute/study_distribute_system.png)

[分布式学习最佳实践：从分布式系统的特征开始（附思维导图）](https://www.cnblogs.com/xybaby/p/8544715.html) 作者分享了“如何学习分布式系统”的探索历程， 深有体会。**当一个知识点很复杂时，如何学习它，也是一门学问。**

## 大咖文章

2018.11.25 补充 [可能是讲分布式系统最到位的一篇文章](http://www.10tiao.com/html/46/201811/2651011019/1.html)回答了几个问题：

1. “分布式系统”等于 SOA、ESB、微服务这些东西吗？
2. “分布式系统”是各种中间件吗？中间件起到的是标准化的作用。**中间件只是承载这些标准化想法的介质、工具**，可以起到引导和约束的效果，以此起到大大降低系统复杂度和协作成本的作用。为了在软件系统的迭代过程中，避免将精力过多地花费在某个子功能下众多差异不大的选项中。
3. 海市蜃楼般的“分布式系统”

我们先思考一下“软件”是什么。 软件的本质是一套代码，而代码只是一段文字，除了提供文字所表述的信息之外，本身无法“动”起来。但是，想让它“动”起来，使其能够完成一件我们指定的事情，前提是需要一个宿主来给予它生命。这个宿主就是计算机，它可以让代码变成一连串可执行的“动作”，然后通过数据这个“燃料”的触发，“动”起来。这个持续的活动过程，又被描述为一个运行中的“进程”。

[分布式系统的本质其实就是这两个问题](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651011140&idx=1&sn=37b734deb9523dbde221708baa43fb39&chksm=bdbec0178ac9490102e6072967092b5a04445bbe8f2bcf95a154f4e5d7eaf1717a342e7650b5&scene=27#wechat_redirect)

1. 分治
2. 冗余，为了提高可靠性
3. 再连接，如何将拆分后的各个节点再次连接起来，从模式上来说，主要是去中心化与中心化之分。前者完全消除了中心节点故障带来的全盘出错的风险，却带来了更高的节点间协作成本。后者通过中心节点的集中式管理大大降低了协作成本，但是一旦中心节点故障则全盘出错。

[三问阿里云：CIPU究竟是什么？](https://mp.weixin.qq.com/s/_AA_OvI3jw-3s54yfiRfzw)阿里云基础产品首席架构师黄瑞瑞将云计算的发展分成分布式技术阶段、资源池化技术阶段和如今的 CIPU 阶段。
1. 分布式架构严格来说还不是云，只是企业内部使用了相应的分布式架构去解决自身扩展性的问题。在未使用分布式技术前，企业通过不停地增加小型机或者数据库的方式应对计算任务的增加，并不具备可扩展性、且缺少性价比。分布式技术让企业不再需要专门采购一些专用的大型机或者定向购买小型机，解放了供应链的弹性。
2. 但是企业由于业务状态不同，**对于计算算力的要求会有波峰和波谷**。不同公司的不同 IT 部门开始引入相对应的虚拟化技术，实现分时复用，解决单个企业内集群资源利用率相对比较低的问题。分布式架构下一个时代就是公有云，从技术的维度去看就是资源池化的时代。资源池化的关键技术能力在于能否将云上的资源提供给对应的弹性。将云上的计算、存储、网络等技术的算力资源，通过不同的搭配方式，在云上搭建各种各样的应用。从技术的维度看，资源池化阶段，虚拟化技术再向前一步，将不同的物理资源变成虚拟化的资源，变成统一的池化管理。资源池化将计算的虚拟化资源、存储的虚拟化资源、网络的虚拟化资源放在一起管理，企业就不需要承担早期分布式阶段中自己管理基础资源、不需要自己管理虚拟化资源。云厂商们迎来了新的业务需求挑战：企业在进行上云时不仅仅注重技术问题，也更加关注经济问题。
[阿里云新一代云计算体系架构 CIPU 到底是啥？超全技术解读来了](https://mp.weixin.qq.com/s/EMgTu79k-KKmo8TrNCiLrw)
1. 计算、网络、存储等基础设施层面发展的不同步，将对数据库和大数据等 PaaS 层的系统架构产生关键影响

### 不要沉迷与具体的算法

[distributed-systems-theory-for-the-distributed-systems-engineer](http://the-paper-trail.org/blog/distributed-systems-theory-for-the-distributed-systems-engineer/) 文中提到：

My response of old might have been “well, here’s the FLP paper, and here’s the Paxos paper, and here’s the Byzantine generals（拜占庭将军） paper…”, and I’d have prescribed(嘱咐、规定) a laundry list of primary source material which would have taken at least six months to get through if you rushed. **But I’ve come to thinking that recommending a ton of theoretical papers is often precisely the wrong way to go about learning distributed systems theory (unless you are in a PhD program).** Papers are usually deep, usually complex, and require both serious study, and usually significant experience to glean(捡拾) their important contributions and to place them in context. What good is requiring that level of expertise of engineers?

也就是说，具体学习某一个分布式算法用处有限。一个很难理解，一个是你很难  place them in contex（它们在解决分布式问题中的作用）。


## 分布式系统不可能三角——CAP

2020.06.28补充：[条分缕析分布式：到底什么是一致性？](https://mp.weixin.qq.com/s/qnvl_msvw0XL7hFezo2F4w)在历史上，CAP定理具有巨大的知名度，但它实际的影响力却没有想象的那么大。随着分布式理论的发展，我们逐渐认识到，CAP并不是一个「大一统」的理论，远不能涵盖分布式系统设计中方方面面的问题。相反，CAP引发了很多误解和理解上的混乱（细节不讨论了）。

下文来自 李运华 极客时间中的付费专栏《从0到1学架构》和韩健的《分布式协议和算法实战》。

Robert Greiner （http://robertgreiner.com/about/） 对CAP 的理解也经历了一个过程，他写了两篇文章来阐述CAP理论，第一篇被标记为outdated

||第一版|第二版|韩健的《分布式协议和算法实战》|
|---|---|---|---|
|地址|http://robertgreiner.com/2014/06/cap-theorem-explained/|http://robertgreiner.com/2014/08/cap-theorem-revisited/|
|理论定义|Any distributed system cannot guaranty C, A, and P simultaneously.|In a distributed system (a collection of interconnected nodes that share data.), you can only have two out of the following three guarantees across a write/read pair: Consistency, Availability, and Partition Tolerance - one of them must be sacrificed.<br>在一个分布式系统（指互相连接并共享数据的节点的集合）中，当涉及读写操作时，只能保证一致性（Consistence）、可用性（Availability）、分区容错性（Partition Tolerance）三者中的两个，另外一个必须被牺牲。|
|一致性|all nodes see the same data at the same time|A read is guaranteed to return the most recent write for a given client 总能读到 最新写入的新值|客户端的每次读操作，不管访问哪个节点，要么读到的都是同一份最新的数据，要么读取失败|
|可用性|Every request gets a response on success/failure|A non-failing node will return a reasonable response within a reasonable amount of time (no error or timeout)|任何来自客户端的请求，不管访问哪个节点，都能得到响应数据，但不保证是同一份最新数据|
|分区容忍性|System continues to work despite message loss or partial failure|The system will continue to function when network partitions occur|当节点间出现任意数量的消息丢失或高延迟的时候，系统仍然可以继续提供服务|


1. 只要有网络交互就一定会有延迟和数据丢失，也就是说，分区容错性（P）是前提，是必须要保证的，**于是只能在可用性和一致性两者间做出选择。**当网络分区失效，也就是网络不可用的时候，如果选择了一致性，系统就可能返回一个错误码或者干脆超时，即系统不可用。如果选择了可用性，那么系统总是可以返回一个数据，但是并不能保证这个数据是最新的。在工程上，我们关注的往往是如何在保持相对一致性的前提下，提高系统的可用性。
2. 大部分人对 CAP 理论有个误解，认为无论在什么情况下，分布式系统都只能在 C 和 A 中选择 1 个。 其实，在不存在网络分区的情况下，也就是分布式系统正常运行时（这也是系统在绝大部分时候所处的状态），就是说在不需要 P 时，C 和 A 能够同时保证。
3. 还是读写问题。[多线程](http://qiankunli.github.io/2014/10/09/Threads.html) 提到，多线程本质是一个并发读写问题，数据库系统中，为了描述并发读写的安全程度，还提出了隔离性的概念。具体到cap 理论上，副本一致性本质是并发读写问题（A主机写入的数据，B主机多长时间可以读到，或者说B主机也在写同一个数据）。

[从CAP理论到Paxos算法](http://blog.longjiazuo.com/archives/5369?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) 

[服务发现技术选型那点事儿](https://mp.weixin.qq.com/s/boh5smQ6ApTwScKYyhuD-Q) Eureka通过“最大努力的复制（best effort replication）” 可以让整个模型变得简单与高可用，我们在进行 A -> B 的调用时，服务 A 只要读取一个 B 的地址，就可以进行 RESTful 请求，如果 B 的这个地址下线或不可达，则有 Hystrix 之类的机制让我们快速失败。PS：也就是不单纯局限于 CAP 来考虑系统的可用性

