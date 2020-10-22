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

本文主要聊一下 分布式系统（尤其指分布式计算系统）的 基本组成和理论问题

一个有意思的事情是现在已经不提倡使用Master-Slave 来指代这种主从关系了，毕竟 Slave 有奴隶的意思，在美国这种严禁种族歧视的国度，这种表述有点政治不正确了，所以目前大部分的系统都改成 Leader-Follower 了。

## 分布式的难点

[分布式算法小结[1]: Broadcast Abstraction](https://zhuanlan.zhihu.com/p/93322366?utm_source=qq&utm_medium=social&utm_oi=550222171771973632)分布式系统的特点和难点在于不确定性，这种不确定性主要来自于:

1. 旧信息: 得到的信息"可能是过老而无用的信息", 没有有效的手段来瞬间得知其他process的状况, 所以要在"不清楚自己现在得到的信息到底有没有过时的不确定性下做决定"
2. 弱控制: 很多因素无法控制, 比如消息是否丢失什么时候会丢失, 进程是否被杀掉, 机器是否挂了。
3. 不可靠: 通信是不可靠的, 机器是会挂的, 消息是会丢失的, 时钟是会偏移的; 且到底消息传到了么？到底机器挂了么？都无法准确的检测.

**利用不准确的信息, 很弱的控制在不可靠的“基础设施”上构建非常可靠的系统, 这是分布式系统独特的艺术, 也是这个领域的魅力所在**. (还有一个我很喜欢的比喻是: 用一块正确率只有99.99%的cpu, 写出正确率100%的程序来.)

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


1. 只要有网络交互就一定会有延迟和数据丢失，也就是说，分区容错性（P）是前提，是必须要保证的，**于是只能在可用性和一致性两者间做出选择。**在工程上，我们关注的往往是如何在保持相对一致性的前提下，提高系统的可用性。
2. 大部分人对 CAP 理论有个误解，认为无论在什么情况下，分布式系统都只能在 C 和 A 中选择 1 个。 其实，在不存在网络分区的情况下，也就是分布式系统正常运行时（这也是系统在绝大部分时候所处的状态），就是说在不需要 P 时，C 和 A 能够同时保证。
3. 还是读写问题。[多线程](http://qiankunli.github.io/2014/10/09/Threads.html) 提到，多线程本质是一个并发读写问题，数据库系统中，为了描述并发读写的安全程度，还提出了隔离性的概念。具体到cap 理论上，副本一致性本质是并发读写问题（A主机写入的数据，B主机多长时间可以读到，或者说B主机也在写同一个数据）。

[从CAP理论到Paxos算法](http://blog.longjiazuo.com/archives/5369?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io) 

[服务发现技术选型那点事儿](https://mp.weixin.qq.com/s/boh5smQ6ApTwScKYyhuD-Q) Eureka通过“最大努力的复制（best effort replication）” 可以让整个模型变得简单与高可用，我们在进行 A -> B 的调用时，服务 A 只要读取一个 B 的地址，就可以进行 RESTful 请求，如果 B 的这个地址下线或不可达，则有 Hystrix 之类的机制让我们快速失败。PS：也就是不单纯局限于 CAP 来考虑系统的可用性

