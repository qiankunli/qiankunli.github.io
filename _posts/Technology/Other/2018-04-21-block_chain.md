---

layout: post
title: 区块链泛谈
category: 技术
tags: Other
keywords: block chain

---

## 前言（未完成）

[以太坊白皮书](https://github.com/ethereum/wiki/wiki/White-Paper)(未读)

[以太坊白皮书中文版](https://ethfans.org/posts/ethereum-whitepaper)

是什么？
为什么重要？
重点在哪里？
	
## 区块链

结构

1. 交易，区块链所存储的 数据
2. 区块，hash(区块数据  + nonce) 符合一个规定的结果。格式上合法 
3. 链，区块被大部分节点 接受，即区块所存储的交易数据 得到确认。区块在节点内采用 链式存储。

	
[区块链以及区块链技术总结](https://zhuanlan.zhihu.com/p/22609209)

1. 100个节点分散在广域网中，且每个的节点数据都是一样的
2. 分布式的冗余的链式总帐本方案


重点在哪里：冗余。

区块链 和 拜占庭将军的渊源 [ 图解区块链：14张图看懂什么是“区块链技术”？](https://blog.csdn.net/wo541075754/article/details/54743138)


[区块链演义](https://zhuanlan.zhihu.com/c_106064493)(未读)




## 共识

《区块链原理、设计与应用》 提到：Proof of work 不是实现 面向最终确认的共识，而是基于概率、随时间 逐步 增强确认的共识。现有达成的 结果 在理论上 可能被推翻（A矿工的链没有B的长），只是攻击者要付出的代价随时间 而指数级上升，被推翻的可能性随之指数级 下降。此外，考虑到 Internet 的尺度，达成共识的时间相对比较长，因此按照 区块（一组交易） 来进行阶段性的确认（快照），从而提高 网络整体的可用性（否则就是一个交易来一次共识了，也就没有区块链了）。

##  书籍

《区块链核心算法解析》 据说评价很高，蚂蚁金服cto 推荐

[区块链系统的思考框架](https://zhuanlan.zhihu.com/p/35967209) 要点：

1. 作者的思考路径[重温比特币论文](https://mp.weixin.qq.com/s?__biz=MzUzNTI3NjkwMw==&mid=2247483757&idx=1&sn=bc6d71d981178609a62b2c5d1e6ea588&chksm=fa86bd65cdf13473f0b78612071df3ba1e8da8c761038393e01924927dae8cd2aed36740a680&scene=21#wechat_redirect) [为区块链呐喊几声](https://mp.weixin.qq.com/s?__biz=MzUzNTI3NjkwMw==&mid=2247483766&idx=1&sn=02cdc48d6676bc2c720d6b89bc5e59d3&chksm=fa86bd7ecdf134680231c266162e9c66116f218a2316754494dad6045da3566d2c4e724df8b6&scene=21#wechat_redirect)[区块链技术实践过程中的一些思考](https://mp.weixin.qq.com/s?__biz=MzUzNTI3NjkwMw==&mid=2247483772&idx=1&sn=2a509a471be80fb7e49652cf0d0b40c2&chksm=fa86bd74cdf13462e366ce809fc5cce196405930cc00235172abbece188cf310858666f7292b&scene=21#wechat_redirect)
2. 投入大，产出少，这种无法满足“短、平、快”需求的事情，最终结果就是：**某个开源项目独领风骚**。让少数人去搞定最困难的事情，大多数人直接使用他们的成果，整体上将是最优的，这是社会自动调整的结果。旁白：总结的太精辟了
3. 剩下的内容还跟不上作者的思路

[我的数据，凭什么让我入链？](https://mp.weixin.qq.com/s?__biz=MzUzNTI3NjkwMw==&mid=2247483788&idx=1&sn=13d60d6a9986e13c94b63a13e564b4f5&chksm=fa86bd84cdf13492d881d0e7bbabbc37b229456c1da73af0ae934f0fcd853bb0c0d6d3f726aa&mpshare=1&scene=23&srcid=04245TRCWGNY9pZLhj7at4U5#rd) 要点：

1. 区块链可以为那些势力均等、离不开彼此，但又互相不信任的主体，提供安全感。
2. 国际清算、原油贸易等，存在着超出我们认知的低效率环节。如果是被信任问题卡住了信息化进程，那么引入区块链后，会有立杆见影的效果。**旁白：我们重新考虑下拜占庭将军问题及其代表的信任问题， 如果拜占庭将军问题 普适性足够， 那么区块链 ==> 拜占庭将军问题 ==> 信任问题，便是说的通的。**
2. 当遇到一个具体的应用场景的时候，可以先从下面这个方向思考一下：**这是一个信息化问题，还是一个信任问题？**
3. 比如在供应链这个场景中，首要的是信息化和原始数据的真实性。或许让一个有公信力的公司做一套系统，以公司信誉为担保，效果更好。