---

layout: post
title: 副本一致性
category: 技术
tags: Distribute
keywords: 一致性协议

---

## 简介（未完成）

* TOC
{:toc}



## Paxos

Paxos算法

	* Phase 1
		
		* proposer向网络内超过半数的acceptor发送prepare消息
		* acceptor正常情况下回复promise消息
	* Phase 2
		* 在有足够多acceptor回复promise消息时，proposer发送accept消息
		* 正常情况下acceptor回复accepted消息

只有一个Proposer能进行到第二阶段运行。

目前比较好的通俗解释，以贿选来描述 [如何浅显易懂地解说 Paxos 的算法？ - GRAYLAMB的回答 - 知乎](https://www.zhihu.com/question/19787937/answer/107750652)。

一些补充

1. proposer 和 acceptor，异类交互，同类不交互

	![](/public/upload/architecture/distributed_system_2.png)
	
2. proposer 贿选 不会坚持 让acceptor 遵守自己的提议。出价最高的proposer 会得到大部分acceptor 的拥护（谁贿金高，acceptor最后听谁的。换个说法，acceptor 之间没有之间交互，但），  但会以最快达成一致 为目标，若是贿金高但提议晚，也是会顺从 他人的提议。
