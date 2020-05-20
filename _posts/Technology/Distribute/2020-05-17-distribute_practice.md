---

layout: post
title: hashicorp raft源码学习
category: 技术
tags: Distribute
keywords: 分布式系统

---

## 简介

* TOC
{:toc}

本文主要内容 来自极客时间《分布式协议与算法实践》

## 实现

[hashicorp/raft](https://github.com/hashicorp/raft) 是常用的 Golang 版 Raft 算法的实现，被众多流行软件使用，如 Consul、InfluxDB、IPFS 等

1. Hashicorp Raft是个package，可以将它理解成库（lib），是没有main函数的。 
2. raft.go 是 Hashicorp Raft 的核心代码文件，大部分的核心功能都是在这个文件中实现。
3. 提供了一个fuzzy 包，用来模拟测试 启动一个集群， 启动多个raftNode 节点。
4. 在 Hashicorp Raft 中，支持两种节点间通讯机制，内存型和 TCP 协议型，其中，内存型通讯机制，主要用于测试，2 种通讯机制的代码实现，分别在文件 inmem_transport.go 和 tcp_transport.go 中。

```
raft
    fuzzy   
    raft.go
    state.go    // 节点状态相关的数据结构和函数
        type RaftState uint32
        type raftState struct 
    commands.go // RPC 消息相关的数据结构
        type AppendEntriesRequest struct    // 日志复制 RPC 的请求消息
        type AppendEntriesResponse struct
        type RequestVoteRequest struct
        type RequestVoteResponse struct
        type InstallSnapshotRequest struct
        type InstallSnapshotResponse struct
    log.go      // 日志对应的数据结构和函数接口
        type Log struct
        type LogStore interface
```

![](/public/upload/distribute/raft_object.png)

在 Hashicorp Raft 中，可以通过 NewRaft() 函数来创建 Raft 节点。**跟随者、候选人、领导者 3 种节点状态都有分别对应的功能函数**

```go
func (r *Raft) run() {
	for {
		// Check if we are doing a shutdown
		select {
		case <-r.shutdownCh:
			// Clear the leader to prevent forwarding
			r.setLeader("")
			return
		default:
		}

		// Enter into a sub-FSM
		switch r.getState() {
		case Follower:
			r.runFollower()
		case Candidate:
			r.runCandidate()
		case Leader:
			r.runLeader()
		}
	}
}
```


## leader 选举 / runFollower

1. Follower is the initial state of a Raft node.  runFollower() ==> 等待心跳消息超时  ==> 设置节点状态为候选人，并退出 runFollower() 函数
2. 当节点推举自己为候选人之后，函数 runCandidate() 执行 ==> electSelf() 发起选举, send a RequestVote RPC to all peers and vote for
ourself ==> 进入到 for 循环中，通过 select 实现多路 IO 复用，周期性地获取消息和处理
   1. 如果发生了选举超时，退出 runCandidate() 函数，然后再重新执行 runCandidate() 函数，发起新一轮的选举。
   2. 如果候选人在指定时间内赢得了大多数选票，那么候选人将当选为领导者，调用 setState() 函数，将自己的状态变更为领导者，并退出 runCandidate() 函数。
3. 当节点当选为领导者后，函数 runLeader() 执行 ==> startStopReplication: Start a replication routine for each peer ==> replicate 针对每个节点 主要干两个活儿
   1. heartbeat， 周期性地发送心跳信息，通知其他节点，我是领导者，我还活着，不需要你们发起新的选举。
   2. replicateTo， 执行日志复制功能

## 日志复制

数据结构

```go
// raft/log.go
type Log struct {
    Index uint64 // 索引值
    Term uint64 // 任期编号
    Type LogType // 日志项类别
    Data []byte  // 指令
    Extensions []byte // 扩展信息
}
type LogStore interface {
	FirstIndex() (uint64, error)
	LastIndex() (uint64, error)
	GetLog(index uint64, log *Log) error
	StoreLog(log *Log) error
	StoreLogs(logs []*Log) error
	DeleteRange(min, max uint64) error
}
```

leader复制日志和follower接收日志的入口函数，应该分别在 runLeader() 和 runFollower() 函数中调用的

### leader 复制日志
raft.runLeader() ==> r.startStopReplication() ==> 对每个节点 replicate ==> heartbeat; replicateTo ==> r.trans.AppendEntries

流水线复制模式

1. 在不需要进行日志一致性检测，复制功能已正常运行的时候，开启了流水线复制模式（对应pipelineReplicate 函数）
2. 目标是在环境正常的情况下，提升日志复制性能，如果在日志复制过程中出错了，就进入 RPC 复制模式，继续调用 replicateTo() 函数，进行日志复制。

### follower 接收日志

runFollower ==> processRPC ==> appendEntries

1. 比较日志一致性
2. 将新日志项存放在本地
3. 根据领导者最新提交的日志项索引值，来计算当前需要被应用的日志项，并应用到本地状态机。

## InfluxDB 企业版的架构

InfluxDB 企业版是由 META 节点和 DATA 节点 2 个逻辑单元组成的
1. META 节点存放的是系统运行的关键元信息，比如数据库（Database）、表（Measurement）、保留策略（Retention policy）等。它的特点是一致性敏感，但读写访问量不高，需要一定的容错能力。
2. DATA 节点存放的是具体的时序数据。它有这样几个特点：最终一致性、面向业务、性能越高越好，除了容错，还需要实现水平扩展，扩展集群的读写性能。

**对于 META 节点来说，节点数的多少代表的是容错能力**，一般 3 个节点就可以了，因为从实际系统运行观察看，能容忍一个节点故障就可以了。但**对 DATA 节点而言，节点数的多少则代表了读写性能**，一般而言，在一定数量以内（比如 10 个节点）越多越好，因为节点数越多，读写性能也越高，但节点数量太多也不行，因为查询时就会出现访问节点数过多而延迟大的问题。

1. META 节点需要强一致性，实现 CAP 中的 CP 模型。使用 Raft 算法实现 META 节点的一致性（一般推荐 3 节点的集群配置）
2. DATA 节点存放的是具体的时序数据，对一致性要求不高，实现最终一致性就可以了。但是，DATA 节点也在同时作为接入层直接面向业务，考虑到时序数据的量很大，要实现水平扩展，所以必须要选用 CAP 中的 AP 模型，因为 **AP 模型不像 CP 模型那样采用一个算法（比如 Raft 算法）就可以实现了**

也就是说，AP 模型更复杂，具体有这样几个实现步骤。
1. 自定义副本数
2. Hinted-handoff。一个节点接收到写请求时，需要将写请求中的数据转发一份到其他副本所在的节点，那么在这个过程中，远程 RPC 通讯是可能会失败的，比如网络不通了、目标节点宕机了、临时的突发流量也会导致系统过载。那么如何处理这种情况呢？答案是实现 Hinted-handoff。在 InfluxDB 企业版中，Hinted-handoff 是这样实现的:
    1. 写失败的请求，会缓存到本地硬盘上 ;
    2. 周期性地尝试重传 ;
    3. 相关参数信息，比如缓存空间大小 (max-szie)、缓存周期（max-age）、尝试间隔（retry-interval）等，是可配置的。

    虽然 Hinted-handoff 可以通过重传的方式来处理数据不一致的问题，但当写失败请求的数据大于本地缓存空间时，比如某个节点长期故障，写请求的数据还是会丢失的，最终的节点的数据还是不一致的，那么怎么实现数据的最终一致性呢？答案是反熵。
3. 反熵。一种异步修复、实现最终一致性的协议

## 其它

在海量系统中建议直面问题，通过技术手段在代码和架构层面解决它，而不是引入和堆砌更多的开源软件。