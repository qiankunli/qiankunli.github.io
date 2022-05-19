---

layout: post
title: 分布式计算系统的那些套路
category: 技术
tags: Compute
keywords: 分布式系统

---

## 简介


||大数据技术|传统数据处理技术|
|---|---|---|
|服务器规模|百、千、万|几十台服务器|
|服务器架构|x86/普通磁盘无需RAID|专用硬件：大型机、小型机；专用设备：RAID等|
|编程模型|良好的封装，无需考虑容错等一系列分布式系统问题|自己处理分布式、容错，比如MPI|

参见[Spark Stream 学习](http://qiankunli.github.io/2018/05/27/spark_stream.html)  中对spark stream 和storm 对比一节，有以下几点：

1. 分布式计算系统，都是用户以代码的方式预定义好计算逻辑，系统将计算 下发到各个节点。这一点都是一样的，不同处是对外提供的抽象不同。比如spark的`rdd.filter(function1).map(function2)`，而在storm 中则可能是 两个bolt
2. 任务分片：有的计算 逻辑在一个节点即可执行完毕，比如不涉及分区的spark rdd，或分布式运行一个shell。有的计算逻辑则 拆分到不同节点，比如storm和mapreduce。此时系统就要做好 调度和协调。

## 《大数据经典论文解读》三驾马车和基础设施

Google 能成为散播大数据火种的人，是有着历史的必然性的。作为一个搜索引擎，Google 在数据层面，面临着比任何一个互联网公司都更大的挑战。无论是 Amazon 这样的电商公司，还是 Yahoo 这样的门户网站，都只需要存储自己网站相关的数据。而 Google，则是需要抓取所有网站的网页数据并存下来。而且光存下来还不够，早在 1999 年，两个创始人就发表了 PageRank 的论文，也就是说，Google 不只是简单地根据网页里面的关键字来排序搜索结果，而是要通过网页之间的反向链接关系，进行很多轮的迭代计算，才能最终确认排序。而不断增长的搜索请求量，让 Google 还需要有响应迅速的在线服务。

1. GFS 的论文发表于 2003 年，它主要是解决了数据的存储问题。作为一个上千节点的分布式文件系统，Google 可以把所有需要的数据都能很容易地存储下来。
2. 光存下来还不够，我们还要基于这些数据进行各种计算。这个时候，就轮到 2004 年发表的 MapReduce 出场了。通过借鉴 Lisp，Google 利用简单的 Map 和 Reduce 两个函数，对于海量数据计算做了一次抽象，这就让“处理”数据的人，不再需要深入掌握分布式系统的开发了。而且他们推出的 PageRank 算法，也可以通过多轮的 MapReduce 的迭代来实现。
3. 这样，无论是 GFS 存储数据，还是 MapReduce 处理数据，系统的吞吐量都没有问题了，因为所有的数据都是顺序读写。但是这两个，其实都没有办法解决好数据的高性能随机读写问题。因此，面对这个问题，2006 年发表的 Bigtable 就站上了历史舞台了。
4. 到这里，GFS、MapReduce 和 Bigtable 这三驾马车的论文，就完成了“存储”“计算”“实时服务”这三个核心架构的设计。不过你还要知道，这三篇论文其实还依赖了两个基础设施。
	1. 第一个是为了保障数据一致性的分布式锁。Google 在发表 Bigtable 的同一年，就发表了实现了 Paxos 算法的 Chubby 锁服务的论文
	2. 第二个是数据怎么序列化以及分布式系统之间怎么通信。

进化
1. 作为一个“计算”引擎，MapReduce朝着以下方式进化
	1. 首先是编程模型。MapReduce 的编程模型还是需要工程师去写程序的，所以它进化的方向就是通过一门 DSL，进一步降低写 MapReduce 的门槛。在这个领域的第一阶段最终胜出的，是 Facebook 在 2009 年发表的 Hive。
	2. 其次是执行引擎。Hive 虽然披上了一个 SQL 的皮，但是它的底层仍然是一个个 MapReduce 的任务，所以延时很高
	3. 多轮迭代问题。在 MapReduce 这个模型里，一个 MapReduce 就要读写一次硬盘，而且 Map 和 Reduce 之间的数据通信，也是先要落到硬盘上的。这样，无论是复杂一点的 Hive SQL，还是需要进行上百轮迭代的机器学习算法，都会浪费非常多的硬盘读写。后来就有了 Spark，通过把数据放在内存而不是硬盘里，大大提升了分布式数据计算性能。
2. 作为一个“在线服务”的数据库，Bigtable 的进化是这样的
	1. 首先是事务问题和 Schema 问题。Google 先是在 2011 年发表了 Megastore 的论文，在 Bigtable 之上，实现了类 SQL 的接口，提供了 Schema，以及简单的跨行事务。如果说 Bigtable 为了伸缩性，放弃了关系型数据库的种种特性。那么 Megastore 就是开始在 Bigtable 上逐步弥补关系型数据库的特性。
	2. 其次是异地多活和跨数据中心问题。Google 在 2012 年发表的 Spanner，能够做到“全局一致性”。
3. 实时数据处理的抽象进化。首先是 Yahoo 在 2010 年发表了 S4 的论文，并在 2011 年开源了 S4。而几乎是在同一时间，Twitter 工程师南森·马茨（Nathan Marz）以一己之力开源了 Storm，并且在很长一段时间成为了工业界的事实标准。接着在 2011 年，Kafka 的论文也发表了。最早的 Kafka 其实只是一个“消息队列”，后来进化出了 Kafka Streams 这样的实时数据处理方案。2015 年，Google 发表的 Dataflow 的模型，可以说是对于流式数据处理模型做出了最好的总结和抽象。一直到现在，Dataflow 就成为了真正的“流批一体”的大数据处理架构。
4. 将所有服务器放在一起的资源调度，因为数据中心里面的服务器越来越多，我们会发现原有的系统部署方式越来越浪费。原先我们一般是一个计算集群独占一系列服务器，而往往很多时候，我们的服务器资源都是闲置的。这在服务器数量很少的时候确实不太要紧，但是，当我们有数百乃至数千台服务器的时候，浪费的硬件和电力成本就成为不能承受之重了。于是，尽可能用满硬件资源成为了刚需。由此一来，我们对于整个分布式系统的视角，也从虚拟机转向了容器，这也是 Kubernetes 这个系统的由来。

## 计算 与 数据


1. 计算与数据 不在一个地方，比如常规业务系统，很少业务和数据库是同一台机器
3. 计算和数据在一起

	* 计算跟着数据走，比如hadoop、spark等，当然，必要的时候数据还得挪挪窝。
	* 数据跟跟着计算走，比如storm。这也是为什么，我们不建议在storm 中调用rpc，因为这样 就又是将 storm 退化为常规业务系统。
	* 数据和计算放在一起，这是性能最高的方式。不是通过rpc 去各地强拉数据源，而是将各路数据推向 同一个位置，计算只管 处理数据即可。

## 随便玩玩

学习分布式应用系统的路径最好是

1. 一个简单的任务分发系统。将一个可执行文件、main函数 下发到一台 特定主机并执行。
2. 下发代码， 上一点是下发main函数，工作节点收到后直接另起进程运行就行了。下发代码即，工作节点另起 线程执行。**这其实跟rpc 差不多，只是rpc 事先定义好了函数和接口，逻辑比较受限。**
2. 监控任务节点、可执行任务运行监控、重启等
3. 下发一个复杂任务，这个任务需要多个任务节点 协作执行，这需要任务节点间通信等
4. 学习storm，相对通用的复杂任务抽象，高效的节点间通信机制等
5. 学习hadoop，节点间通信 直接成了 读写分布式文件系统，使得对外抽象得以简化。
6. 学习spark，节点间 通信直接成了“内存复制”，并利用函数式思想，简化了对外api
	
### 将计算异地执行

[huangll99/DistributedTask](https://github.com/huangll99/DistributedTask/tree/master/src/main/java/com/hll/dist)

类结构

	com.hll.dist
		common
			Constants
			Context
		io
			InputFormat
			OutputFormat
			DefaultInputFormat
			DefaultOutputFormat
		scheduler
			Runner
			WorkerClient
			WorkerRunnable
			WorkerServer
		task
			ProcessLogic
			TaskProcessor
		userapp
			UserApp
			WordCount
		

有以下几点

1. 该项目只实现了 java 代码传输和远程执行
2. WorkerClient 发送数据，WorkerServer 接收数据并执行
3. WorkerClient 发送了三个数据

	1. jar包（在实际的业务中，代码通常依赖很多第三方jar）
	2. conf 数据，此处是一个hashMap
	3. 启动命令：`java -cp xx/job.jar com.hll.dist.task.TaskProcessor`

4. WorkerServer 是一个socket server

	1. 接收jar 包存在本地
	2. 接收 conf，以文件形式存在本地
	3. 现在，WorkerServer 所在节点 具备了 可执行文件及 配置数据。
	3. 接收命令，` Process process = Runtime.getRuntime().exec(command);` 另起进程 执行`java -cp xx/job.jar com.hll.dist.task.TaskProcessor`

TaskProcessor 逻辑

1. 根据约定目录 读取conf，并反序列化为 HashMap
2. 从conf 中读取输入源 配置，并实例化 输入源

		Class<?> inputFormatClass = Class.forName(conf.get(Constants.INPUT_FORMAT));
		InputFormat inputFormat= (InputFormat) inputFormatClass.newInstance();
    	inputFormat.init(context);
    	
3. 从conf 中读取 逻辑类名，也就是WordCount，并实例化
4. 驱动输入源 读取数据，并调用 逻辑类执行

	 	while (inputFormat.hasNext()){
      		int key = inputFormat.nextKey();
      		String value = inputFormat.nextValue();
      		processLogic.process(key,value,context);
    	}
    	
5. 从conf 中读取输出配置，并实例化 输出

 	  	Class<?> outputFormatClass = Class.forName(conf.get(Constants.OUTPUT_FORMAT));
    	OutputFormat outputFormat= (OutputFormat) outputFormatClass.newInstance();
   	 	outputFormat.write(context);
   	 	
5. 退出

该项目是为了demo 展示的一个特例，从中可以看到

1. worker 在demo 中只是一个 socket 服务端，socket handler 的逻辑逻辑 就是 接收文件和 Runtime.exec 启动子进程。**从这个角度看，这与web request ==> web server ==> rpc service） 并无不同。**
2. WorkClient 向 worker 节点 传输了 jar 文件、配置文件和 运行指令。worker 节点 有输入输出、有配置、有计算逻辑（jar）
1.  子进程 的计算逻辑 **从代码上** 分为两部分，业务逻辑抽取为wordcount。驱动逻辑则负责外围的输入、输出、Context 封装等工作。（**以前一直在困惑 如何将wordcount交付给 节点执行，现在看，节点运行的 根本不是wordcount本身，wordcount 支持其中一环**）
2. **conf 在这里像是一个dsl文件，worker 节点 根据conf 这个dsl 文件加载数据、加载类（计算逻辑） 执行即可**

在实践中

1. 感觉无需 向worker 节点发送完整jar，对于特定业务，只需将wordcount.class 及其依赖jar 发往 worker 节点即可。
2. 在分布式环境下，容错、通信等都是通用的，用户只需关注 数据的处理逻辑（也就是wordcount）。从某种角度来说，worker 节点准备好 class 运行的上下文（输入，输出和线程驱动），驱动节点只要告知 类名即可 驱动业务执行。从分布式业务中暴露 几个业务逻辑 与 单机环境下暴露业务逻辑（比如netty），并无不同之处。

### 节点之间的协作

基于上一节，我们设想下 类似mapreduce的 效果如何实现。一个节点运行map，然后另一个节点执行reduce，最后输出结果。

简单点，不考虑容错、健壮及通信、运行效率

1. 驱动节点 与 jobTrack 交互，获知在哪个机器上执行 map，哪个机器上 执行reduce
2. 将 conf、map.class 及其 依赖jar 发往 worker 节点，运行完毕后，向驱动节点 汇报结果。
3. 驱动节点同时命令 map 的worker 节点将map 结果 发往 reduce worker 节点，驱动节点将conf、reduce.class 及其依赖jar 发往 worker 节点，运行完毕后，向驱动节点 汇报结果。

后续 笔者会根据 storm 等源码的阅读 继续重试该文档，包括但不限于

1. 通信方式
2. 容错方式
3. 监控
