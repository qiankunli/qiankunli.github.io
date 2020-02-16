---

layout: post
title: 分布式计算系统的那些套路
category: 技术
tags: Compute
keywords: 分布式系统

---

## 简介

参见[Spark Stream 学习](http://qiankunli.github.io/2018/05/27/spark_stream.html)  中对spark stream 和storm 对比一节，有以下几点：

1. 分布式计算系统，都是用户以代码的方式预定义好计算逻辑，系统将计算 下发到各个节点。这一点都是一样的，不同处是对外提供的抽象不同。比如spark的`rdd.filter(function1).map(function2)`，而在storm 中则可能是 两个bolt
2. 任务分片：有的计算 逻辑在一个节点即可执行完毕，比如不涉及分区的spark rdd，或分布式运行一个shell。有的计算逻辑则 拆分到不同节点，比如storm和mapreduce。此时系统就要做好 调度和协调。

## 常规业务系统也是分布式系统

最近研究一个系统设计方案，学习spark、storm等，包括跟同事交流，有以下几个感觉

1. 我们平时的系统其实也是分布式系统，若是归纳起来， 很多做法跟分布式系统差不多。比如你通过jdbc 访问mysql，spark 也是，spark rdd 做数据处理，我们又何尝不是。因此，特定的业务上，也没必要一定套spark、storm这些，系统的瓶颈有时也不是 spark、storm 可以解决的。
2. 笔者以前熟悉的项目，都是一个个独立的节点，节点是按功能划分的，谈不上主次，几个功能的节点组合形成架构。分布式系统也包括多个节点，但通常有Scheduler和Executor，业务功能都由Executor 完成，Scheduler 监控和调度Executor。
2. spark、storm 这些系统 一个很厉害的地方在于，抽象架设在分布式环境下。比如spark 的rdd，storm的topology/spout/bolt 这些。笔者以前的业务系统也有抽象，但抽象通常在单机节点内。
3. 部署方式上，也跟笔者熟悉的tomcat、springboot jar 有所不同
	
	1. 代码本身是一个进程，即定了main 函数
	2. 通常有一个额外的提交工作比如spark-submit 等

4. [JStorm概叙 & 应用场景](https://github.com/alibaba/jstorm/wiki/%E6%A6%82%E5%8F%99-&-%E5%BA%94%E7%94%A8%E5%9C%BA%E6%99%AF) 中有一句话：

	* 从应用的角度，JStorm应用是一种遵守某种编程规范的分布式应用。
	* 从系统角度， JStorm是一套类似MapReduce的调度系统。 
	* 从数据的角度，JStorm是一套基于流水线的消息处理机制。

6. 既然大致差不多，通常也可以用storm 来优化甚至 替换 业务系统的一些设计。比如storm 确实 减少或隐藏了 数据流转中的序列化、失败重试等问题。

分布式系统也有点常规系统的意思

1. 在storm中， Topology 的定义是一个Thrift结构，并且Nimbus 就是一个Thrift 服务

## 计算 与 数据

1. 计算与数据 不在一个地方，比如常规业务系统，很少业务和数据库是同一台机器
3. 计算和数据在一起

	* 计算跟着数据走，比如hadoop、spark等，当然，必要的时候数据还得挪挪窝。
	* 数据跟跟着计算走，比如storm。这也是为什么，我们不建议在storm 中调用rpc，因为这样 就又是将 storm 退化为常规业务系统。
	* 数据和计算放在一起，这是性能最高的方式。不是通过rpc 去各地强拉数据源，而是将各路数据推向 同一个位置，计算只管 处理数据即可。

## 学习路径

学习分布式应用系统的路径最好是

1. 一个简单的任务分发系统。将一个可执行文件、main函数 下发到一台 特定主机并执行。
2. 下发代码， 上一点是下发main函数，工作节点收到后直接另起进程运行就行了。下发代码即，工作节点另起 线程执行。**这其实跟rpc 差不多，只是rpc 事先定义好了函数和接口，逻辑比较受限。**
2. 监控任务节点、可执行任务运行监控、重启等
3. 下发一个复杂任务，这个任务需要多个任务节点 协作执行，这需要任务节点间通信等
4. 学习storm，相对通用的复杂任务抽象，高效的节点间通信机制等
5. 学习hadoop，节点间通信 直接成了 读写分布式文件系统，使得对外抽象得以简化。
6. 学习spark，节点间 通信直接成了“内存复制”，并利用函数式思想，简化了对外api
	
## 将计算异地执行

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

## 节点之间的协作

基于上一节，我们设想下 类似mapreduce的 效果如何实现。一个节点运行map，然后另一个节点执行reduce，最后输出结果。

简单点，不考虑容错、健壮及通信、运行效率

1. 驱动节点 与 jobTrack 交互，获知在哪个机器上执行 map，哪个机器上 执行reduce
2. 将 conf、map.class 及其 依赖jar 发往 worker 节点，运行完毕后，向驱动节点 汇报结果。
3. 驱动节点同时命令 map 的worker 节点将map 结果 发往 reduce worker 节点，驱动节点将conf、reduce.class 及其依赖jar 发往 worker 节点，运行完毕后，向驱动节点 汇报结果。

后续 笔者会根据 storm 等源码的阅读 继续重试该文档，包括但不限于

1. 通信方式
2. 容错方式
3. 监控
