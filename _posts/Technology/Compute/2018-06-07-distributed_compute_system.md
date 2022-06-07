---

layout: post
title: 如何分发计算
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

## 计算 与 数据


1. 计算与数据 不在一个地方，比如常规业务系统，很少业务和数据库是同一台机器
3. 计算和数据在一起

	* 计算跟着数据走，比如hadoop、spark等，当然，必要的时候数据还得挪挪窝。
	* 数据跟跟着计算走，比如storm。这也是为什么，我们不建议在storm 中调用rpc，因为这样 就又是将 storm 退化为常规业务系统。
	* 数据和计算放在一起，这是性能最高的方式。不是通过rpc 去各地强拉数据源，而是将各路数据推向 同一个位置，计算只管 处理数据即可。

移动计算程序到数据所在位置进行计算是如何实现的呢？

1. 将待处理的大规模数据存储在服务器集群的所有服务器上，主要使用 HDFS 分布式文件存储系统，将文件分成很多块（Block），以块为单位存储在集群的服务器上。
2. 大数据引擎根据集群里不同服务器的计算能力，在每台服务器上启动若干分布式任务执行进程，这些进程会等待给它们分配执行任务。
3. 使用大数据计算框架支持的编程模型进行编程，比如 Hadoop 的 MapReduce 编程模型，或者 Spark 的 RDD 编程模型。应用程序编写好以后，将其打包，MapReduce 和 Spark 都是在 JVM 环境中运行，所以打包出来的是一个 Java 的 JAR 包。
4. 用 Hadoop 或者 Spark 的启动命令执行这个应用程序的 JAR 包，首先执行引擎会解析程序要处理的数据输入路径，根据输入数据量的大小，将数据分成若干片（Split），每一个数据片都分配给一个任务执行进程去处理。
5. **任务执行进程收到分配的任务后，检查自己是否有任务对应的程序包，如果没有就去下载程序包，下载以后通过反射的方式加载程序。走到这里，最重要的一步，也就是移动计算就完成了**。PS：传递是jar包，但是 worker 进程加载的应该是 Map 或Reduce类 人不是执行jar本身的main函数，jar包本身的main函数是提交任务用的
6. 加载程序后，任务执行进程根据分配的数据片的文件地址和数据在文件内的偏移量读取数据，并把数据输入给应用程序相应的方法去执行，从而实现在分布式服务器集群中移动计算程序，对大规模数据进行并行处理的计算目标。

## 随便玩玩

学习分布式应用系统的路径最好是

1. 一个简单的任务分发系统。将一个可执行文件、main函数 下发到一台 特定主机并执行。
2. 下发代码， 上一点是下发main函数，工作节点收到后直接另起进程运行就行了。下发代码即，工作节点另起 线程执行。**这其实跟rpc 差不多，只是rpc 事先定义好了函数和接口，逻辑比较受限。**
2. 监控任务节点、可执行任务运行监控、重启等
3. 下发一个复杂任务，这个任务需要多个任务节点 协作执行，这需要任务节点间通信等
4. 学习storm，相对通用的复杂任务抽象，高效的节点间通信机制等
5. 学习hadoop，节点间通信 直接成了 读写分布式文件系统，使得对外抽象得以简化。
6. 学习spark，节点间 通信直接成了“内存复制”，并利用函数式思想，简化了对外api
	
[基于任务调度的企业级分布式批处理方案](https://mp.weixin.qq.com/s/prmaTNv_c5cRPrVXojwung)	

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
