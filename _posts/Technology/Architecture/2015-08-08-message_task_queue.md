---

layout: post
title: 消息/任务队列
category: 技术
tags: Architecture
keywords: 消息队列 rabbitmq kafka

---

## 简介

* TOC
{:toc}

## linux的消息队列

进程间通信的三大手段之一（共享内存、信号和消息队列），对于不同的通信手段，进程消费“通信数据”的方式不同。

消息队列和管道的区别：

- 管道的中间介质是文件。
- 消息队列是消息缓冲区，在内存中以队列的形式组织，传输时以消息（一个数据结构）为单位。

## 本地队列

本地队列的几种应用场景

1. 存储数据，单纯的利用下先进先出特性
2. 缓存任务/消息，缓解生产者和消费者的速度差异，便于消费者批量处理，通常用于消费者需要进行文件io或网络io的场景

	1. log4j，调用线程将日志加入到队列中，由write线程批量写入日志
	2. kafka producer

一个功能完善的队列通常要解决以下问题：

1. 有界队列还是无界队列
2. 多生产者和消费者的并发生产和消费问题
3. 性能问题

java 提供内置的队列实现，同时有必要提到一个高性能队列实现[高性能队列——Disruptor](https://tech.meituan.com/disruptor.html)，要点如下：

1. 队列要有界
2. 队列采用数组存储有利于堆管理
2. 有界队列的线程安全问题，java内置队列 都采用加锁方式实现线程安全，Disruptor采用cas实现线程安全
3. cas玩了一些花活儿：

	* 环形数组（ring buffer），这里环形不是首位相顾，数组通过下标访问，Disruptor的“下标”会一直递增，通过“下标%数组长度”得到实际的数组index。数组长度2^n
	* 假设ring buffer长度为length，则还有一个length bit的数组available buffer，用于标记ring buffer 对应下标的元素是否被生产者占用。意图就是，java内置队列加锁，同一时刻数组只能被一个线程访问。而ring buffer允许多个线程同时访问，通过检查 available buffer是否被其他线程捷足先登（**通过新增数据结构，来感知竞争激烈程度**），然后重试。**将加锁  改为 探测-重试**

## 消息队列中间件

基于内存或磁盘的队列，为分布式应用提供通信手段。

### 消息模型

消息模型可以分为两种， 队列和发布-订阅式。 

1. 队列,此时队列单纯的是一个"削峰"的作用
2. 发布-订阅模型，**在producer 和 consumer 之间提取一个独立的第三方**。可以看到，相对于“生产消费者”的生产消费，**有一个订阅的过程来描述消费者的兴趣**，不是producer发的所有消息消费者都会接

	![](/public/upload/architecture/subscribe_publish.png)

	体现在实现上，一个rabbitmq为例
	
		1. rabbitmq producer和 consumer 都明确绑定exchange
		2. rabbitmq producer发消息时，除了消息本身，要通过routingkey描述消息的“特征”。

[RabbitMQ下的生产消费者模式与订阅发布模式](https://blog.csdn.net/zwgdft/article/details/53561277)

[Kafka下的生产消费者模式与订阅发布模式](https://blog.csdn.net/zwgdft/article/details/54633105)

### rabbitmq的发布订阅模式

1. Exchange、Queue与Routing Key三个概念是理解RabbitMQ消息投递的关键。
2. RabbitMQ中一个核心的原则是，消息不能直接投递到Queue中。
3. Producer只能将自己的消息投递到Exchange中，由Exchange按照routing_key投递到对应的Queue中
4. 在Consumer Worker中，声明自己对哪个Exchange感兴趣，并将自己的Queue绑定到自己感兴趣的一组routing_key上，建立相应的映射关系；
5. 在Producer中，将消息投递一个Exchange中，并指明它的routing_key。

![](/public/upload/architecture/rabbitmq_subscribe_publish.png)

Queue这个概念只是对Consumer可见，Producer并不关心消息被投递到哪个Queue中

![](/public/upload/architecture/rabbitmq_subscribe_publish_2.png)

只保留逻辑概念的话

![](/public/upload/architecture/rabbitmq_subscribe_publish_3.png)

### kafka的发布订阅模型

![](/public/upload/architecture/kafka_subscribe_publish.png)

kafka 中partition 是物理上的概念，仅从逻辑概念上看

![](/public/upload/architecture/kafka_subscribe_publish_3.png)

所有接入topic 的consumer group 都会收到消息，producer 没有类似routingkey 可以进行topic 内部 consumer group的指定，因此Kafka只提供了广播和单播（一对一广播）的消息模型，不支持组播（消息只发给topic 内的特定n个consumer group）。因此对于复杂场景，一般rabbitmq 项目只需要一个exchange即可，而kafka 通常要多个topic，**所以kafka topic 其实跟 rabbitmq的 routingkey 逻辑上作用更像**。

### 发布订阅模型小结

1. 两者都包含 物理概念和逻辑概念
2. producer 只负责向逻辑概念发布数据
3. consumer 一般与物理概念紧密关联，并绑定相关的逻辑概念（exchange + routingkey/topic，也就是订阅自己感兴趣的“事件”）

### 消息中间件的选型

笔者一度很迷kafka，并打算用其替换项目中的rabbitmq，但实际上kafka 不是银弹，比如其不支持优先级队列，而这个feature 在某些业务场景下很重要。

选型消息中间件 应该注意哪些问题 可以参见[消息中间件选型分析——从Kafka与RabbitMQ的对比来看全局](http://blog.didispace.com/%E6%B6%88%E6%81%AF%E4%B8%AD%E9%97%B4%E4%BB%B6%E9%80%89%E5%9E%8B%E5%88%86%E6%9E%90/)

![](/public/upload/architecture/queue_middleware_choose.png)

消息中间件犹如小马过河，选择合适的才最重要，这需要贴合自身的业务需求，技术服务于业务。**RabbitMQ在于routing，而Kafka在于streaming**，了解其根本对于自己能够对症下药选择到合适的消息中间件尤为重要。从功能维度上来说，RabbitMQ的优势要大于Kafka，但是Kafka的吞吐量要比RabbitMQ高出1至2个数量级，一般RabbitMQ的单机QPS在万级别之内，而Kafka的单机QPS可以维持在十万级别，甚至可以达到百万级。

消息中间件选型切忌一味的追求性能或者功能，性能可以优化，功能可以二次开发。**如果要在功能和性能方面做一个抉择的话，那么首选性能，因为总体上来说性能优化的空间没有功能扩展的空间大**。然而对于长期发展而言，生态又比性能以及功能都要重要。

## 消息队列监控

### rabbitmq

channel 接口本身 也提供

	// 队列未消费的消息数
    long messageCount(String queue) throws IOException;
    // 队列消费者个数
    long consumerCount(String queue) throws IOException;

RabbitMQ 有灵活的插件机制，启用 [rabbitmq-management](http://www.rabbitmq.com/management.html) 就可以对服务进行监控和管理。`http://rabbitmq_host:15672/api/` 展示了rabbitmq 的http api 列表，基本具备所有 Rabbitmq java client  功能。http api的优势在于，可以在队列的消费实例之外构建专门的监控系统。

1. 所有的请求需要授权，请求的返回数据都是json
2. 很多请求返回的都是一个列表，你可以在请求中 加入sort 和 sort_reverse 为返回数据排序
3. 默认vhost 使用%2f 代替

`curl -i -u guest:guest http://ip:port/api/queues/%2f/queue_name` 返回值 json 

	{
		...
		"consumers":xx
		"state":xx,
		"vhost":xx,
		"durable":xx,
		"auto_delete":xx
		"messages_unacknowledged":xx,
		"messages_unacknowledged_details":{
			"rate":xx
		},
		"messages_ready":xx,
		"messages_ready_details":{
			"rate":xx
		},
		"messages":xx,
		"messages_details":{
			"rate":xx
		}
		...
	}
	
据此，就可以对队列拥堵，consumer 实例是否正常 来做出预警。
    
    
## 其它

消息队列系统有很多，主要有以下不同：

1. 消息队列存在哪？内存、磁盘？
2. 是否支持分布式
3. 消息的格式是什么？
4. 系统是否能够确保：消息一定会被接收，消息未被正确处理时通知发送方。


至于kafka、flume和rabbitmq等系统，网上资料很多，此处不再赘述。


## 引用

[Kafka快速入门](http://colobu.com/2014/08/06/kafka-quickstart/)

[RabbitMQ AMQP 消息模型攻略](https://segmentfault.com/a/1190000007123977)

