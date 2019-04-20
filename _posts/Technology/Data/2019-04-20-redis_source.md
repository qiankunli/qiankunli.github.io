---

layout: post
title: Redis源码分析
category: 技术
tags: Data
keywords: Redis

---

## 前言（持续更新）

建议看下前文 [Redis 学习](http://redisdoc.com/topic/protocol.html)

参考[《Apache Kafka源码分析》——server](http://qiankunli.github.io/2019/01/30/kafka_learn_2.html)服务端网络开发的基本套路

![](/public/upload/architecture/network_communication.png)

### `set msg 'hello world'` 发生了什么

类图和序列图

1. 服务端启动流程
2. 一次操作流程

![](/public/upload/data/redis_class_diagram.png)


## 启动过程

redis.c

	int main(int argc, char **argv) {
		...
		// 初始化服务器
		initServerConfig();
		...
		// 将服务器设置为守护进程
		if (server.daemonize) daemonize();
		// 创建并初始化服务器数据结构
		initServer();
		...
		// 运行事件处理器，一直到服务器关闭为止
		aeSetBeforeSleepProc(server.el,beforeSleep);
		aeMain(server.el);
		// 服务器关闭，停止事件循环
		aeDeleteEventLoop(server.el);
		return 0
	}

Redis的网络监听没有采用libevent等，而是自己实现了一套简单的机遇event驱动的API，具体见ae.c。事件处理器的主循环 

	void aeMain(aeEventLoop *eventLoop) {
		eventLoop->stop = 0;
		while (!eventLoop->stop) {
			// 如果有需要在事件处理前执行的函数，那么运行它
			if (eventLoop->beforesleep != NULL)
				eventLoop->beforesleep(eventLoop);
			// 开始处理事件
			aeProcessEvents(eventLoop, AE_ALL_EVENTS);
		}
	}

[Redis 中的事件循环](https://draveness.me/redis-eventloop)

## Sentinel(哨兵模式)

sentinel是redis高可用的解决方案，sentinel系统可以监视一个或者多个redis master服务，以及这些master服务的所有从服务；当某个master服务下线时，自动将该master下的某个从服务升级为master服务替代已下线的master服务继续处理请求。

