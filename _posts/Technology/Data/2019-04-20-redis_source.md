---

layout: post
title: Redis源码分析
category: 技术
tags: Data
keywords: Redis

---

## 前言（持续更新）

* TOC
{:toc}

建议看下前文 [Redis 学习](http://redisdoc.com/topic/protocol.html)

参考[《Apache Kafka源码分析》——server](http://qiankunli.github.io/2019/01/30/kafka_learn_2.html)服务端网络开发的基本套路

![](/public/upload/architecture/network_communication.png)

源码来自[带有详细注释的 Redis 3.0 代码（annotated Redis 3.0 source code）](https://github.com/huangz1990/redis-3.0-annotated)

## 宏观梳理

**序列图待补充，应一直通到内存操作**

![](/public/upload/data/redis_sequence_diagram.png)

整个轴线是redisServer 初始化并启动eventloop， eventLoop 创建redisClient 及processCommand 方法进而 执行redisCommand 向 dict 中保存数据

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



## 网络层

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

![](/public/upload/data/redis_eventloop_proces_event.png)

	int aeProcessEvents(aeEventLoop *eventLoop, int flags)
	{
		struct timeval tv, *tvp;
		... 
        // 获取最近的时间事件
        if (flags & AE_TIME_EVENTS && !(flags & AE_DONT_WAIT))
            shortest = aeSearchNearestTimer(eventLoop);
        if (shortest) {
            // 如果时间事件存在的话，那么根据最近可执行时间事件和现在时间的时间差来决定文件事件的阻塞时间
            // 计算距今最近的时间事件还要多久才能达到，并将该时间距保存在 tv 结构中
            aeGetTime(&now_sec, &now_ms);
        } else {
            // 执行到这一步，说明没有时间事件，那么根据 AE_DONT_WAIT 是否设置来决定是否阻塞，以及阻塞的时间长度
        }
        // 处理文件事件，阻塞时间由 tvp 决定
		// 类似于 java nio 中的select
        numevents = aeApiPoll(eventLoop, tvp);
        for (j = 0; j < numevents; j++) {
            // 从已就绪数组中获取事件
            aeFileEvent *fe = &eventLoop->events[eventLoop->fired[j].fd];
            int mask = eventLoop->fired[j].mask;
            int fd = eventLoop->fired[j].fd;
            // 读事件
            if (fe->mask & mask & AE_READABLE) {
                fe->rfileProc(eventLoop,fd,fe->clientData,mask);
            }
            // 写事件
            if (fe->mask & mask & AE_WRITABLE) {
                if (!rfired || fe->wfileProc != fe->rfileProc)
                    fe->wfileProc(eventLoop,fd,fe->clientData,mask);
            }
        }
		// 执行时间事件
		if (flags & AE_TIME_EVENTS)
			processed += processTimeEvents(eventLoop);
	}

这个event loop的逻辑可不孤单，netty中也有类似的[EventLoop 中的 Loop 到底是什么？](http://qiankunli.github.io/2017/04/14/network_channel.html)

Redis 中会处理两种事件：时间事件和文件事件。在每个事件循环中 Redis 都会先处理文件事件，然后再处理时间事件直到整个循环停止。 aeApiPoll 可看做文件事件的生产者（还有一部分文件事件来自accept等），processEvents 和 processTimeEvents 作为 Redis 中发生事件的消费者，每次都会从“事件池”（aeEventLoop的几个列表字段）中拉去待处理的事件进行消费。

## 协议层

我们以读事件为例，但发现数据可读时，执行了` fe->rfileProc(eventLoop,fd,fe->clientData,mask);`，那么rfileProc 的执行逻辑是啥呢？

1. initServer ==> aeCreateFileEvent. 初始化server 时，创建aeCreateFileEvent（aeFileEvent的一种），当accept （可读事件的一种）就绪时，触发aeCreateFileEvent->rfileProc 方法 也就是  acceptTcpHandler

		// redis.c 
		void initServer() {
			...
			// 为 TCP 连接关联连接应答（accept）处理器，用于接受并应答客户端的 connect() 调用
    		for (j = 0; j < server.ipfd_count; j++) {
        		if (aeCreateFileEvent(server.el, server.ipfd[j], AE_READABLE,acceptTcpHandler,NULL) == AE_ERR){...}
    		}
			...
		}

2. 创建客户端，并绑定读事件到loop：acceptTcpHandler ==> createClient ==> aeCreateFileEvent ==> readQueryFromClient

		void acceptTcpHandler(aeEventLoop *el, int fd, void *privdata, int mask) {
    int cport, cfd, max = MAX_ACCEPTS_PER_CALL;
		...
			while(max--) {
				// accept 客户端连接
				cfd = anetTcpAccept(server.neterr, fd, cip, sizeof(cip), &cport);
				if (cfd == ANET_ERR) {
					...
					return;
				}
				// 为客户端创建客户端状态（redisClient）
				acceptCommonHandler(cfd,0);
			}
		}
		static void acceptCommonHandler(int fd, int flags) {
			// 创建客户端
			redisClient *c;
			if ((c = createClient(fd)) == NULL) {
				...
				close(fd); /* May be already closed, just ignore errors */
				return;
			}
			// 如果新添加的客户端令服务器的最大客户端数量达到了，那么向新客户端写入错误信息，并关闭新客户端
			// 先创建客户端，再进行数量检查是为了方便地进行错误信息写入
			...
		}
		redisClient *createClient(int fd) {
			// 分配空间
			redisClient *c = zmalloc(sizeof(redisClient));
			if (fd != -1) {
				...
				//绑定读事件到事件 loop （开始接收命令请求）
				if (aeCreateFileEvent(server.el,fd,AE_READABLE,
					readQueryFromClient, c) == AE_ERR){
					// 清理/关闭资源退出
				}
			}
			// 初始化redisClient其它数据
		}

3. 拼接和分发命令数据 readQueryFromClient ==> processInputBuffer ==> processCommand

		networking.c
		void readQueryFromClient(aeEventLoop *el, int fd, void *privdata, int mask) {
			redisClient *c = (redisClient*) privdata;
			// 获取查询缓冲区当前内容的长度
			// 如果读取出现 short read ，那么可能会有内容滞留在读取缓冲区里面
			// 这些滞留内容也许不能完整构成一个符合协议的命令，
			qblen = sdslen(c->querybuf);
			// 如果有需要，更新缓冲区内容长度的峰值（peak）
			if (c->querybuf_peak < qblen) c->querybuf_peak = qblen;
			// 为查询缓冲区分配空间
			c->querybuf = sdsMakeRoomFor(c->querybuf, readlen);
			// 读入内容到查询缓存
			nread = read(fd, c->querybuf+qblen, readlen);
			// 读入出错
			// 遇到 EOF
			if (nread) {
				// 根据内容，更新查询缓冲区（SDS） free 和 len 属性
				// 并将 '\0' 正确地放到内容的最后
				sdsIncrLen(c->querybuf,nread);
				// 记录服务器和客户端最后一次互动的时间
				c->lastinteraction = server.unixtime;
				// 如果客户端是 master 的话，更新它的复制偏移量
				if (c->flags & REDIS_MASTER) c->reploff += nread;
			} else {
				// 在 nread == -1 且 errno == EAGAIN 时运行
				server.current_client = NULL;
				return;
			}
			// 查询缓冲区长度超出服务器最大缓冲区长度
			// 清空缓冲区并释放客户端
			// 从查询缓存重读取内容，创建参数，并执行命令
			// 函数会执行到缓存中的所有内容都被处理完为止
			processInputBuffer(c);
			server.current_client = NULL;
		}
		redis.c
		processCommand（待充实）

### 业务层


## Sentinel(哨兵模式)

sentinel是redis高可用的解决方案，sentinel系统可以监视一个或者多个redis master服务，以及这些master服务的所有从服务；当某个master服务下线时，自动将该master下的某个从服务升级为master服务替代已下线的master服务继续处理请求。

