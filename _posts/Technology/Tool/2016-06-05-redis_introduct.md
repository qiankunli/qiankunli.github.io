---

layout: post
title: Redis概述
category: 技术
tags: Tool
keywords: redis

---

## 简介(未完成)

很多事情联系起来想很有意思，比如rpc(跨主机进程通信)，然后一些大牛搞出redis，可以理解为**跨主机访问内存**，360推出一个pika，可以理解为跨主机访问磁盘（支持redis协议）。

跨主机通信，当然免不了网络通信的一些约定，这不是本文的重点，所以不多谈。不管跨主机访问内存还是磁盘，都不是提供一个byte[]让客户端随便用，而是像rpc一样，传输一些约定好的数据结构。区别是，rpc传输的数据结构描述了调用信息，redis的客户端与服务端传输的数据结构是为了存储和查询。

把一些数据结构存在本机或存在远程主机，有一些隐含的意味：

1. "本机的"数据结构包括：基本数据类型，复合类型（string，list，map等）。基本数据类型往往用不着跨主机存储，因为不值当。
2. 对于本地访问内存而言，访问一个数据结构要指明两个要素：内存地址和类型。内存地址说明去哪取数据，类型说明取多少数据，取出的数据如何处理。远端访问内存类似，只不过”地址“不再是一个内存地址，而是一个具备唯一性的key，由远端主机完成key到该主机的内存地址的映射。

上述逻辑或许能够解释，很多类似redis的工具为什么是key-value的，并且value可以是各种数据结构。

## redis概述

redis分为server端和client端，主要分为socket处理和业务处理（对于redis而言业务处理就是，命令的解析，数据的存取）。

server端以字典（dict，参见redis对dict数据结构的实现）存储所有的key-value。为了客户端划分业务的需要，key-value被划分到不同的redisDb中。


redis的服务器进程就是一个事件循环(loop)，这个循环中的文件事件负责接收客户端的命令请求，以及向客户端发送命令回复，而时间事件则负责执行像serverCron函数这样需要定时运行的函数(来自《Redis设计与实现》)。**简直跟netty的Eventloop一个样**：

1. 文件事件负责socket操作，基于Reactor模式。
2. 时间事件负责socket之外的操作，主要是定时操作。
3. 文件事件和时间事件在一个线程中执行

        def main(){
			init_server();
            while (server_is_not_shutdown{
            	aeProcessEvents(){
                	获取离当前时间最近的时间事件，根据算法计算一个值timeVal
                    阻塞等待文件事件（timeout为timeVal）。即下一个时间事件快了就少等会儿，否则就多等会儿。
                    处理文件事件（如果有的话）
                    处理时间事件
                }
            }
            clean_server();

        }
        
从socket处理的角度讲，redis的客户端和服务端实现与其它CS工具并无不同，都是基于Reactor模式，基于自定义的一套协议。

## 集群与Sharding

Redis算是一种内存数据库，即便它支持持久化特性，将所有数据放在一个Redis主机上也是不科学的，更何况对于数据量较大的场景，一个Redis主机根本存不下，所以要做Sharding。有以下几种方案

1. 服务器端Sharding，服务端Redis集群拓扑结构变化时，客户端不需要感知，客户端像使用单Redis服务器一样使用Redis集群
2. 客户端Sharding，客户端sharding技术其优势在于服务端的Redis实例彼此独立，相互无关联
3. sharding中间件，结合了前两种优势，规避了前两种缺点，不过客户端到服务端中间多一个到中间件的通信过程。


## 对redis操作的再封装

先回顾下数据库访问的情况。web项目数据库的访问最后基本稳定成了“controller，service，dao”三级模式。service封装了数据访问的复杂性（比如一个service方法中有多个dao操作），controller一般不直接操作dao来使用数据。数据操作的动作也比较稳定：CRUD。

有了redis之后，算是可以“跨主机访问内存”了。我们往往将redis操作与db操作结合起来，其基本动作是：“先取redis，没有则查db并加入到redis”，如果把直接“redisTemplate.set(xx,xx)”比作redis的“dao”的话，那么我们也需要一个类似“service”的包装，来封装对redis和db的复合操作。并在这个redis的“service”操作中解决以下问题：

1. 请求合并。对于同一个key，a线程在cache中没找到，发起了数据库查询操作。那么b线程在这个过程中，应该等待a线程完成数据库查询，在开始对cache和db的查询。
2. 对redis做shard
3. 当某个shard新增和移除时，能够实现cache数据的均衡分布（这个需求有时候没有必要）。