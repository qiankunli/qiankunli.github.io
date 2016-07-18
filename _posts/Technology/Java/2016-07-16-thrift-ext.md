---

layout: post
title: Thrift基本原理与实践（二）
category: 技术
tags: Java
keywords: thrift,batch,async

---

## 简介

thrfit调用客户端例子

    TTransport transport = new TSocket("localhost", 1234);
    TProtocol protocol = new TBinaryProtocol(transport);
    RemotePingService.Client client = new RemotePingService.Client(protocol);
    transport.open();
    client.ping(2012);
    transport.close();
    
熟悉spring的人看到这段代码，应该会想到有办法简化它，因为这段代码就做了一件事`client.ping(2012)`，而client的初始化（比如通过一个FactoryBean来创建client）和销毁工作可以交给spring。最终的效果是

    @Autowired
    private RemotePingservice.Iface pingService;
    public void test(){
        pingService.ping(2012);
    }
    
除了spring化，我们还可以做以下几项工作。

    
## 使用连接池

`TTransport transport = new TSocket("localhost", 1234);`使用common-pool2对TTransport进行池化。

## 基于异步特性实现批量处理

同步和异步的底层实现一般不太一样。

同步的话，`void ping(ing length){send_command(length);receive_command()}`业务线程直接操刀数据的发送与回收。

而对于异步通信，通常由io线程(eventloop)而不是业务线程来监听响应数据，io线程与业务线程的交互有以下几种形式；

1. 传入Listener（比如thrift异步调用会传入一个resultHandler）。io线程拿到响应数据后，执行listener。
2. Future。io线程拿到响应后设置Future，业务线程从Future中拿到响应。

其中，对于Future模式，我们可以学习redis的pipeline对齐进一步封装：提供批量执行接口，异步执行多条命令，集中处理返回结果。


## 一个端口监听多个thrift服务

