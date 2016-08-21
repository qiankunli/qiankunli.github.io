---

layout: post
title: Thrift基本原理与实践（三）
category: 技术
tags: Java
keywords: thrift,service discovery

---

## 简介


## 数据流向跟引用依赖

数据从客户端到server端的流向：

client : client ==> protocol ==> transport
server : Transport ==> protocol ==> processor ==> Fuction

观察最简单的thrift客户端和服务端代码：

    TTransport transport = new TSocket("localhost", 1234);
    TProtocol protocol = new TBinaryProtocol(transport);
    RemotePingService.Client client = new RemotePingService.Client(protocol);
    transport.open();
    client.ping(2012);
    transport.close();

client其实就是iface在客户端的实现

    TServerSocket serverTransport = new TServerSocket(1234);
    Test.Processor processor = new Processor(new TestImpl());
    Factory protocolFactory = new TBinaryProtocol.Factory(true, true);
    Args args = new Args(serverTransport);
    args.processor(processor);
    args.protocolFactory(protocolFactory);
    TServer server = new TThreadPoolServer(args);
    server.serve();
    
    


客户端是，Client对象（client对象就是客户端的ifaceimpl）依赖protocol。protocol依赖transport。因为是protocol的sendMessage驱动transport的senddata(byte[]).


服务端，是先接到byte[]，然后变成message，然后执行processor（包装服务端ifaceimpl）


## 动态发现

说动态发现之前，先说下扩展代码的事。**增加一个feature，我们要将feature增加在哪里，如何增加**

如果你抽象除了一个Server，可以open、close。那么，你想把这个server注册到zk上时，肯定不是写在`Server.open()`里。而是定义一个更高层级的抽象，比如Exportor。

    Exporter{
        open(){
            zk.xxx
            server.open();
        }
    }
    
mainstay 提供各种api的初衷，估计是想各种实现，比如rpc是thrift，config center是zk

1. 到时候各个组件可以替换
2. 可以集中，比如rpc使用thrift和rmi，都用了zk，那么mainstay是可以集中管理的。