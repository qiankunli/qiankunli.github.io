---

layout: post
title: Thrift基本原理
category: 技术
tags: Java
keywords: future

---

## 简介

thrfit调用客户端例子

    TTransport transport = new TSocket("localhost", 1234);
    TProtocol protocol = new TBinaryProtocol(transport);
    RemotePingService.Client client = new RemotePingService.Client(protocol);
    transport.open();
    client.ping(2012);
    transport.close();
    
熟悉spring的人看到这段代码，应该会想到有办法简化它，因为这段代码就做了一件事`client.ping(2012)`，而client的初始化（比如通过一个FactoryBean）和销毁工作可以交给spring。最终的效果是

    @Autowired
    private RemotePingservice.Iface pingService;
    public void test(){
        pingService.ping(2012);
    }
    
## 连接，使用连接池