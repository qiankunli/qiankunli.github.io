---

layout: post
title: Thrift基本原理与实践（二）
category: 技术
tags: Java
keywords: thrift,batch,async

---

## 简介

重复下上节的thrift示例

    namespace java org.lqk.thrift
    service RemotePingService{
        void ping(1: i32 length)
    }

- RemotePingService.Iface			void ping(int length)
- RemotePingService.syncIface	    void ping(int length, org.apache.thrift.async.AsyncMethodCallback resultHandler)

thrift提供了RemotePingService.Client和RemotePingService.AsyncClient实现了接口和方法，方法的实际逻辑就是发送和接收数据。


thrfit调用客户端例子

    TTransport transport = new TSocket("localhost", 1234);
    TProtocol protocol = new TBinaryProtocol(transport);
    RemotePingService.Client client = new RemotePingService.Client(protocol);
    transport.open();
    client.ping(2012);
    transport.close();
    
熟悉spring的人看到这段代码，应该会想到有办法简化它，因为这段代码就做了一件事`client.ping(2012)`，而client的初始化和销毁工作可以交给spring。最终的效果是

    @Autowired
    private RemotePingservice.Iface pingService;
    public void test(){
        pingService.ping(2012);
    }
    
## spring化 + 代理模式 + 连接池

    FactoryBean{
        ClientConfig
        ConnectionConfig
        ConnectionPoolConfig
    }

`TTransport transport = new TSocket("localhost", 1234);`使用common-pool2对TTransport进行池化。

代理模式，我们一般要搞清楚在哪个方法上加代理，在这个方法的前后要干点什么。此处，**我们要代理的便是RemotePingService.Client和RemotePingService.AsyncClient的ping方法**（这是切入点），它们已经实现了发送和接收数据的逻辑，代理方法中，我们要在ping方法之前做的工作是通过一系列xxConfig得到Client和AsyncClient。以对同步方法的代理为例，伪代码如下：

静态代理

    ping_proxy(){
        RemotePingService.Client client= XXBuilder.build(ClientConfig,ConnectionConfig,ConnectionPoolConfig);
        client.ping();
    }

动态代理

    ProxyHandler implements InvocationHandler{
         ClientConfig
         ConnectionConfig
         ConnectionPoolConfig
         public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            RemotePingService.Client client= XXBuilder.build(ClientConfig,ConnectionConfig,ConnectionPoolConfig);
            Object result = method.invoke(client, args);
            return result;
         }
    }

异步方法相对同步方法的特别之处在于，异步方法还要传入一个callback。我们可以定制这个callback以达到一些目的。

## 对异步调用的代理和改造

同步和异步的底层实现一般不太一样。

同步的话，`void ping(ing length){send_command(length);receive_command()}`业务线程直接操刀数据的发送与回收。

而对于异步通信，因为nio的使用，通常由io线程(eventloop)而不是业务线程来监听响应数据，io线程与业务线程的交互有以下几种形式；

1. 传入callback，比如thrift异步调用会传入一个AsyncMethodCallback。io线程拿到响应数据后，执行`AsyncMethodCallback.onComplete()`。
2. Future。业务线程执行调用时（也就是发送请求数据），先创建一个future并返回。io线程拿到响应后设置Future，业务线程从Future中拿到响应。

callback和future本质都是一样的，都是业务线程先创建好并传给io线程，io线程拿到数据后，由io线程处理设置future的结果 or 执行`callback.onComplete`。

- future提供设置和获取结果的api
- callback提供onComplete方法，想对结果干什么事直接干就是了，但无法返回结果。

其中，对于Future模式，有以下优点（也是我们为什么要对异步调用进行改造的原因）：

1. **在调用形式上，跟同步模式是一样的。**
2. 我们可以学习redis的pipeline对齐进一步封装（或者说是java的ExecutorService）：提供批量执行接口，异步执行多条命令，对拿到的Future集中处理。好比`Future aynscMethod();list<Future> = for(){}`。

thrift原生的异步调用是

    RemotePingService.AsyncClient asyncClient = new RemotePingService.AsyncClient(protocol, clientManager, transport);
    asyncClient.ping(111, new AsyncMethodCallback(){
        public void onComplete(Object response) {}
        public void onError(Exception exception) {}
    });

thrift原生提供的不是Future模式，是callback模式，这就需要我们将callback转化成Future。如果说以前`callback.onComplete`的目的是对结果的处理的话，现在也是，只是重点变成了：实际的ping执行完毕后，将结果作为ping_proxy的返回值。

转化的关键就是：将callback变成一个Future，增加一个result成员存储结果，callback的onComplete就是对结果的设置，再增加一些锁的工具类用于安全的获取result。

    CustomerAsyncCallback{
        Result result;
        CountDown latch;
        onComploete(Result result){
            set(result)
            latch.countDown(); // 手动标记结果可用
        }
    }
    // 静态代理
    Result ping_proxy(){
        RemotePingService.AsyncClient asyncClient = xx;
        CustomerAsyncCallback callback = xx;    // 初始化callback
        ping(length,callback.onComplete(Result result){
            set(result)
            latch.countDown();     
        });                                
        latch.await()
        return callback.getResult()            
    }
    
## 基于异步特性实现批量处理   

为什么要异步调用批量化？因为这样做可以在一个线程中，一段串行代码里，在很短的时间内发出多个调用（调用之间没有先后关系），调用的总时长取决于最慢的那个调用而不是所有调用的耗费时间之和。

批量化封装，我们的目的是提供类似下列代码的接口

    List<Object> results = batch.execute(new xxx{
        RemotePingService.Iface.ping(length);   
        RemotePingService.Iface.ping(length);
        RemotePingService.Iface.ping(lenght);
    });
    
1. 以同步接口调用，但实际执行的是异步方法
2. 获取异步调用的返回值，并集中返回。
    
每一个`RemotePingService.Iface.ping(length)`实际逻辑就是：**创建相应的AsyncClient实例**，传入自定义callback，执行实际的ping方法，执行过程中，callback会拿到result值。

那么如何将每个callback拿到的result放到一起呢？如果` RemotePingService.Iface.ping(lenght)`能传入一个List参数就好了。

既然不能传，那么熟悉threadlocal的童鞋知道，threadlocal可以实现无参数传参的效果。

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

## 小结

基于原生的thrift调用，我们逐步讨论了以下扩展：

1. spring化 + 代理模式 + 连接池封装
2. 异步调用的改造
3. 异步调用的批量化封装

大大简化和丰富了thrift的使用。

