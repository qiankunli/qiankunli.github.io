---

layout: post
title: Spring下的设计模式
category: 技术
tags: Java
keywords: proxyFactoryBean

---

## 简介

现在一个项目很少说不用spring的，spring的应用也使得程序的写法跟传统程序有了很大的不同。

1. 使用pojo即可完成工作，简化了类的设计。

    - 依赖类的注入使得依赖对象可以直接使用，省去了依赖类的创建工作。
    - 类之间的继承和组合关系非常简单

2. 简化了配置的读取

    照以前，读取配置信息的常见步骤：使用Properties类读取配置，并加载到一个xxConstant类中。
    
除了类的使用外，对框架的使用也发生了很大的改变。

## spring与其它框架的整合

1. 框架本身只是让spring ioc构造了外部使用所必须的类。比如spring与线程池的结合

        <bean id="taskExecutor"
        class="org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor">
            <property name="corePoolSize" value="5" />  
            <property name="maxPoolSize" value="10" />  
            <property name="WaitForTasksToCompleteOnShutdown" value="true" />  
        </bean>  
    
        public class Test{
            @Autowire
            private  ThreadPoolTaskExecutor taskExecutor;
            public void run(){
                // 使用taskExecutor
            }
        }

2. 框架本身具备完整的运行流程。比如quartz，本身具备独立的线程池和任务触发机制。还有一个start和shutdown的“动作”。此时，spring不仅像第一种方式一样，提供对quartz一些接口类的使用，还整合了quartz的Scheduler的生命周期。

## spring与远程调用框架的结合

为什么要把这个单独拿出来说呢？因为如今已经不是单台服务器打天下的时候了，通过网路，扩展计算能力，实现分布式计算变得越来越重要。

远端调用本身有独立的框架，比如HttpClient（基于http），Thrift，RMI，RabbitMQ等，每个框架有自己独立的客户端和服务端，不用spring也工作的很好。但基于spring平台的整合，spring对这些框架的客户端进行了进一步封装（**这种封装的模式基本上是固定的，非常值得自己在实战中应用**），能够大大简化业务逻辑的编写。比如，Spring HTTP调用器的实现，就为我们提供了以下抽象：

    <bean id="proxy" class="org.springframework.remoting.httpinvoker.HttpInvokerProxyFactoryBean">
        <property name="serviceUrl" value="http://host:port/xx"/>
        <property name="serviceInterface" value="your.interface"/>
    </bean>
    <bean id="yourBean" class="yourClass">
        <property name="remoteService">
            <ref bean="proxy"/>
        </property>
    </bean>
    
通过一个ProxyFactoryBean，即可返回一个代理实例，在该实例中，会做好“请求封装，发送请求，接收响应，解析响应”等工作，使远程调用过程透明。更进一步，这个封装的过程可以做的更复杂、更强大：

1. 加入连接池，提高远程访问的效率
2. 加入Zookeeper，服务端定期向zookeeper注册自己的状态。一旦状态发生变更，即由zookeeper通知客户端做出反应。
3. 为请求做负载均衡，请求被转向不同的后端处理。

## 小结

Spring的两个基本特性，ioc管理pojo对象以及它们之间的耦合关系。以前java对象随意存储在堆中，有了spring之后，还是如此，但有一个容器（说白了是个map）来保有对象的引用。另一方面，通过aop以动态和非侵入的方式来增强服务的功能，原先，增强一个方法的功能要改变原有的代码，现在新增的功能可以单独编写，然后织入（甚至是替换）。这完全改变了我们编写代码的方式，最终的目的就是：简化业务模型，以pojo类为基础实现项目。

## 引用

[Spring AOP源码分析（七）ProxyFactoryBean介绍][]

[Spring AOP源码分析（七）ProxyFactoryBean介绍]: http://m.oschina.net/blog/376308