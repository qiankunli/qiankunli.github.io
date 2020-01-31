---

layout: post
title: java系并发的发展
category: 技术
tags: Java
keywords: Java  

---

## 前言

笔者到目前学习过scala、java、go，它们在并发程序的实现上模型不同，汇总一下是个蛮有意思的事情。

## 线程操作

1. 创建它：继承Thread，实现Runnable，实现TimerTask（现在不推荐）
2. 启动它：start
3. 暂停它（唤醒它）：sleep（自动唤醒），wait和notify
4. 停止它（取消它）：
    
    a. interrupt，注意这种停止并不是抢占式的，代码中要遵守一定的约定。 [java exception](http://qiankunli.github.io/2017/04/22/exception.html)
    
    b. 设置一个变量（显式的interrupt）
        
        class thread{
            public boolean isRun = "true";
            void run(){
                 while(isRun){
                     xx
                 }
            }
            void stop(){
                isRun = false;
            }
        }

## Java/Jvm 内部工作线程

看看C语言下写多线程程序什么感觉


        #include <stddef.h>
        #include <stdio.h>
        #include <unistd.h>
        #include <pthread.h>		//用到了pthread库
        #include <string.h>
        void print_msg(char *ptr);
        int main(){
            pthread_t thread1, thread2;
            int i,j;
            char *msg1="do sth1\n";
            char *msg2="do sth2\n";
            pthread_create(&thread1,NULL, (void *)(&print_msg), (void *)msg1);
            pthread_create(&thread2,NULL, (void *)(&print_msg), (void *)msg2);
            sleep(1);
            return 0;
        }
        void  print_msg(char *ptr){
            int retval;
            int id=pthread_self();
            printf("Thread ID: %x\n",id);
            printf("%s",ptr);
            // 一个运行中的线程可以调用 pthread_exit 退出线程, 参数表示线程的返回值
            pthread_exit(&retval);
        }

pthread_create 四个参数

1. 线程对象
2. 线程的属性，比如线程栈大小
3. 线程运行函数
4. 线程运行函数的参数
    
从c语言中线程的代码实例和操作系统的基本原理（进程通常是执行一个命令，或者是fork），我们可以看到，线程可以简单的认为是在并发执行一个函数（pthread_create类似于go 代码中常见的`go function(){xxx}`）。

[java 内部工作线程介绍](http://java-boy.iteye.com/blog/464953)哪怕仅仅 简单的跑一个hello world ，java 进程也会创建如下线程

	"Low Memory Detector" 
	"CompilerThread0"
	"Signal Dispatcher"
	"Finalizer"
	"Reference Handler"
	"main" 
	"VM Thread"
	"VM Periodic Task Thread"


笔者有一次，试验一个小程序，main 函数运行完毕后，idea 显示 java 进程并没有退出，笔者还以为是出了什么bug。thread dump之后，发现一个thread pool线程在waiting，才反应过来是因为thread pool 没有shutdown。进而[Java中的main线程是不是最后一个退出的线程](https://blog.csdn.net/anhuidelinger/article/details/10414829)

1. JVM会在所有的非守护线程（用户线程）执行完毕后退出；
2. main线程是用户线程；
3. 仅有main线程一个用户线程执行完毕，不能决定JVM是否退出，也即是说main线程并不一定是最后一个退出的线程。

这也是为什么thread pool 若没有shutdown，则java 进程不会退出的原因。

## java并发的发展历程

1. 使用原始的synchronized关键字，wait和notify等方法，实现锁和同步。

2. jkd1.5和jdk1.6提供了concurrent包，包含Executor，高效和并发的数据容器，原子变量和多种锁。更多的封装减少了程序员自己动手写并发程序的场景，并提供lock和Condition对象的来替换替换内置锁和内置队列。

3. jdk1.7提供ForkJoinTask支持，还未详细了解，估计类似于MapReduce，其本身就是立足于编写可并行执行程序的。

通过阅读《java并发编程实战》全书的脉络如下

1. 什么是线程安全，什么导致了线程不安全？
2. 如何并行程序串行化，常见的并行化程序结构是什么？Executor，生产者消费者模式
3. 如何构造一个线程安全的类（提高竞争效率），如何构造一个依赖状态的类（提高同步效率）？提高性能的手段有哪些？ 使用现有工具类 or 扩充已有父类？

性能优化的基本点就是：减少上下文切换和线程调度（挂起与唤醒）操作。从慢到快的性能对比：

1. synchronized操作内置锁，wait和notify操作内置队列。考虑到现在JVM对其实现进行了很大的优化，其实性能也还好。
2. AQS及AQS包装类
3. Lock和Condition（如果业务需要多个等待线程队列的话）

从上到下，jvm为我们做的越少，灵活性越高，更多的问题要调用者自己写在代码里（执行代码当然比劳烦jvm和os效率高很多），使用的复杂性越高。

## 理念变化

[应用 fork-join 框架](https://www.ibm.com/developerworks/cn/java/j-jtp11137.html) 基本要点：

1. 硬件趋势驱动编程语言，一个时代的主流硬件平台形成了我们创建语言、库和框架的方法，语言、库和框架形成了我们编写程序的方式。


	||语言|类库|硬件的并行性越来越高|
	|---|---|---|---|
	|| synchronized、volatile | Thread |大部分是单核，线程更多用来异步|
	|java1.5/1.6|  | java.util.concurrent 包 |多核，适合**粗粒度**的程序，比如web服务器、数据库服务器的多个独立工作单元|
	|java1.7|  | fork-join |多核、每核多逻辑核心，**细粒度**的并行逻辑，比如分段遍历集合|

2. 将一个任务分解为可并行执行的多个任务，Divide and conquer

		Result solve(Problem problem) { 
		    if (problem.size < SEQUENTIAL_THRESHOLD)
		        return solveSequentially(problem);
		    else {
		        Result left, right;
		        INVOKE-IN-PARALLEL { 
		            left = solve(extractLeftHalf(problem));
		            right = solve(extractRightHalf(problem));
		        }
		        return combine(left, right);
		    }
		}


[并发之痛 Thread，Goroutine，Actor](http://lenix.applinzi.com/archives/2945)中的几个基本要点：

1. 那我们从最开始梳理下程序的抽象。开始我们的程序是面向过程的，数据结构+func。后来有了面向对象，对象组合了数结构和func，我们想用模拟现实世界的方式，抽象出对象，有状态和行为。但无论是面向过程的func还是面向对象的func，**本质上都是代码块的组织单元，本身并没有包含代码块的并发策略的定义。**于是为了解决并发的需求，引入了Thread（线程）的概念。

2. We believe that writing correct concurrent, fault-tolerant and scalable applications is too hard. Most of the time it’s because we are using the wrong tools and the wrong level of abstraction. —— Akka。，有论文认为当前的大多数并发程序没出问题只是并发度不够，如果CPU核数继续增加，程序运行的时间更长，很难保证不出问题

3. 最让人头痛的还是下面这个问题：系统里到底需要多少线程？从外部系统来观察，或者以经验的方式进行计算，都是非常困难的。于是结论是：让"线程"会说话，吃饱了自己说，自管理是最佳方案。

4. 能干活的代码片段就放在线程里，如果干不了活（需要等待，被阻塞等），就摘下来。我自己的感觉就是：**按需（代码被阻塞）调度，有别于cpu的按时间片调度。**

	* 异步回调方案 典型如NodeJS，遇到阻塞的情况，比如网络调用，则注册一个回调方法（其实还包括了一些上下文数据对象）给IO调度器（linux下是libev，调度器在另外的线程里），当前线程就被释放了，去干别的事情了。等数据准备好，调度器会将结果传递给回调方法然后执行，执行其实不在原来发起请求的线程里了，但对用户来说无感知。
	* GreenThread/Coroutine/Fiber方案 这种方案其实和上面的方案本质上区别不大，关键在于回调上下文的保存以及执行机制。为了解决回调方法带来的难题，这种方案的思路是写代码的时候还是按顺序写，但遇到IO等阻塞调用时，将当前的代码片段暂停，保存上下文，**让出当前线程**。等IO事件回来，然后再找个线程让当前代码片段恢复上下文继续执行，写代码的时候感觉好像是同步的，仿佛在同一个线程完成的，但实际上系统可能切换了线程，但对程序无感。
	* 小结一下：前者即全异步操作，代码直观体现。后者还是阻塞操作，代码顺序写，只是阻塞的是goroutine 之类。

[Scala与Golang的并发实现对比](https://zhuanlan.zhihu.com/p/20009659)

## fork-join实现原理

[Fork and Join: Java Can Excel at Painless Parallel Programming Too!](http://www.oracle.com/technetwork/articles/java/fork-join-422606.html)  在介绍 forkjoin 原理的同时，详述了java 多线程这块的 演化思路。

[forkjoin 泛谈](http://qiankunli.github.io/2018/04/08/forkjoin.html)

## akka实现原理

[akka actor的运行原理](http://colobu.com/2015/05/28/Akka-actor-scheduling/)

`actor ! msg` 本质上是 `executorService execute mbox`，mox实现了ForkJoinTask和Runnable接口。所以说，actor模式的消息是异步的，除了设计理念外，实现上也是没办法。

**如何理解akka代表的actor模式？** 

2019.03.24补充：**actor 是一种并发计算模型**，其中所有的通信，通过发送方的消息传递机制和接收方的信箱队列，在被称为Actor的实体之间发生。Erlang使用Actor 作为它的主体架构成分，随着Akka工具在JVM平台上的成功，actor模型随后人气激增。

## 实现细粒度并行的共同点

1. 提供新的并行执行体抽象、线程level的调度逻辑，线程的业务逻辑变成：决定下一个执行体 ==> 执行
2. 针对共享数据、io等问题，不能执行当前任务的时候，不会阻塞线程（硬件并行资源），执行下一个执行体，绝不闲着。这需要改写共享数据的访问、io等代码。

只是fork join做的还比较简单，体现在

1. 提出了新的并行执行体抽象，这个抽象要交给专门的fork-join pool执行
2. 对共享数据、io等阻塞操作无能为力，只是在合并任务时（特别场景下，可能阻塞的代码），不再干等而已。

golang从设计上就支持协程，goroutine是默认的并行单元，单独开辟线程反而要特别的代码。不像java有历史负担，fork-join着眼点更多在于，基于现有的基本面，如何提高并发性 ==> 需要先分解任务等。fork-join只是提高在特定场景（可以进行子任务分解与合并）下的并行性。所以，goroutine和fork-join根本就不是一回事。**前者是匹敌进程、线程的并发模型，后者是特定问题的并发框架**




