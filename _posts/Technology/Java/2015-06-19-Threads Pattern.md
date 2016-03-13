---

layout: post
title: Java多线程设计模式
category: 技术
tags: Java
keywords: 多线程 JAVA

---

## 前言

java性能的提高，io和多线程是其中两个重要部分，io方面java实现了nio和aio，后者则有一系列的设计模式和lambda表达式的支持，相信java在以后的很长一段时间，还会继续发光发热的。

了解一些设计模式，不仅可以提高编程能力，对学习一些框架的源码也是很有帮助的。

## 线程安全

编写多线程程序时，一个逃不过的坑就是线程安全。

### 什么导致线程不安全

大学课本上的操作系统原理，只说了多线程要处理资源的争用，没有直接说下面的事

1. 多个线程操作同一块内存（比如变量），这个是最主要的。
2. 重排序：编译器，为了优化所做的重排序；处理器，硬件指令重拍排序
2. 缓存：CPU各个核心虽然共用一个内存，但都有自己的缓存
3. 在原子性：比如count++，再字节码中则体现为"读取-修改-写入"。在多线程环境下，某个线程操作的值可能已经失效。

### 如何确保线程安全

基本的思路是针对上述线程不安全的原因进行规避。

1. 我们自己写代码时，尽量安全的操作共享内存。比如在Servlet类中操作实例变量时加锁。
2. 编写并使用线程安全对象。比如使用ConcurrentHashMap来替代HashMap，通过继承和组合等方式，在现有的线程安全类的基础上扩展新的类等
3. 使用线程工具类

    - 同步类，比如vector等，同一个时刻只有一个线程可以操作容器数据
    - 并行类，比如ConcurrentHashMap，同一个时刻可以有多个线程操作容器数据，其本身负责线程安全。
    - 同步工具类，同步工具类都包含一些特定的结构化属性，它们封装了一些状态，这些状态将决定了执行同步工具类的线程是继续执行还是等待（即可以根据自身的状态来协调线程的控制流），此外还提供了一些方法对状态进行操作，以及另一些方法用于高效的等待同步工具类进入到预期状态。（来自《Java并行编程实战》）

## 线程之间如何发生关系

线程之间影响彼此的办法，根本上就是共享数据，具体的说是，相关线程操作数据来影响彼此（前提是相关线程，都有操作共享数据数据的手段）。同步和锁，本质上也是如此。

此处以两个线程为例

### 显式的共享数据

1. 异构方式（以不同的逻辑处理共享数据）

        main(){
            Data data;
            New ThreadA(data).start();
            New ThreadB(data).start();  
        }
    
2. 同构方式（以相同的逻辑处理共享数据）

        class MyTask implement Runnable{
        	Data data;
        	run(){}
        }
        main(){
            MyTask task = new MyTask();
            New Thread(task).start();
            New Thread(task).start();
        }
        
### 隐式的共享数据

数据虽然保有在threadB中，但threadA有操作它的手段。

1. 直接操作

        class ThreadA{
        	TheadB threadB;
            run(){
            	theadB.opt();
            }
        }
    
2. 间接操作

        DataObj dataObj{
            Data data;
            ThreadB threadB;
            func(){
                threadB.opt();
            }
        }
        class ThreadA{
            DataObj dataObj
            run(){
                dataObj.func();
            }
        }
        main(){
            new ThreadA().start();
        }


        
共享对象的形式有以下几种：

1. 两个线程都会访问共享对象的所有成员，用来同步状态（比如锁），分担任务（比如共享一个队列，两个线程不停的从队列中取任务）等。
2. 两个线程访问共享对象的成员没有“严格的”交叉

    这里谈一个很有意思的Future类，整个Future、Callable、Executor这一套体系的本质是有一个Sync对象，该对象简单说：

        class Sync{
            Callable callable;
            V result;
            Exception execption;
        }

    调用线程来这里拿结果result，被调用线程来这里拿任务callable，并把执行结果放到result中。

3. 两个线程访问共享对象的成员有交叉


这里着重谈一下第二点。生产者和消费者模式的本质是中间一个队列，从而**将线程之间的共享数据转化了收发消息**。Future、Callable、Executor这一套体系更进一步，解决了消费者处理完消息（也就是task）之后的返回值问题。封装了“数据从调度线程发到工作线程，返回结果从工作线程发到调度线程”的过程，为java的并发编程提供了新的“设计模式”。

## 目标对象和线程的关系

以对象之间关系的角度（继承，依赖，组合等），来理解对象和线程。

1. 目标对象不保有使用它的线程的引用。类似于上节“显式的共享数据”
2.	目标对象保有使用它数据的线程的引用。这类似于上节"隐式的共享数据"，因为object的func方法中操作了threada，这个fun方法必然被某个threadb执行。


## 引用

[Java多线程设计模式（一）][]

[Promise, Future 和 Callback][]

[JAVA并发设计模式学习笔记（一）—— JAVA多线程编程][]

[JAVA并发设计模式学习笔记（一）—— JAVA多线程编程]: http://www.cnblogs.com/chenying99/p/3321866.html
[Java多线程设计模式（一）]: http://www.cnblogs.com/chenying99/p/3322032.html
[Promise, Future 和 Callback]: http://isouth.org/archives/354.html