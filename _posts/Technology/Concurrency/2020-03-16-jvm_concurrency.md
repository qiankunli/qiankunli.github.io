---

layout: post
title: jvm 线程实现
category: 技术
tags: Concurrency
keywords: jvm concurrency

---

* TOC
{:toc}

## 使用

### C语言下的线程使用

看看C语言下写多线程程序什么感觉

```c
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
```

pthread_create 四个参数

1. 线程对象
2. 线程的属性，比如线程栈大小
3. 线程运行函数
4. 线程运行函数的参数
    
从c语言中线程的代码实例和操作系统的基本原理（进程通常是执行一个命令，或者是fork），我们可以看到，线程可以简单的认为是在并发执行一个函数（pthread_create类似于go 代码中常见的`go function(){xxx}`）。

### java 下的线程使用

1. 创建它：继承Thread，实现Runnable，实现TimerTask（现在不推荐）
2. 启动它：start
3. 暂停它（唤醒它）：sleep（自动唤醒），wait和notify
4. 停止它（取消它）：除了 `unsafe.park` 还可以interrupt，注意这种停止并不是抢占式的，代码中要遵守一定的约定。 [java exception](http://qiankunli.github.io/2017/04/22/exception.html) 或者设置一个变量（显式的interrupt）
        
    ```java
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
    ```

## jvm层实现


OSThread: JVM中C++定义的类，代表了JVM中对底层操作系统的osthread的抽象，它维护着实际操作系统创建的线程句柄handle，可以获取底层osthread的状态。

![](/public/upload/concurrency/jvm_thread.png)

[从Java到C++,以JVM的角度看Java线程的创建与运行](https://www.jianshu.com/p/3ce1b5e5a55e)

1. JavaThread: JVM中C++定义的类，一个JavaThread的instance代表了在JVM中的java.lang.Thread的instance, 它维护了线程的状态，并且维护一个指针指向java.lang.Thread创建的对象(oop)。它同时还维护了一个指针指向对应的OSThread，来获取底层操作系统创建的osthread的状态
3. VMThread: JVM中C++定义的类，所有的GC操作都是从VMThread 触发的

[聊聊 Java 并发——基石篇（上）](https://www.infoq.cn/article/Nwq2WyKWevl0mGk_g96C)在创建一个 Thread 对象的时候，除了一些初始化设置之外就没有什么实质性的操作，真正的工作其实是在 start 方法调用中产生的。start() 方法最终调用的是 start0() 这个本地方法，查阅 jdk 源码知道，start0() 方法映射到了 JVM_StartThread 这个方法中，在 `hotspot\src\share\vm\prims\jvm.cpp`

![](/public/upload/jvm/hospot_thread_sequence.png)

## 线程的状态

jvm 运行在不同的操作系统上，独立设计了一套线程状态。比如当jvm thread 创建时，它的状态为NEW，当执行时转变为RUNNABLE，在windows 和linux 上的实现稍有区别：在linux 上创建线程后，虽然设置成NEW，但是Linux 的线程创建完之后就可以执行，所以为了让线程只能在start 之后才能执行， 当linux 线程初始化之后通过一个信号将线程暂停。[Understanding Linux Process States](https://access.redhat.com/sites/default/files/attachments/processstates_20120831.pdf) 

|进程的基本状态|Linux|Java|
|---|---|---|
|运行|TASK_RUNNING||
|就绪||RUNNABLE|
|阻塞|TASK_INTERRUPTIBLE<br>TASK_UNINTERRUPTIBLE|BLOCKED<br>WAITING<br>TIMED_WAITING|
|退出|TASK_STOPPED/TASK_TRACED<br>TASK_DEAD/EXIT_ZOMBIE|TERMINATED|

在 POSIX 标准中（POSIX标准定义了操作系统应该为应用程序提供的接口标准），thread_block 接受一个参数 stat ，这个参数也有三种类型，TASK_BLOCKED， TASK_WAITING， TASK_HANGING，而调度器只会对线程状态为 READY 的线程执行调度，另外一点是线程的阻塞是线程自己操作的，相当于是线程主动让出 CPU 时间片，所以等线程被唤醒后，他的剩余时间片不会变，该线程只能在剩下的时间片运行，如果该时间片到期后线程还没结束，该线程状态会由 RUNNING 转换为 READY ，等待调度器的下一次调度。

![](/public/upload/java/thread_status.jpg)

[Java和操作系统交互细节](https://mp.weixin.qq.com/s/fmS7FtVyd7KReebKzxzKvQ)对进程而言，就三种状态，就绪，运行，阻塞，而在 JVM 中，阻塞有四种类型，**我们可以通过 jstack 生成 dump 文件查看线程的状态**。

1. BLOCKED （on object monitor)  通过 synchronized(obj) 同步块获取锁的时候，等待其他线程释放对象锁，dump 文件会显示 waiting to lock <0x00000000e1c9f108>
2. TIMED WAITING (on object monitor) 和 WAITING (on object monitor) 在获取锁后，调用了 object.wait() 等待其他线程调用 object.notify()，两者区别是是否带超时时间
3. TIMED WAITING (sleeping) 程序调用了 thread.sleep()，这里如果 sleep(0) 不会进入阻塞状态，会直接从运行转换为就绪
4. TIMED WAITING (parking) 和 WAITING (parking) 程序调用了 Unsafe.park()，线程被挂起，等待某个条件发生，waiting on condition

从linux内核来看， BLOCKED、WAITING、TIMED_WAITING都是等待状态。做这样的区分，是jvm出于管理的需要（两个原因的线程放两个队列里管理，如果线程运行出了synchronized这段代码，jvm只需要去blocked队列放一个线程出来。而某人调用了notify()，jvm只需要去waitting队列里取个出来。），本质上是：who when how唤醒线程。


## 锁

Java中往往是按照是否含有某一特性来定义锁

![](/public/upload/concurrency/java_lock.png)

## 其它

![](/public/upload/jvm/hospot_thread_object.png)

### 设置多少线程数

[多线程到底该设置多少个线程？](https://mp.weixin.qq.com/s/IgEXyxA2y0tRMfA7OiqoDA)

1. CPU 密集型任务， N（CPU 核心数）+1，比 CPU 核心数多出来的一个线程是为了防止线程偶发的缺页中断，或者其它原因导致的任务暂停而带来的影响。
2. I/O 密集型任务，这种任务应用起来，系统会用大部分的时间来处理 I/O 交互，而线程在处理 I/O 的时间段内不会占用 CPU 来处理，这时就可以将 CPU 交出给其它线程使用。因此在 I/O 密集型任务的应用中，我们可以多配置一些线程，具体的计算方法是 2N。

我们编码的时候可能不确定运行在什么样的硬件环境中，可以通过 Runtime.getRuntime（).availableProcessors() 获取 CPU 核心。

[程序设计的5个底层逻辑，决定你能走多快](https://mp.weixin.qq.com/s/ar3BRRjAgGXShZ0tuFdubQ) 具体设置多少线程数，主要和线程内运行的任务中的阻塞时间有关系，如果任务中全部是计算密集型，那么只需要设置 CPU 核心数的线程就可以达到 CPU 利用率最高，如果设置的太大，反而因为线程上下文切换影响性能，如果任务中有阻塞操作，而在阻塞的时间就可以让 CPU 去执行其他线程里的任务，我们可以通过 线程数量=内核数量 / （1 - 阻塞率）这个公式去计算最合适的线程数，阻塞率我们可以通过计算任务总的执行时间和阻塞的时间获得。

目前微服务架构下有大量的RPC调用，所以利用多线程可以大大提高执行效率，**我们可以借助分布式链路监控来统计RPC调用所消耗的时间，而这部分时间就是任务中阻塞的时间**，当然为了做到极致的效率最大，我们需要设置不同的值然后进行测试。

### jvm内部工作线程

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