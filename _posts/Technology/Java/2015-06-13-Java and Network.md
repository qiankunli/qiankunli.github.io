---

layout: post
title: 网络通信模型与Java
category: 技术
tags: Java
keywords: JAVA network

---

## 前言 ##

从Netty开始，笔者开始了解java nio，进而发现编程语言提供的功能不过是内核的延伸，而linux内核提供了多种网络通信模型。


## 网络编程模型

socket是什么？怎样理解socket和tcp/ip的关系？

“TCP/IP只是一个协议栈，就像操作系统的运行机制一样，必须要具体实现，同时还要提供对外的操作接口，比如create、listen、connect、accept、send、read和write等等。这个就像操作系统会提供标准的编程接口，比如win32编程接口一样，TCP/IP也要提供可供程序员做网络开发所用的接口，这就是Socket编程接口。java提供socket、inputstream或者socketchannel等，本质上也是为了**映射**这些底层的api”

linux支持的网络通信模型

1. 阻塞式io模型        （对应java bio）
2. 非阻塞式io模型
3. io复用模型        （对应java nio）
4. 信号驱动io模型
5. 异步io（对应java aio）

对于一开始就学java socket的人来说，千万不要以为io操作必定会引起阻塞。linux内核提供很多模型，只是上层语言是否提供支持的问题（java是在1.4版本后才通过nio库的方式支持io复用模型和非阻塞模型）。并且，nio中关于Pipe、Channel、Buffer和Selector等概念并不是java首创，而是linux内核本身就提供支持的。

## 各个通信模型的特点

各个模型的不同，**主要**体现在read和write系统调用（accept也可以算上）的行为上，以执行read系统调用为例。

1. 阻塞式io模型

    执行read系统调用会引起线程阻塞，读取完毕后，read返回。
    
2. 非阻塞式io模型

    read调用会立即返回，只是返回的不一定是数据，有可能是一个错误号（表示请求数据还未到达）。
    
3. io复用模型

    在linux中，一切设备皆文件，故socket也可以用文件描述符fd来表示。

    linux提供select系统调用监听（多个）fd的状态，当某个fd就绪时，通知我们处理。
    
        1. // 传入需要监控的描述符，readfds表示一个fd集合，你只关心这些fd是否可读
        2. int select (int n, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
        3. // 如果某个fd可读，select会更改其状态
        4. 遍历readfds{
        5.     如果fd可读{   
        6.         //将读取的数据复制到缓冲区，read立即返回，一定是可以读到数据的，但数据不一定完整
        7.         read(fd,已定义的缓冲区,xx)   
        8.         // 处理逻辑
        9.     }
        10. }
        
        
    linux同时提供epoll系统调用，也是监听fd的状态。
    
        1. epoll_create(xxx)
        // 将事件（比如可读）与fd绑定到一起，当数据到达时，内核会执行一个与读取事件相关的（回调）函数，找到绑定的fd并更改其状态
        2. epoll_ctl(xx,fd,事件,xx,..)
        // 在给定时间内，收集发生的事件
        3. 发生的事件个数 = epoll_wait(xx，发生的事件集合,xx)
        4. for(i=0;i<事件发生的个数;i++){
            // 根据该"事件"结构，可以拿到相应的fd
            // read(fd,已定义的缓冲区,xx)   
        }
        
    select 和 epoll 主要区别是，select轮询所有fd（如果监控的fd过多，轮询一次挺费劲的），select发现fd就绪时（比如可读、可写等），由select更改fd的状态。epoll则是注册fd（与某个事件绑定），当事件发生时，由内核更改fd的状态。epoll其实应用了观察者模式，内核知道数据什么时候到达，在数据处理逻辑外，执行实现注册的“回调”函数。
    
    epoll还相对select进行了其它很多优化措施。

4. 异步io

        // 这表示通知内核读取数据，并立即返回
        1. 执行read系统调用（同时传一个缓冲区和回调函数）。
        // 内核发现数据到达，处理相关逻辑，将数据从内核复制到自定义的缓冲区，执行回调函数
    

io复用和异步io的不同之处是：

不管是select还是epoll模型，都会专门弄一个线程监听fd的状态，fd就绪时，执行read，read系统调用执行时，肯定是可以读到数据的，只是数据不一定完整。

而对于异步io模型，read系统调用执行时，可能数据还未接收，read调用的作用只是告诉内核，数据过来的时候放到哪里，怎么处理，剩下的由内核执行。

## 伪异步io模型

在java支持异步io模型前，为了解决阻塞io的性能问题，或者说破解“一个请求必须占用一个线程”的缺点，人们想了很多办法，比如使用线程池。

当收到一个请求时，将请求暂存到队列中，线程池中的线程依次取出“请求”并执行，这种方式有以下优点：

- 网络通信占用的线程数是可控的
- 使用线程池，省掉了线程创建和销毁的时间

但这样，读写操作还是会引起线程的阻塞。

## 小结

本文对各个网络模型进行了介绍和对比，如果错误，欢迎大家指正。

    
    

    
    



