---
layout: post
title: ThreadLocal小结
category: 技术
tags: Java
keywords: ThreadLocal 线程安全
---

# 前言 #
今年四月份面阿里，前一阵子面美团，一说JAVA基础，都会提到ThreadLocal，看来一句“多线程这方面做的不多”是不会让面试官客气的，好在亡羊补牢，为时未晚，在本文中我来谈谈我对ThreadLocal的理解。
本文的很多观点来自《深入理解java虚拟机》。

# 线程安全 #
我们很难想象在计算机的世界，程序执行时，被不停地中断，共享的数据可能会被修改和变“脏”。为保证程序的正确性，通常我们会想到，确保共享数据在某一时刻只能被一个线程访问，即线程同步。而实现线程同步的一个常用手段便是“互斥”，具体到java代码，通常是使用synchronized关键字。“互斥”后，线程访问是安全了，但并发执行的效率下降了，怎么办？

“互斥”之所以会引起效率下降，是因为就解决“线程安全”这个问题而言，它太“重量级”了，考虑的太过直接和全面。比如，线程A和线程B共享数据data，线程B访问data时，需要先申请锁，但发现锁已经“锁住”，这时，“互斥”通常的做法是挂起线程B。然而，Java线程通常由内核线程实现，线程的挂起和切换等需要系统进行用户态和内核态的转换，太“费劲儿”了。而线程访问data的操作可能耗时很短，这时，我们可以变通一下，线程B发现数据“锁住了”，就空转一下，不要切换。

因此针对一些具体的使用场景，我们放宽要求甚至不采用互斥，也能达到“线程安全”，同时在效率上有所提高。ThreadLocal便是其中之一，“以空间换时间”，既然线程A和线程B竞争data，在某些场景下，可以让线程A和线程B都保有一份data，就可以去掉竞争，井水不犯河水。

# 本地化变量 #

以上是从线程安全的角度出发，那么从线程本身角度看，线程操作时，往往需要一些对象（或变量），这些对象只有这个线程才可以使用。Java在语法层面没有提供线程的“局部变量（成员变量）” 这个支持，当然，我们可以变通一下：
   
    class MyThread extends Thread{
      int abc;	//	我们自定义的局部变量
      public void run(){xxx}
    }


其实为实现这个特性，除了我们自己继承Thread类以外，观察Java Thread类源码，可以发现其有一个ThreadLocalMap成员。我们可以揣测，开发Java的那些大咖们估计我们会有这样的需求，但不知道我们会需要什么样的成员变量，所以预留这样一个“容器”，留给我们来存储自定义成员变量。

	//	Thread类部分源码
	public class Thread implements Runnable {  
	    ThreadLocal.ThreadLocalMap threadLocals= null ;  
		xxx;
	}  


ThreadLocalMap是一个以ThreadLocal为key，Object为值的map，由ThreadLocal维护。

    threadLocals ==> <ThreadLocal1,value1>
	                 <ThreadLocal2,value2>
	                 <ThreadLocal3,value3>


	线程运行{
	//	运行时，线程想通过ThreadLocal拿到这个本地化对象。如果ThreadLocalMap为空，便创建一个ThreadLocalMap挂接到threadLocals上。创建对象，存入到threadLocals中，并返回。
		本地化对象 =  xxx(ThreadLocal)
	//	这个xxx的过程本该由线程负责，但线程往往是预定义好的。便将这部分代码归到了ThreadLocal名下，从而使ThreadLocal不只是作为一个key值存在了。
	
	}

这里，有一个跟寻常开发习惯不同的地方，一般，一个类的成员变量由这个类自己负责初始化，而在Thread类中，由ThreadLocal类负责对其threadLocals成员初始化，对我来讲，了解了这点，ThreadLocal就没有什么神秘了。

目前，我还没有找到解释这个方案的更通俗易懂的方式，真奇怪，想出这个方法的人，怎么不整一个设计模式呢？


# 使用限制 #
