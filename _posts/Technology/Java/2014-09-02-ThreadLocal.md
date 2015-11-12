---
layout: post
title: ThreadLocal小结
category: 技术
tags: Java
keywords: ThreadLocal 线程安全
---

## 前言 ##
今年四月份面阿里，前一阵子面美团，一说JAVA基础，都会提到ThreadLocal，看来一句“多线程这方面做的不多”是不会让面试官客气的，好在亡羊补牢，为时未晚，在本文中我来谈谈我对ThreadLocal的理解。
本文的很多观点来自《深入理解java虚拟机》以及《java特种兵》。

## 线程安全 ##
我们很难想象在计算机的世界，程序执行时，被不停地中断，共享的数据可能会被修改和变“脏”。为保证程序的正确性，通常我们会想到，确保共享数据在某一时刻只能被一个线程访问。一个常用手段便是“互斥”，具体到java代码，通常是使用synchronized关键字。“互斥”后，线程访问是安全了，但并发执行的效率下降了，怎么办？

“互斥”之所以会引起效率下降，是因为就解决“线程安全”这个问题而言，它太“重量级”了（或者说，粒度太大了），考虑的太过直接和全面。比如，线程A和线程B共享数据data，线程B访问data时，需要先申请锁，但发现锁已经“锁住”，怎么办？

1. 挂起线程B。

    然而，Java线程通常由内核线程实现，线程的挂起和切换等需要系统进行用户态和内核态的切换，太“费劲儿”了。而线程访问data的操作可能耗时很短，为此挂起线程会引起一些浪费。
2. 使用读写锁，读操作之间是不互斥的。
3. 线程B发现数据“锁住了”，就空转一下，等一会儿再试试可不可以访问。（这完全就是另外一种并发编程模型了，基于这个模型，也产生了一系列的组件，比如队列等）
4. 在某些场景下，可以让线程A和线程B都保有一份data，就可以去掉竞争，“以空间换时间”。

因此针对一些具体的使用场景，我们放宽要求甚至不采用互斥，也能达到“线程安全”，同时在效率上有所提高。

## 线程原生的局部变量 ##

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

threadLocals是Thread的default类型的成员，ThreadLocal跟Thread类在一个包下，所以在ThreadLocal类中可以`Thread.currentThread().threadLocals`来操作threadLocals成员。

    threadLocals(是一个map) ==> <ThreadLocal1,value1>
	                           <ThreadLocal2,value2>
	                           <ThreadLocal3,value3>

ThreadLocal有以下方法：

    set(v){
        当前线程.threadlocals.put(this,v);
    }
    get(v){
        当前线程.threadlocals.get(this);
    }
    remove(v){
        当前线程.threadlocals.remove(this);
    }

这里，有一个跟寻常开发习惯不同的地方，一般，一个类的成员变量由这个类自己负责初始化，而在Thread类中，由ThreadLocal类负责对其ThreadLocalMap成员初始化。由于一个ThreadLocal包装一个value，所以ThreadLocal对象也可以和value形成一对一映射。

换句话说，变量有类作用域，对象作用域和线程作用域。只要将一个变量放在线程的threadLocals成员中，这个变量便有了线程作用域。与类作用域和对象作用域不同，这两种作用域的变量直接用关键字注明即可。一个变量要想拥有线程作用域，也就是要进入threadLocals这个map中，必须通过ThreadLocal类的操作（ThreadLocal类和Thread类一个包，可以直接操作threadLocals成员），同时还要一个key搭伙，ThreadLocal类对象也可以代劳。

## 使用模式 ##

变相传递参数的一个例子（实现变量在同一线程内，跨类使用）

    MyContext{
        public static ThreadLocal<Integer> numThreadLocal = new ThreadLocal<Integer>();
        public void set(Integer num){
            numThreadLocal.set(num);
        }
        public Integer get(){
            return numThreadLocal.get();
        }
        public void close(){
            numThreadLocal.remove();
        }
    }
    MyComponent{
        public void say(){
            System.out.println("num ==> " + MyContext.get())
        }
    }
    Main{
        public static void main(String[] args){
            for(int i=0;i<10;i++){
                final int num = i;
                new Thread(){
                    public void run(){
                        MyContext.set(num);
                        new MyComponent().say();
                        MyContext.close();
                    }
                }.start();
            }
        }
    }

使用ThreadLocal时，要注意释放资源，对于一个正常的线程，线程运行结束后，ThreadLocal数据会自动释放。而对于线程池提供的线程，有时很长时间都不会释放（线程是被复用的），ThreadLocal变量的积累会导致线程占用资源过多。

在这个例子中，如果将MyContext按如下方式书写：

    MyContext{
        public static Integer num = new Integer();
        public void set(Integer num){
            this.num = num;
        }
        public Integer get(){
            return this.num;
        }
    }

那么输出的内容，就很有可能相互干扰了。

## 引用

严重推荐这篇文章： [Java中ThreadLocal模拟和解释][]

[Java中ThreadLocal模拟和解释]: http://woshixy.blog.51cto.com/5637578/1275284