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

1. 数据作为双方的参数

        main(){
            Data data;
            New ThreadA(data).start();
            New ThreadB(data).start();  
        }
    
2. 数据作为参数或返回值的方式共享

        threadA{
			Data data;
            New ThreadB(data).start();  
            opt(data)
        }
        
        threadA{
            threadb = New ThreadB(data)；
			Data data = new Task().submit(threadb);
            opt(data)
        }
        
        Task{
        	Data data
        	Data submit(thread){
            	thread.setdata(data).start();
            	retutn data;
            }
        }

### 隐式的共享数据

1. 线程直接互操作

        class ThreadA{
        	TheadB threadB;
            run(){
            	theadB.opt();
            }
        }



## 回调

我们先来用模板模式实现一个功能，下面的B类可以看成是现实中的HibernateTemplate类

    public abstract class B{  
        public void execute(){   
            getConnection();    
            doCRUD();    
            releaseConnection();    
    	}    
        public abstract void doCRUD();  
        public void getConnection(){    
            System.out.println("获得连接...");    
        }    
        public void releaseConnection(){    
            System.out.println("释放连接...");    
        }    
    }  
    public class A extends B{  
        public void doCRUD(){    
        	add()
        }    
        public void add(){    
        	...
        }    
    }  
    public class C extends B{  
        public void doCRUD(){    
        	delete()
        }    
        public void delete(){    
        	...
        }    
    }  

用回调的办法实现下

    interface CallBack{   
        public void doCRUD();     
    }    
    class A{
        private B b;
        public void add(){    
           b.execute(new CustomCallBack(){
                public void doCRUD(){    
                    System.out.println("执行add操作...");    
                }
           });
        }    
    }
    public class B{  
        public void execute(CallBack action){ 
            getConnection();    
            action.doCRUD(); 
            releaseConnection();    
        }    
        public void getConnection(){    
            System.out.println("获得连接...");    
        }    
        public void releaseConnection(){    
            System.out.println("释放连接...");    
        }    
    } 

可以看到，使用回调后，AB从模板模式中的父子关系，变成了依赖关系。

在多线程领域，回调是异步调用的基础


生产者和消费者模式的本质是中间一个队列，从而**将线程之间的共享数据转化了收发消息**。Future、Callable、Executor这一套体系更进一步，解决了消费者处理完消息（也就是task）之后的返回值问题。封装了“数据从调度线程发到工作线程，返回结果从工作线程发到调度线程”的过程，为java的并发编程提供了新的“设计模式”。

线程之间的回调（异步调用），则是调用线程向被调用线程发送一个callback接口，该接口标注了一个待执行方法。

	logthread(){
    	sumtread;
    	func(){
        	sumtread.sum(a,b,callback);
        }
    }
    sumtread{
    	callbacks
        sum(int a,int b,callback){
        	opt(a,b)
        	callbacks.add(callback)
        }
   		run(){
        	for(;;){
				callback = callbacks.get(xx);
            	callback.run();
            }
        
        }
    }

## 引用

[Java多线程设计模式（一）][]

[Promise, Future 和 Callback][]

[JAVA并发设计模式学习笔记（一）—— JAVA多线程编程][]

[JAVA并发设计模式学习笔记（一）—— JAVA多线程编程]: http://www.cnblogs.com/chenying99/p/3321866.html
[Java多线程设计模式（一）]: http://www.cnblogs.com/chenying99/p/3322032.html
[Promise, Future 和 Callback]: http://isouth.org/archives/354.html