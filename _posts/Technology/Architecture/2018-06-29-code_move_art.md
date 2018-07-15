---

layout: post
title: 代码腾挪的艺术
category: 技术
tags: Architecture
keywords: abtest

---

## 简介

编程思想分为面向过程、面向对象、响应式编程、函数式编程等， 但大部分程序猿 是业务开发，具体的说是controller-service-dao，大部分代码是面向过程式的代码。 

倒不是controller-service-dao 有问题，但写惯了controller-service-dao，固化了顺序编程思维，碰到业务上来就controller-service-dao 一通，久而久之就会很难受，代码难写，写完难看，看完不敢改。

以笔者目前的开发经历，碰到各种框架，可以分为两类

1. 针对具体业务，为提高代码可读性的 腾挪。最大程度的 隔离control 和 logic。[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)指出：程序=control + logic
2. 代码在线程/主机之间腾挪。为了性能。

## 代码别写在一块

笔者在写项目时，通常是maven父子结构，若出现了五个以上的module，笔者便会将项目分拆出去，为何呀？idea 加载项目的时候能快一点。

我们必须认识到，代码是不断迭代的。这带来的问题就是，你一开始很难 将代码的结构梳理的很好，一次加点代码 最终导致代码很臃肿。因此， 有时必须借助“外力”的作用，找个适合业务场景的框架，**逼着你尽量实现 通过新增 去应对修改**。比如下面两个框架：

1. commons-pipeline [Apache Commons Pipeline 使用学习（一）](http://caoyaojun1988-163-com.iteye.com/blog/2124833)
2. commons-filter

反过来的说，如果每次修改都要动很多东西， 这就是代码的“坏味道”，说明抽象的 不是足够好。

## rxjava

[rxjava](http://qiankunli.github.io/2018/06/20/rxjava.html)
 
响应式编程，笔者认为最关键的是 将观察者模式 扩展到数据/事件流上，而事件/数据流 是一种新的写代码的方式。

顺序流的缺点在于，如果一个类的依赖过多，业务较为复杂，代码将成为紧密联系的一个整体，很精巧，但牵一发而动全身，冗余度并不大。

比如用户该买一个商品，关联着商品、库存、订单、红包、优惠券等服务，一旦产品想额外搞个活动，你要改好几处代码。而事件流则不然，一个用户购买事件出来，相关业务方去监听即可。 

观察者模式/响应式编程 使得 调用方和依赖方法 的“接口” 是不变的——都是事件监听。

## 代码在别的函数中执行

	public class App {
	    public static void main(String[] args) {
	        Task task = new App().print("hello world", new Callback() {
	            @Override
	            public void callback() {
	                System.out.println("print finish");
	            }
	        });
	        task.run();
	    }
	    Task print(final String str, final Callback callback) {
	        return new Task() {
	            @Override
	            public void run() {
	                System.out.println(str);
	                callback.callback();
	            }
	        };
	    }
	    interface Callback {
	        void callback();
	    }
	
	    interface Task {
	        void run();
	    }
	}
	
此处代码的一个特点就是 执行了`new App().print("hello world",callback)` 却并没有触发 print 动作的执行。从函数式编程的角度来说，实现了从一个函数 到另一个函数的 转换。

面向对象的基本的理念中 有封装，我们姑且将这种行为 称之为函数封装（当然，这在函数式编程里有专业名词）。

## 代码在另一个线程执行

此时，代码调用 变成了代码提交

[异步编程](http://qiankunli.github.io/2017/05/16/async_servlet.html)

一个整体同步的逻辑里 加上一个异步执行，写起代码来依然很难受，因为你要处理异步调用返回的数据。

	Future future = Executors.execute(xx);
	Data data = future.get();
	handle(data)

为此，干脆处理数据的逻辑 也让 执行线程给干了。但若是 数据处理逻辑很复杂呢？上文的函数封装 就派上用场了，具体参见 [rxjava](http://qiankunli.github.io/2018/06/20/rxjava.html)
 
弄了一堆的`AsnycJob.map(Function1).map(Function2).run()` 在run 中将这些逻辑 封装成一个函数 交给 另一个线程执行。

函数的封装 使得我们 不管同步代码 还是异步代码，都可以进行一个统一的流式的处理。

## 另一台主机的进程和线程执行

[分布式计算系统的那些套路](http://qiankunli.github.io/2018/06/07/write_distributed_system.html)


## 换个思路看

腾挪代码，本质上都是基于一个抽象，接管你的顺序流，只留一两个logic 部分交给你实现。

[程序员的编程世界观 ](https://www.cnblogs.com/tracyzeng/articles/4108027.html)

1. 过程化编程的步骤是：将待解问题的解决方案抽象为一系列概念化的步骤。然后通过编程的方式将这些步骤转化为程序指令集
2. 过程化语言的不足之处就是它不适合某些种类问题的解决，例如那些非结构化的具有复杂算法的问题。问题出现在，**过程化语言必须对一个算法加以详尽的说明**，并且其中还要包括执行这些指令或语句的顺序。实际上，给那些非结构化的具有复杂算法的问题给出详尽的算法是极其困难的。 
3. 对于我个人来说，过程化语言使得 理解多线程代码 非常困难，至少通常和直觉 违背。