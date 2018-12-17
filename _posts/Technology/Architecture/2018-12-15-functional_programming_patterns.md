---

layout: post
title: 函数式编程的设计模式
category: 技术
tags: Architecture
keywords: functional programming patterns

---

## 简介(持续更新)

可预先看下 [函数式编程](http://qiankunli.github.io/2018/09/12/functional_programming.html)  

在面向对象的观念里: 一切皆对象。但知道这句话并没有什么用，大部分人还是拿着面向对象的写着面向过程的代码，尤其是结合spring + springmvc 进行controller-service-dao 的业务开发。所以，看一个代码是不是“面向对象” 一个立足点就是对java 设计模式的应用。

对应的，函数式编程时知道一切皆函数的意义也很有限，应用函数式编程的一个重要立足点就是：函数式编程中的设计模式。 



## Functional Programming Design Patterns

本小节膝盖给 ScottWlaschin 大神，其slide 有一幅图[Functional Programming Patterns (NDC London 2014)](https://www.slideshare.net/ScottWlaschin/fp-patterns-ndc-london2014) 其在youtube 有对应的演讲。

![](/public/upload/architecture/function_programming_patterns.jpg)

[Gang of Four Patterns in a Functional Light: Part 1
](https://www.voxxed.com/2016/04/gang-four-patterns-functional-light-part-1/)

a simple exercise of grammatical analysis. Consider a sentence like: “smoking is unhealthy” or even “running is tiring”. What are “smoking” and “running” in this context? In English, the -ing suffix transforms verbs like to smoke or to run into nouns. The biggest part of the design patterns listed in the Gang of Four book, especially the ones classified as behavioural patterns, follow exactly the same approach. Like the -ing suffix, they turn verbs into nouns – or in this case, functions into objects. 面向对象设计模式经常在搞一件事，**把动词转换为名词**，但很不幸，这个动作很多时候没有必要。 

Unfortunately, this transformation process is often unnecessary, or merely serves the purpose of shoehorning some concepts that are natural in functional programming into the object oriented paradigm. Moreover, this adaptation comes at the cost of a higher verbosity, lower readability and more difficult maintainability. In fact, it not only requires you to create objects with the exclusive purpose of wrapping one or more functions into them, but it also makes it necessary to develop some extra logic to glue these objects together down the line. The same goal could be achieved with a straightforward function composition. 把动作搞成对象，不仅多一个对象的概念，还要你花精力将几个对象黏合在一起（胶水代码），远不如function composition 来的直接。

### 命令模式的“缩写”

	public static void log(String message) {
	    System.out.println("Logging: " + message);
	}
	 
	public static void save(String message) {
	    System.out.println("Saving: " + message);
	}
	 
	public static void send(String message) {
	    System.out.println("Sending: " + message);
	}

	List<Runnable> tasks = new ArrayList<>();
	tasks.add(() -> log("Hi"));
	tasks.add(() -> save("Cheers"));
	tasks.add(() -> send("Bye"));
	execute( tasks );
	
LogCommand、SaveCommand、SendCommand 三个函数便实现了。

Rethinking the command implementation in terms of plain function brings the benefit of dramatically increase the signal / noise ratio of our code, where the signal is the body of the functions while the noise is all the additional code used to represent that function as the method of an object. **作者提到了一个词儿：signal/noise ratio（为代码可读性提供了一种新的评价标尺）**，signal是真正干活儿的代码， noise是为设计模式服务、作为一个对象要额外添加的代码 。传统的命令模式，Command interface 是主角，100 行代码真正干活儿 就是那几行代码，剩下的代码体现了设计模式的设计，但跟业务没啥直接关系。 而使用函数式编程，signal / noise ratio 比例很高。

函数式编程还有一点影响，Runnable 从java8 跟线程强绑定 编程了一个类似Function、Consumer 之类的通用 Functional interface。


It is worth noticing that the functions are actually finer grained than the strategy classes (they can be combined in a way not available by any class) and allow even better reusability. **函数比类更适合作为 逻辑的最小单元**，因为一些聚合方式类并不支持（比如高阶函数），函数也比类更容易被复用（Class::function；class::funciton 就可以复用了）

## 观察者模式

[Gang of Four Patterns in a Functional Light: Part 2](https://www.voxxed.com/2016/05/gang-four-patterns-functional-light-part-2/)

the Template and the Observer patterns, which can both be reimplemented through the Java 8 Consumer interface.

	interface Listener {
	    void onEvent(Object event);
	}
	public class Observable {
	    private final Map<Object, Listener> listeners = new ConcurrentHashMap<>();
	    public void register(Object key, Listener listener) {
	        listeners.put(key, listener);
	    }
	    public void unregister(Object key) {
	        listeners.remove(key);
	    }
	 	// Observable will send an event it will be broadcast to both
	 	// broadcast 一词用的贴切 
	    public void sendEvent(Object event) {
	        for (Listener listener : listeners.values()) {
	            listener.onEvent( event );
	        }
	    }
	}

the Listener interface we defined above is semantically equivalent to the Consumer，所以等价替换下就成了

	Observable observable = new Observable();
	observable.register( "key1", e -> System.out.println(e) );
	observable.register( "key2", System.out::println );
	observable.sendEvent( "Hello World!" );


以下未读

[Gang of Four Patterns in a Functional Light: Part 3](https://www.voxxed.com/2016/05/gang-four-patterns-functional-light-part-3/)

[Gang of Four Patterns in a Functional Light: Part 4](https://www.voxxed.com/2016/05/gang-four-patterns-functional-light-part-4/)