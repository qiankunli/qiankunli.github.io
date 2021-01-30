---

layout: post
title: 函数式编程的设计模式
category: 架构
tags: Architecture
keywords: functional programming patterns

---

## 简介

在面向对象的观念里: 一切皆对象。但知道这句话并没有什么用，大部分人还是拿着面向对象的写着面向过程的代码，尤其是结合spring + springmvc 进行controller-service-dao 的业务开发。所以，看一个代码是不是“面向对象” 一个立足点就是对java 设计模式的应用。这个思想学名叫“设计模式驱动编程”，在 [如此理解面向对象编程](https://coolshell.cn/articles/8745.html) 被批判了。

对应的，函数式编程时知道“一切皆函数”的意义也很有限，应用函数式编程的一个重要立足点就是：函数式编程中的设计模式。 

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


**有人说过一句话，大意是如果语言支持，就不需要设计模式**。

## 责任链

[Gang of Four Patterns in a Functional Light: Part 3](https://www.voxxed.com/2016/05/gang-four-patterns-functional-light-part-3/)

有些设计模式，很重要，但用的少， 就是被笨重的“面向对象”逻辑耽误了。

	public abstract class AbstractFileParser implements FileParser {
	    protected FileParser next;
	    public void setNextParser( FileParser next ) {
	        this.next = next;
	    }
	}
	public class TextFileParser extends AbstractFileParser {
	    @Override
	    public String parse( File file ) {
	        if ( file.getType() == File.Type.TEXT ) {
	            return "Text file: " + file.getContent();
	        } else if (next != null) {
	            return next.parse( file );
	        } else {
	           throw new RuntimeException( "Unknown file: " + file );
	        }
	    }
	}
	public class AudioFileParser extends AbstractFileParser {
		...
	}
	public class VideoFileParser extends AbstractFileParser {
		...
	}
	
责任链有好几种实现方式，上例是每个节点通过指针串联， 使用时

	FileParser textParser = new TextFileParser();
	FileParser audioParser = new AudioFileParser();
	FileParser videoParser = new VideoFileParser();
	textParser.setNextParser( audioParser );
	audioParser.setNextParser( videoParser );
	File file = new File( File.Type.AUDIO, "Dream Theater  - The Astonishing" );
	String result = textParser.parse( file );

其实呢，责任链的每一个节点可以是一个方法，然后通过 Stream 串联

	String result = Stream.<Function<File, Optional<String>>>of( // [1]
	        ChainOfRespLambda::parseText,
	        ChainOfRespLambda::parseAudio,
	        ChainOfRespLambda::parseVideo )
	        .map(f -> f.apply( file )) // [2]
	        .filter( Optional::isPresent ) // [3]
	        .findFirst() // [4]
	        .flatMap( Function.identity() ) // [5]
	        .orElseThrow( () -> new RuntimeException( "Unknown file: " + file ) ) ); [6]

## 装饰者模式

```go
// 为通过参数传人的内部方法添加运行过程计时
func timeSpent(inner func(op int) int) func(op int) int {
    return func(n int) int {
        start := time.Now()
        ret := inner(n)
        fmt.Println("time spent:",time.Since(start).Seconds())
        return ret
    }
}
```


## 访问者模式

[Gang of Four Patterns in a Functional Light: Part 4](https://www.voxxed.com/2016/05/gang-four-patterns-functional-light-part-4/)

In object-oriented programming the Visitor pattern is commonly used when it is required to add new operations to existing objects but it’s impossible (or not wanted for design reason) to modify the objects themselves and add the missing operations directly inside their implementation.  以前总结过一个[为对象附着一个函数](http://qiankunli.github.io/2018/06/20/rxjava.html)，没想到竟然有官方名称。


	interface Element {
	    <T> T accept(Visitor<T> visitor);
	}
	public static class Square implements Element {
	    public final double side;
	    public Square(double side) {
	        this.side = side;
	    }
	    @Override
	    public <T> T accept(Visitor<T> visitor) {
	        return visitor.visit(this);
	    }
	}
	public static class Circle implements Element {
	    public final double radius;
	    public Circle(double radius) {
	        this.radius = radius;
	    }
	    @Override
	    public <T> T accept(Visitor<T> visitor) {
	        return visitor.visit(this);
	    }
	}
	
假设我们想求	Square 和Circle 的面积area 和周长perimeter

1. 需要将area 和 perimeter放到Element 里
2. 使用Visitor模式

一个简单实现
	
	interface Visitor<T> {
	    T visit(Square element);
	    T visit(Circle element);
	    T visit(Rectangle element);
	}
	public static class AreaVisitor implements Visitor<Double> {
	    @Override
	    public Double visit( Square element ) {
	        return element.side * element.side;
	    }
	    @Override
	    public Double visit( Circle element ) {
	        return Math.PI * element.radius * element.radius;
	    }
	    @Override
	    public Double visit( Rectangle element ) {
	        return element.height * element.width;
	    }
	}
	public static class PerimeterVisitor implements Visitor<Double> {...}
	
用函数式编程翻译一下。如果对scala 等支持pattern match 的代码，此处会更简洁

	public class LambdaVisitor<A> implements Function<Object, A> {
	    private Map<Class<?>, Function<Object, A>> fMap = new HashMap<>();
	    public <B> Acceptor<A, B> on(Class<B> clazz) {
	        return new Acceptor<>(this, clazz);
	    }
	    @Override
	    public A apply( Object o ) {
	        return fMap.get(o.getClass()).apply( o );
	    }	 
	    static class Acceptor<A, B> {
	        private final LambdaVisitor visitor;
	        private final Class<B> clazz;
	        Acceptor( LambdaVisitor<A> visitor, Class<B> clazz ) {
	            this.visitor = visitor;
	            this.clazz = clazz;
	        }
	        public LambdaVisitor<A> then(Function<B, A> f) {
	            visitor.fMap.put( clazz, f );
	            return visitor;
	        }
	    }
	}

笔者对函数替代倒不是很在意，但一串链式操作完成map的赋值，感觉还是很神奇的
	
	static Function<Object, Double> areaCalculator = new LambdaVisitor<Double>()
	        .on(Square.class).then( s -> s.side * s.side )
	        .on(Circle.class).then( c -> Math.PI * c.radius * c.radius )
	        .on(Rectangle.class).then( r -> r.height * r.width );

### 关于访问者模式的一点感觉

**理解角色的分离是理解大部分设计模式的关键**

[访问者模式详解（伪动态双分派）](https://www.cnblogs.com/zuoxiaolong/p/pattern23.html)

1. 静态单分派，方法重载
2. 动态单分派，多态
2. 动态双分派，依据两个实际类型去判断一个方法的运行行为

		for (Bill bill : billList) {
	    	bill.accept(viewer);
	   	}
	   	
	Bill 可能是信用卡账单，也可能是支付宝账单；viewer可能是保存，也可能是打印。假设Bill 有3种，viewer 有2种，则Bill 和 viewer 只实现一个接口，便可以达到 2*3=6 的组合效果。



## 小结

数据结构 描述了数据与数据之间的关系；面向对象描述了对象与对象之间的关系；函数式编程则基于函数与函数之间的关系

1. 实现逻辑时，先考虑用函数实现最小逻辑单元
2. 优先复用编程语言提供的通用函数，比如Runnable、Function、Consumer、XXConsumer
3. 对函数进行逻辑聚合

	* 函数数组/Stream
	* Consumer.andThen(xx).andThen
	* 函数作为参数或返回值
	* 将函数与特定key构成一个map