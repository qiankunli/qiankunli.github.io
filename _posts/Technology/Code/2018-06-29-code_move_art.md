---

layout: post
title: 代码腾挪的艺术
category: 架构
tags: Code
keywords: programming

---

## 简介

以笔者目前的开发经历，碰到各种框架，可以分为两类

1. 针对具体业务，为提高代码可读性的 腾挪。最大程度的 隔离control 和 logic。[程序的本质复杂性和元语言抽象](https://coolshell.cn/articles/10652.html)指出：程序=control + logic
2. 代码在线程/主机之间腾挪。为了性能。

## controller-service-dao

编程思想分为面向过程、面向对象、响应式编程、函数式编程等， 但大部分程序猿 是业务开发，具体的说是controller-service-dao，大部分代码是面向过程式的代码。 

倒不是controller-service-dao 有问题，但写惯了controller-service-dao，固化了顺序编程思维，碰到业务上来就controller-service-dao 一通，久而久之就会很难受，代码难写，写完难看，看完不敢改。

关于controller/service/dao 更精炼的部分参见 [系统设计的一些体会](https://qiankunli.github.io/2018/09/28/system_design.html)

## 代码别写在一块

笔者在写项目时，通常是maven父子结构，若出现了五个以上的module，笔者便会将项目分拆出去，为何呀？idea 加载项目的时候能快一点。

我们必须认识到，代码是不断迭代的。这带来的问题就是，你一开始很难 将代码的结构梳理的很好，一次加点代码 最终导致代码很臃肿。因此， 有时必须借助“外力”的作用，找个适合业务场景的框架，**逼着你尽量实现 通过新增 去应对修改**。比如下面两个框架：

1. commons-pipeline [Apache Commons Pipeline 使用学习（一）](http://caoyaojun1988-163-com.iteye.com/blog/2124833)
2. commons-filter

反过来的说，如果每次修改都要动很多东西， 这就是代码的“坏味道”，说明抽象的 不是足够好。

## 代码要写在一块儿

有一个springmvc的业务系统，在某些动作完成后，要对外界发一个通知。常规实现是，在相关Service 中注入一个 RabbitTemplate，在相关代码的位置发消息。这样做其实也挺好，但还是有一些问题

1. RabbitTemplate 散落在系统的多个位置，若是在发消息这块有个改动，就麻烦一点，比如更改消息的序列化方式。
2. 发消息的代码还是无法更直接的表达代码意图，若是代码重构或移交他人，比较容易有纰漏

所以一个比较好的方式是

1. 采用观察者模式
2. 代码上只有一个地方需要注入RabbitTemplate
3. **将代码的主干逻辑与次要逻辑分开**，很多event listener 其实失败了都关系不是特别大。

## 回调很有用

2018.11.16 补充

以按行读取文件代码为例

    public void readFile() throws IOException {
        FileInputStream in = new FileInputStream("test");
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        String str = null;
        while((str = reader.readLine()) != null) {
            System.out.println(str);
        }
        //close
        in.close();
        reader.close();
    }
    
在本例中，一行内容读取后，直接输出`System.out.println(str);`，逻辑不复杂。但若是每行的内容是一个复杂的json，且需要进行复杂的业务处理， 代码就很长了。

此外，本例是读取一个磁盘文件，但若是读取hdfs文件，则读取代码至少扩充一倍。若是hdfs 很大，多线程读取时更为复杂。

最后，读取文件一般是只关注读取的数据，弄一堆文件读取代码 和 数据处理逻辑写在一起，“坏味道”很大。

    public void readFile(LineHandler lineHandler) throws IOException {
        FileInputStream in = new FileInputStream("test");
        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        String str = null;
        while((str = reader.readLine()) != null) {
            lineHandler.handle(str);
        }
        //close
        in.close();
        reader.close();
    }
    
    public void test(){
    	...
    	readFile(new LineHandler(){
    		public void handle(String str){
    			 System.out.println(str);
    		}
    	})
    	...
    }
    
**使用回调分离关注**

从这个例子还可以看到

1. 逻辑是分层的，读取逻辑和数据处理逻辑 不要混在一起。换句话说，如果一个事情有两个明显不同的部分，那么代码应该写在两个地方
2. 程序=逻辑 + 控制，在这个具体的例子中， 读取文件是控制，数据处理是逻辑
3. 实现同样的效果，在java 里要定义一个接口，在scala 则可以直接写。**如果一个逻辑，你用不同的语言实现，最后发现样子差别好大，就说明你没有做好抽象，任由语言特性干扰了代码结构。** 随需求所欲，不滞于物。**你要先知道理想状态是什么样子，然后用具体的语言、技术实现，而不是受困于语言和技术。**
3. 我们写在代码的时候，天然受语言的影响，过程式的、序列化的叙事/代码逻辑。但写代码 应该先想“应该有什么”，而不是“怎么做”。比如，从业务逻辑看，应该有一个观察者模式

	1. 实现时应该先写观察者、监听者等代码， 然后再根据语言 将其串起来。观察者 模式java 与 go的实现很不一样，若是先从语言层面出发，则极易受语言的影响。对于本例来说，在写代码时，最好是先文件读取和数据处理分开写，然后将想办法它们串在一起（学名叫胶水代码 [系统设计的一些体会](http://qiankunli.github.io/2018/09/28/system_design.html)）。
	2. 观察者模式 本身的代码与 业务逻辑 不应混在一起，java 通过提取父类 等形式，将观察者模式本身的代码 与 业务逻辑分开。


## rxjava

[rxjava](http://qiankunli.github.io/2018/06/20/rxjava.html)

[ddd(三)——controller-service-dao的败笔](http://qiankunli.github.io/2018/11/13/controller_service_dao_defect.html)
 
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

类似于函数式编程，返回函数 或者 函数接口的，一定要小心，**代码写在哪里 跟 代码什么时候执行 没啥关系**， 经常违反直觉。

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

### 保护调用者/驱动线程

代码交给另一个线程执行，还有一个好处，就是保护调用者线程。

在一个项目中，不同的线程的重要性是不同的，比如tomcat 线程池中的线程、mq 消费者线程、netty 的事件驱动线程等，它们是驱动 代码执行的源动力。假设tomcat 线程池一共10个线程，当中有一个任务处理较慢，一个线程被占用较长的时间，会严重限制tomcat的吞吐量。

但总有各种耗时的任务，此时，一个重要方法是将 任务交给另一个 线程执行。调用线程 持有 future 对象，可以主动选择 等、不等或者等多长时间。这一点 可以在hystrix 看到。

## 另一台主机的进程和线程执行

[分布式计算系统的那些套路](http://qiankunli.github.io/2018/06/07/write_distributed_system.html)


## 换个思路看

腾挪代码，本质上都是基于一个抽象，接管你的顺序流，只留一两个logic 部分交给你实现。

[程序员的编程世界观 ](https://www.cnblogs.com/tracyzeng/articles/4108027.html)

1. 过程化编程的步骤是：将待解问题的解决方案抽象为一系列概念化的步骤。然后通过编程的方式将这些步骤转化为程序指令集
2. 过程化语言的不足之处就是它不适合某些种类问题的解决，例如那些非结构化的具有复杂算法的问题。问题出现在，**过程化语言必须对一个算法加以详尽的说明**，并且其中还要包括执行这些指令或语句的顺序。实际上，给那些非结构化的具有复杂算法的问题给出详尽的算法是极其困难的。 
3. 对于我个人来说，过程化语言使得 理解多线程代码 非常困难，至少通常和直觉 违背。