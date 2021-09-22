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

如果每次修改都要动很多东西， 这就是代码的“坏味道”，说明抽象的 不是足够好。

[如何写好代码？](https://mp.weixin.qq.com/s/w1hVGfzg-8QyrS2Strvy_g)

1. **虽然代码最后的执行者是机器，但是实际上代码更多的时候是给人看的**。一段代码的生命周期：开发 --> 单元测试 --> Code Review --> 功能测试 --> 性能测试 --> 上线 --> 运维、Bug修复 --> 测试上线 --> 退休下线。开发到上线的时间也许是几周或者几个月，但是线上运维、bug修复的周期可以是几年。在这几年的时间里面，**几乎不可能还是原来的作者在维护了**。
2. 代码本身就是一种交流语言，并且一般来说编程语言比我们日常使用的口语更加的精确。在保持代码逻辑简单的情况下，使用良好的命名规范，代码本身就很清晰并且可能读起来就已经是**一篇良好的文章**。特别是OO的语言的话，本身object（名词）加operation（一般用动词）就已经可以说明是在做什么了。重复一下把这个操作的名词放入注释并不会增加代码的可读性。

[一个加班多新人多团队，我们的代码问题与重构](https://mp.weixin.qq.com/s/MYSF8lCF92ItG_Lc8nOspg)

## 回调很有用

以按行读取文件代码为例
```java
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
```
    
在本例中，一行内容读取后，直接输出`System.out.println(str);`，逻辑不复杂。但若是每行的内容是一个复杂的json，且需要进行复杂的业务处理， 代码就很长了。此外，本例是读取一个磁盘文件，但若是读取hdfs文件，则读取代码至少扩充一倍。若是hdfs 很大，多线程读取时更为复杂。

最后，读取文件一般是只关注读取的数据，弄一堆文件读取代码 和 数据处理逻辑写在一起，“坏味道”很大。

```java
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
```
    
**使用回调分离关注**

从这个例子还可以看到

1. 逻辑是分层的，读取逻辑和数据处理逻辑 不要混在一起。换句话说，如果一个事情有两个明显不同的部分，那么代码应该写在两个地方
2. 程序=逻辑 + 控制，在这个具体的例子中， 读取文件是控制，数据处理是逻辑
3. 实现同样的效果，在java 里要定义一个接口，在scala 则可以直接写。**如果一个逻辑，你用不同的语言实现，最后发现样子差别好大，就说明你没有做好抽象，任由语言特性干扰了代码结构。** 随需求所欲，不滞于物。**你要先知道理想状态是什么样子，然后用具体的语言、技术实现，而不是受困于语言和技术。**
3. 我们写在代码的时候，天然受语言的影响，过程式的、序列化的叙事/代码逻辑。但写代码 应该先想“应该有什么”，而不是“怎么做”。比如，从业务逻辑看，应该有一个观察者模式

	1. 实现时应该先写观察者、监听者等代码， 然后再根据语言 将其串起来。观察者 模式java 与 go的实现很不一样，若是先从语言层面出发，则极易受语言的影响。对于本例来说，在写代码时，最好是先文件读取和数据处理分开写，然后将想办法它们串在一起（学名叫胶水代码 [系统设计的一些体会](http://qiankunli.github.io/2018/09/28/system_design.html)）。
	2. 观察者模式 本身的代码与 业务逻辑 不应混在一起，java 通过提取父类 等形式，将观察者模式本身的代码 与 业务逻辑分开。

## 抽取一个类

假设原来有一个比较复杂的类

    class A{
        void func(){
            1.xx
            2.xx
            3.xx
            4.xx
            5.xx
        }
    }
    
现在我们代码重构，要将步骤234抽出一个类B来，类B需要A的数据初始化，类A需要类B的计算结果。一般有两种方案

1. 方案1

        class A{
            void func(){
                1.xx
                2.B b = new B(xx);    // b作为A的类成员跟这个差不多
                3.xx = b.func();
                4.xx
            }
        }
    
2. 方案2

        class A{
            void func(){
                1. xx
                2. xx
            }
        }
        class B{
            void func(A a){
                1. xx = a.getxx();
                2. xx
                3. a.setxx();
            }
        }
        class Main{
            main{
                A a = new A();
                B b = new B();
                b.func(a);
            }
        }
    
比如spring ioc初始化的一段代码便是如此

```java
// 定义配置文件    
ClassPathResource res = new ClassPathResource(“beans.xml”);
// 创建bean工厂
DefaultListableBeanFactory factory = new DefaultListableBeanFactory();
// 定义读取配置文件的类
XmlBeanDefinitionReader reader = new XmlBeanDefinitionReader(factory);
// 加载文件中的信息到bean工厂中
reader.loadBeanDefinitions(res);
```   

两种方式的不同在于：

1. 前者只是将相关代码抽取为一个函数，然后到了另一个类里。（本质上只算是抽取了一个函数）
2. 后者将相关代码完全抽出来，A类中不用保有任何痕迹，可以算是抽取出了一个类

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
	
此处代码的一个特点就是 执行了`new App().print("hello world",callback)` 却并没有触发 print 动作的执行。从函数式编程的角度来说，实现了从一个函数 到另一个函数的 转换/高阶函数。


## 换个思路看

腾挪代码，本质上都是基于一个抽象，接管你的顺序流，只留一两个logic 部分交给你实现。

[程序员的编程世界观 ](https://www.cnblogs.com/tracyzeng/articles/4108027.html)

1. 过程化编程的步骤是：将待解问题的解决方案抽象为一系列概念化的步骤。然后通过编程的方式将这些步骤转化为程序指令集
2. 过程化语言的不足之处就是它不适合某些种类问题的解决，例如那些非结构化的具有复杂算法的问题。问题出现在，**过程化语言必须对一个算法加以详尽的说明**，并且其中还要包括执行这些指令或语句的顺序。实际上，给那些非结构化的具有复杂算法的问题给出详尽的算法是极其困难的。 
