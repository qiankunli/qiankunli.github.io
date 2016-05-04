---

layout: post
title: 换个角度看待设计模式
category: 技术
tags: Other
keywords: java design pattern

---

## 前言(未完待续) 

笔者最近在看一本设计模式方面比较好的书，《设计模式之禅》，一大牛同事觉得《重构：改善既有代码的设计》写的更好，结合之前的一些学习体验，笔者有以下感觉：

1. 代码完成前设计的艺术，设计模式是代码从无到有的过程中提出的一些设计理念
2. 代码完成后优化的艺术，如果我们提出一个原点：“所有的代码写在一个函数”，那么必然涉及到一种分割代码的艺术。

如果我们以第二个角度看待设计模式，就会有很多有意思的体会。

本来主要描述java编程方面

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

    class A{
        void func(){
            1.xx
            2.B b = new B(xx);    // b作为A的类成员跟这个差不多
            3.xx = b.func();
            4.xx
        }
    }
    
但一些框架经常

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

    // 定义配置文件    
    ClassPathResource res = new ClassPathResource(“beans.xml”);
    // 创建bean工厂
    DefaultListableBeanFactory factory = new DefaultListableBeanFactory();
    // 定义读取配置文件的类
    XmlBeanDefinitionReader reader = new XmlBeanDefinitionReader(factory);
    // 加载文件中的信息到bean工厂中
    reader.loadBeanDefinitions(res);
    

两种方式的不同在于：

1. 前者只是将相关代码抽取为一个函数，然后到了另一个类里。（本质上只算是抽取了一个函数）
2. 后者将相关代码完全抽出来，A类中不用保有任何痕迹，可以算是抽取出了一个类

## 拆分

如果所有功能写在了一个函数里，我们如何拆分它

1. 能并行的并行。（设计模式已经不再局限于三大类型，还扩展到多线程模型，io通信模型）
2. 无关的代码拆成独立的类
3. 可能会变的代码拆成独立的类


## 避免直接使用直接干活的类

操作系统是为了避免我们直接使用硬件，编程语言是为了避免我们直接使用系统调用，笔者一个很强烈的感觉就是，设计模式（创建型，行为型，结构型）为了避免我们直接使用一个类。

明明可以new一个，偏偏要用工厂。平明可以`obj.func`,偏偏要`proxyObj.func`

这一切都应了《设计模式之禅》中的思想，所以思想的基本原则就是：对扩展开放，对修改关闭。

一切可能变化的部分都不应该让程序猿直接调用（或者抽象成参数），为了应对变化，把一个类拆成多个类（按照变化的可能性拆分，按照上层接口聚合在一起），甚至不惜把变化本身单独拆成一个类。