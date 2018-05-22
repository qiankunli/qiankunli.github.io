---

layout: post
title: 基于aop实现自定义注解
category: 技术
tags: Spring
keywords: JAVA Spring aop

---

## 前言 


[Spring AOP,AspectJ, CGLIB 有点晕](https://www.jianshu.com/p/fe8d1e8bd63e)

[Spring AOP 实现原理与 CGLIB 应用](https://www.ibm.com/developerworks/cn/java/j-lo-springaopcglib/index.html)

## AspectJ

	hello.java
	public class SayHelloService {
	    public void say(){
	        System.out.print("Hello  AspectJ");
	    }
	}
	LogAspect.java
	public aspect LogAspect{
    	pointcut logPointcut():execution(void SayHelloService.say());
    	after():logPointcut(){
         	System.out.println("记录日志 ..."); 
    	}
	}
	
运行时
	
	ajc -d . SayHelloService.java LogAspect.java
	// 生成 SayHelloService.class
	java SayHelloService


1. AspectJ是一套独立的面向切面编程的解决方案，和 Spring 是没有任何关系的。
2. AspectJ属于编译时增强，其生成的 SayHelloService.class 和`javac` 生成的不一样

## Spring Aop 和 aspectj

1. Spring Aop 是运行是增强，即SayHelloService.class 没变，但在运行时 变了，并且实际上，内存中 实际运行的是 SayHelloService$1，为此常涉及到二次代理失效的问题
2. Spring AOP使用了AspectJ的Annotation，但是并没有使用它的编译器和织入器。启用 aspectj 注解 `   <aop:aspectj-autoproxy/>`. “用其名，不用其人”，aop 还是默认jdk动态代理，可选用cglib。
3. Spring 容器中配置一个带 @Aspect 注释的 Bean，Spring 将会自动识别该 Bean，并将该 Bean 作为方面 Bean 处理。方面Bean与普通 Bean 没有任何区别，一样使用 <bean.../> 元素进行配置，一样支持使用依赖注入来配置属性值。

也就是， 除非用上了`ajc` 去编译 java 源代码，否则，都是spring aop + jdk proxy/cglib 那一套。

	@Component
	public class SayHelloService {
	    public void say(){
	        System.out.print("Hello  AspectJ");
	    }
	} 
	@Aspect
	@Component
	public class LogAspect {
	     @After("execution(* com.ywsc.fenfenzhong.aspectj.learn.SayHelloService.*(..))")
	     public void log(){
	         System.out.println("记录日志 ...");
	     }
	}


## 重新理解下aop

1. 不侵入现有代码
2. 切面代码 单独写
3. 切面和切点的整合，可以用正则表达式，也可以用注解

## jdk proxy vs cglib

[Spring的两种代理JDK和CGLIB的区别浅谈](https://blog.csdn.net/u013126379/article/details/52121096)

java动态代理是利用反射机制生成一个实现代理接口的匿名类，在调用具体方法前调用InvokeHandler来处理。

而cglib动态代理是利用asm开源包，对代理对象类的class文件加载进来，通过修改其字节码生成子类来处理。

1、如果目标对象实现了接口，默认情况下会采用JDK的动态代理实现AOP 
2、如果目标对象实现了接口，可以强制使用CGLIB实现AOP 
3、如果目标对象没有实现了接口，必须采用CGLIB库，spring会自动在JDK动态代理和CGLIB之间转换

## 自定义注解无效场景

[Spring AOP无效分析](https://www.jianshu.com/p/e130b5b73c1b)

个人发现的一个情况：

Spring + Aspectj annotation， 假设存在

	interface IHelloService {
		void hello(String name);
	}
	class HelloService implements IHelloService {
		public void hello(String name){
			System.out.pringln("hello " + name);
		}
	}
	
使用注解@LogTime 将方法执行时间加入到 线程的threadlocal 中

||注解加在接口上|注解加在类上|
|---|---|---|
|`<aop:aspectj-autoproxy/>`|有效|无效|
|`<aop:aspectj-autoproxy proxy-target-class="true"/>`|有效|有效|

理论上，刨除性能、必须实现接口的因素， cglib 和 jdk proxy 是可以等效替换的，现在却出现了这个情况。

`<aop:aspectj-autoproxy/>`时

1. 在Aspect 类Around 方法中，`ProceedingJoinPoint pjp` 无法获取 `HelloService.hello` 上的注解。但可以获取 `IHelloService.hello` 上的注解


[接口方法上的注解无法被 @Aspect 声明的切面拦截的原因分析](http://www.importnew.com/28788.html) （待进一步明确描述）

## 二次代理

[Spring中DispacherServlet、WebApplicationContext、ServletContext的关系](https://blog.csdn.net/c289054531/article/details/9196149)

