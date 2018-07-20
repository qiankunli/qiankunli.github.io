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
2. Spring AOP使用了AspectJ的Annotation，但是并没有使用它的编译器和织入器。启用 aspectj 注解 `<aop:aspectj-autoproxy/>`. “用其名，不用其人”，aop 还是默认jdk动态代理，可选用cglib。
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

[AOP中获取方法上的注解信息](http://loveshisong.cn/%E7%BC%96%E7%A8%8B%E6%8A%80%E6%9C%AF/2016-06-01-AOP%E4%B8%AD%E8%8E%B7%E5%8F%96%E6%96%B9%E6%B3%95%E4%B8%8A%E7%9A%84%E6%B3%A8%E8%A7%A3%E4%BF%A1%E6%81%AF.html)

分析切面代码

	Object interceptMethod(final ProceedingJoinPoint pjp){
		  Method method = ((MethodSignature) (pjp.getSignature())).getMethod();
		  ...
	}
	
对代码打断点可以发现，MethodInvocationProceedingJoinPoint pjp 的实例数据为 

    MethodInvocationProceedingJoinPoint
	    methodInvocation
	 	    proxy=HelloService
	 	    method=IHelloService.hello
	 	    target=HelloService

因此，通过`((MethodSignature) (pjp.getSignature())).getMethod()` 得到的是接口方法。因为target 指向是正确的，可以根据 taget 简介获取

	 Class<?> classTarget = pjp.getTarget().getClass();
	 Class<?>[] par = ((MethodSignature) pjp.getSignature()).getParameterTypes();
	 Method method = classTarget.getMethod(pjp.getSignature().getName(), par);


[接口方法上的注解无法被 @Aspect 声明的切面拦截的原因分析](http://www.importnew.com/28788.html) 

**如果类本身就不被spring 管理（也就是 new 创建的），则肯定无法被拦截**

## 重新理解下aop

1. 不侵入现有代码
2. 切面代码 单独写
3. 切面和切点的整合，可以用正则表达式，也可以用注解。换个角度看，注解往往是为了更好的描述切点。

## 拦截框架代码的执行

上文中提到的aop，往往是aop 一整套代码 作为一个jar 提供给用户使用，用户将注解加载自己的业务代码上。还有一种场景：拦截框架代码的执行。比如拦截ibatsis sqlMapTemplate 的执行，获取sql 表达式存在队列中。另起线程 读取队列 中的sql，批量执行。

实现时有一个问题，即一旦拦截 sqlMapTemplate，spring 实际会构建一个sqlMapTemplate 的代理类 给 各个dao 类使用。若是dao类 写死了依赖 sqlMapTemplate 类，则代理类是无法强转成 sqlMapTemplate 类的。

为此，要提供一个 sqlMapTemplate 的装饰类，所有 sqlMapTemplate 的调用方 依赖 sqlMapTemplateDecorator 类。aop 拦截 sqlMapTemplateDecorator 获取业务方调用信息，sqlMapTemplate 可以独善其身，继续为框架内 其它强依赖 sqlMapTemplate 的类 提供服务。

## 二次代理

[Spring中DispacherServlet、WebApplicationContext、ServletContext的关系](https://blog.csdn.net/c289054531/article/details/9196149)

[简述Spring容器与SpringMVC的容器的联系与区别](https://blog.csdn.net/wzx104104104/article/details/74937605)

spring mvc 和 spring 是两个ioc，后者是前者的 父ioc。子容器(SpringMVC容器)可以访问父容器(Spring容器)的Bean，父容器(Spring容器)不能访问子容器(SpringMVC容器)的Bean。

假设项目目录如下所示，abtest-client 注解作用于 service包下的HelloService类中

    com.test.abc.controller
    com.test.abc.service
    com.test.abc.service.IHelloService
    com.test.abc.service.imp.HelloService

spring-context.xml 配置如下

    <context:component-scan base-package="com.test.abc"/>
    <aop:aspectj-autoproxy/>

web-context.xml 配置如下

    <context:component-scan base-package="com.test.abc"/>

则spring mvc 与spring 容器中都将保有IHelloService 实例，但spring mvc ioc 下是HelloService 本身，而spring ioc 则是abtest-client
处理过后的代理类，按照"子ioc 有bean 则用自己的，找不到就用父ioc "的规则，生效的是spring mvc ioc 下的HelloService，abtest 无效。

