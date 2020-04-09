---

layout: post
title: spring aop 实现原理简述
category: 技术
tags: Spring
keywords: proxyFactoryBean

---

## 简介（待整理）

* TOC
{:toc}

来自《spring源码深度解析》：我们知道，使用面向对象编程有一些弊端，**当需要为多个不具有继承关系的对象引入同一个公共行为时**，例如日志、安全检测等（负责注册的类和处理商品类，都需要记录日志，但它们之间没什么继承关系），我们只有在每个对象里引入公共行为，这样程序中就产生了大量的重复代码，程序就不便于维护了。所以就有了一个面向对象编程的补充，即面向方面编程，AOP所关注的方向是横向的，不同于OOP的纵向。

## 编程思想 or 编程模型

AOP是一套编程思想，是一种功能分解的方法，类似于责任链模式中的横切（具体的实现还真用了责任链模式）。其实，在web开发中，Filter和Servlet，本身就是一个AOP-like，只是跟Spring AOP有以下不同：

- Web Filter和Servlet 的组合逻辑在web容器中已经实现，我们只需写自己的Filter（advice）和Servlet（pointcut）即可。
- Spring AOP的组合逻辑被封装成一个代理类，在运行时生成字节码。

AOP是一个编程模型，aspectj和jboss AOP对于Aop模型进行了具体的实现。Spring AOP则将aspectj的实现加入到Spring平台中，使得AOP与Spring可以更好的融合起来为开发提供方便的服务。具体的说，**spring aop本身不做aop的事，只是提供一种手段（封装和引入），将aop与spring ioc整合在一起**（就好像spring本身不提供kafka，而是通过spring-kafka 将kafka的功能引入到ioc）。

## 使用

配置方式有两种

1. 注解方式，xml 中配置`<aop:aspectj-autoproxy>`

    ```java
    @Aspect
    public class xxx{
        @Pointcut
        public void xxx(){}
        @Before
        public void beforeSleep(){}
    }
    ```
        
2. 纯xml 文件方式（类中不使用注解）
    ```xml
    <aop:config>
        <aop:aspect id="pAspect" ref="permissionCheckAspect">
			<aop:pointcut id="pPointCut"
				expression="(*..*)" />
			<aop:before pointcut-ref="pPointCut" method="xxx" />
		</aop:aspect>
    </aop:config>
	```

spring aop中的一些概念

1. advice:如拦截到对应的某个方法后，我们要做写什么？advice就是拦截后要执行的动作。 类似于j2ee开发中的filter，举个例子

    ```java
    interface BeforeAdvice extends Advice{
        void before(Method method,Object[] args,Object target) throws Throwable;
    }
    ```


2. Pointcut：决定advice应该作用于哪个方法。举个例子
    ```java
    class TruePointCut implements Pointcut{
        // PointCut中的关键就是MethodMatcher成员
        public MethodMatcher getMethodMatcher(){
            return MethodMatcher.TRUE;
        }
    }
    interface MethodMatcher{
        // 这会让任何目标方法都会被增强
        public Boolean matcher(Method method,Class targetClass){
            return true;
        }
    }
    ```
        
3. advisor，pointcut和advice的结合，举个例子
    ```java
    class DefaultPointcutAdvisor extends AbstractGenericPointcutAdvisor{
        private Pointcut pointcut = Pointcut.TRUE;// advice成员在父类中
        public DefaultPointcutAdvisor(Pointcut pointcut, Advice advice) {
            this.pointcut = pointcut;
            setAdvice(advice);
        }
    }
    class AbstractGenericPointcutAdvisor extends AbstractPointcutAdvisor{
        private Advice advice;
    }
    ```

## 动态代理技术

常规的说法是：Aop的实现用到了动态代理技术。但更准确的说：**动态代理 是java 内嵌的对面向切面编程的支持**，java的动态代理必须是基于接口来进行编程的，有一定的局限性。Spring 整合了ASM、CGLIB还有AspectJ 帮助java 在类上做提升（在类上做AOP拦截）。 

[spring源码分析之——spring aop原理](http://michael-softtech.iteye.com/blog/814047) 从代码上看，Spring AOP的原理大致如下： 

实现一个InstantiationAwareBeanPostProcessor接口的bean。在每次bean初始化的时候找到所有advisor（spring ioc启动时，会采集类信息存储在BeanDefinition中），根据pointcut 判断是不是需要为将实例化的bean生成代理，如果需要，就把advice编制在代理对象里面。

AOP应用了java动态代理技术（或者cglib）：基于反射在运行时生成代理类的字节码，下面是一个简单的例子：

    public class BookFacadeProxy implements InvocationHandler {  
        private Object target;    // 委托类  
        // 返回代理类
        public Object createProxy(Object target) {  
            this.target = target;  
            //取得代理对象  
            return Proxy.newProxyInstance(delegate.getClass().getClassLoader(),  
                    delegate.getClass().getInterfaces(), this);   
        }  
        public Object invoke(Object proxy, Method method, Object[] args)  
                throws Throwable {  
            Object result=null;  
            System.out.println("开始");  
            //执行方法  
            result=method.invoke(target, args);  
            System.out.println("结束");  
            return result;  
        }  
    }  


![](/public/upload/spring/spring_aop.png)
    
通过createProxy方法，就可以返回target的代理类实例（内部按序号命名为`$Proxy0`）。target和`$Proxy0`实现同一个接口，构建代理类时，通过`Proxy.newProxyInstance`方法生成代理类每个方法的字节码。
    
而在上面章节提到，ioc在实例化bean时，预留了很多回调函数。所谓的回调函数，具体到java中就是一系列BeanPostProcessor链，BeanPostProcessor包括两个方法：

```java
Object postProcessBeforeInitialization(Object, String)   // 实例化bean前执行
Object postProcessAfterInitialization(Object, String)    // 实例化bean后执行
```

在postProcessAfterInitialization方法中：

```    
Object postProcessAfterInitialization(Object obj, String beanName){
    1. 根据beanName收集匹配的“增强”
    2. 判断采用何种动态代理技术
    3. 根据obj及相关“增强”获取动态代理后的实例result
    4. retrun result;
}
```
    
通过AOP，我们实际使用的类实例，已经不是我们源码看到的“基础类”，而是“基础类”和“增强类”的有机组合。


## 创建代理对象（假设配置信息都已加入如内存）

spring中aop namespace的handler是AopNamespaceHandler

```java
public class AopNamespaceHandler extends NamespaceHandlerSupport {
	@Override
	public void init() {
		// In 2.0 XSD as well as in 2.1 XSD.
		registerBeanDefinitionParser("config", new ConfigBeanDefinitionParser());
		registerBeanDefinitionParser("aspectj-autoproxy", new AspectJAutoProxyBeanDefinitionParser());
		registerBeanDefinitionDecorator("scoped-proxy", new ScopedProxyBeanDefinitionDecorator());

		// Only in 2.0 XSD: moved to context namespace as of 2.1
		registerBeanDefinitionParser("spring-configured", new SpringConfiguredBeanDefinitionParser());
	}
}
```

可以看到，aop:config标签的解析类是：ConfigBeanDefinitionParser

代理对象由ProxyFactoryBean获取，ProxyFactoryBean是一个工厂bean，其getObject方法主要做了两件事：

1.	加载配置；
2.	创建并调用AopProxy返回代理对象。

Spring通过AopProxy接口类把Aop代理对象的实现与框架的其它部分有效的分离开来。ProxyFactoryBean倒像是一个桥梁，准备了必要的环境（比如将配置文件上的配置加载到属性上），真正的代理对象靠AopProxy生成。

AopProxy的getProxy()方法中调用Proxy.newProxyInstance(xx,xx,Invocationhanlder)创建代理对象，当然了，要为这个调用准备一个InvocationHanlder实现（AopProxy实现类同时也实现了InvocationHanlder接口）。在invoke方法里，触发目标对象方法对应的拦截链的执行。（拦截链的获取下文会提到）

AopProxy实例由AopProxyFactory创建，相关接口描述如下：

```java
public interface AopProxy {
    Object getProxy();
    Object getProxy(ClassLoader classLoader);
}
class JdkDynamicAopProxy implements AopProxy, InvocationHandler{
    private final AdvisedSupport advised;   // 包含大量配置信息
    public Object getProxy(ClassLoader classLoader) {
        Class[] proxiedInterfaces = AopProxyUtils.completeProxiedInterfaces(this.advised);
        findDefinedEqualsAndHashCodeMethods(proxiedInterfaces);
        // 返回代理对象
        return Proxy.newProxyInstance(classLoader, proxiedInterfaces, this);
    }
    // InvocationHandler接口方法实现
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        1. 根据method拿到对应的拦截器链
        2. 执行拦截器链
    }
}
```
    
## 加载AOP配置

AOP的配置主要包括两个方面：哪些类的哪些方法需要增强，这些方法需要什么“增强”。与spring惯用的思路一样，第一次执行时，总是伴随着大量的操作（比如加载advice类），部分操作结果会被缓存，以节省下次操作的时间。

    ProxyFactoryBean extends ProxyCreatorSupport (extends AdvisedSupport){
    	private String[] interceptorNames;
        private List<Advisor> advisors;（来自AdvisedSupport类）
    	getObject(){
    	    1. initializeAdvisorChain();		//加载所有的advisor
    	    2. 返回代理对象
        }
    }
   
其中initializeAdvisorChain方法的主要思路

    initializeAdvisorChain(){
    	根据interceptorNames拿到每一个interceptor（即advice）的名字
    	根据名字通过ioc拿到advice实例，并将其加入到advisors成员中
    }
    

advice加载进内存后，会根据方法的不同组成不同的拦截链（根据配置）。如何根据方法拿到对应的拦截链？

    // 该逻辑的调用时机
    class JdkDynamicAopProxy implements AopProxy,InvocationHandler{
        private final AdvisedSupport advised;
    	public Object invoke(Object proxy, Method method, Object[] args){
    		chain = advised.getInterceptorsAndDynamicInterceptionAdvice(method, targetClass);
    		invocation = new ReflectiveMethodInvocation(xx, target, method, args, targetClass, chain);
    		retVal = invocation.proceed();
        }
        public Object getProxy(){
            // 准备数据
            Proxy.newProxyInstance(xx,xx,this);
        }
    }
    
    // 如何拿到
    class AdvisedSupport{
        private List<Advisor> advisors;
        getInterceptorsAndDynamicInterceptionAdvice(method, targetClass){
            1.	先从缓存里看看
            2.  如果缓存中没有
            3.	遍历advisors（依据配置加载的所有advisors（advisor=advice + pointcut）），通过当前advisor中的pointcut的matches方法判断advisor是否使用这个method，使用则将其转换为Interceptor，加入到chain中
            4.	返回chain并加入缓存
        }
    }
    
## 拦截链的执行

刚才提到，虽然同是责任链模式，但aop拦截器链跟一般的责任链模式还是有所不同的。aop的拦截器分为前置，后置和异常时拦截。而在一般的责任链模式中，前置、后置和异常时拦截是通过代码实现来区分的。

    // 链的执行
    class ReflectiveMethodInvocation implements ProxyMethodInvocation{
    	// 目标方法、参数和类型
        protected final Object target,Method method,Object[] arguments,Class targetClass;
        // 当前拦截器的索引
        private int currentInterceptorIndex = -1;
        protected final List interceptorsAndDynamicMethodMatchers;// 拦截器链（已经从advice转化为了interceptor（适配器模式））
        public Object proceed() throws Throwable {
        		// 如果执行到链的最后，则直接执行目标方法
        		// 获取当前interceptor
        		Object interceptorOrInterceptionAdvice =
        		    this.interceptorsAndDynamicMethodMatchers.get(++this.currentInterceptorIndex);
        		if (interceptorOrInterceptionAdvice符合特定类型) {
        			// 执行特定逻辑
        		}else {
        			// 执行拦截器
        			return ((MethodInterceptor) interceptorOrInterceptionAdvice).invoke(this);
        		}
        	}
    }

aop拦截器链的执行逻辑如下

1.	执行所有的前置通知，如果碰到后置通知，则方法入栈（递归调用）。
2.	执行目标方法
3.	执行后置通知（原来压栈的方法出栈）
4.	异常通知（与后置通知类似（都是在方法的后边执行嘛），不过，貌似一个方法的异常通知只能有一个）



## 其它

2019.12.21补充：考虑以下背景

1. 阿里有一个ARMS系统，相当于每个jvm有一个组件向一个中心汇报信息，同时中心可以下发指令给一个jvm执行， 从而实现 通过一个后台管理线上的所有jvm进程。
2. 公司有一个全链路检测系统，可以动态向某个jvm 注入一段指令，比如在某个方法执行前塞入一个`Thread.sleep` 来模拟该方法超时的效果。
3. 公司有一个日志采集监控系统，仅通过jvm 启动时加入agent，就可以获取jvm 运行时的各种信息，比如数据库的连接池大小等

一个jvm 在运行时，不管是动态的，还是静态的，我们都想在不影响原有代码的情况下，做点什么。


