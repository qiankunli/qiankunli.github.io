---

layout: post
title: spring aop 实现原理简述
category: 技术
tags: Spring
keywords: proxyFactoryBean

---

## 简介

* TOC
{:toc}

《spring源码深度解析》：我们知道，使用面向对象编程有一些弊端，**当需要为多个不具有继承关系的对象引入同一个公共行为时**，例如日志、安全检测等（负责注册的类和处理商品类，都需要记录日志，但它们之间没什么继承关系），我们只有在每个对象里引入公共行为，这样程序中就产生了大量的重复代码，程序就不便于维护了。所以就有了一个面向对象编程的补充，即面向方面编程，AOP所关注的方向是横向的，不同于OOP的纵向。

## 编程思想 or 编程模型

AOP是一套编程思想，是一种功能分解的方法，类似于责任链模式中的横切（具体的实现还真用了责任链模式）。其实，在web开发中，Filter和Servlet，本身就是一个AOP-like，只是跟Spring AOP有以下不同：

- Web Filter和Servlet 的组合逻辑在web容器中已经实现，我们只需写自己的Filter（advice）和Servlet（pointcut）即可。
- Spring AOP的组合逻辑被封装成一个代理类，在运行时生成字节码。

AOP是一个编程模型，aspectj和jboss AOP对于Aop模型进行了具体的实现。Spring AOP则将aspectj的实现加入到Spring平台中，使得AOP与Spring可以更好的融合起来为开发提供方便的服务。具体的说，**spring aop本身不做aop的事，只是提供一种手段（封装和引入），将aop与spring ioc整合在一起**（就好像spring本身不提供kafka，而是通过spring-kafka 将kafka的功能引入到ioc）。

[“JVM” 上的AOP：Java Agent 实战](https://mp.weixin.qq.com/s/mX7v5lgfC7JXj-X6xUE3hw) 未读。

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

## 实现

[Go能实现AOP吗？](https://mp.weixin.qq.com/s/dPsFvlWeda1hcnLPQmTQ1w)AOP在Java中的实现方式可以是JDK动态代理和字节码增强技术。
1. JDK动态代理是在运行时动态地**生成了一个代理类**，JVM通过**加载这个代理类再实例化**来实现AOP的能力。
2. 字节码增强技术，在JVM加载字节码时，字节码有一次被修改的机会，但这个字节码的修改比较复杂，好在有现成的库可用，如ASM、Javassist等。

Go能实现AOP吗？Go没有虚拟机一说，也没有中间码，直接源码编译为可执行文件，可执行文件基本没法修改，所以做不了。但没有直路有“弯路”
1. 运行时拦截，在Github找到了一个能实现类似AOP功能的库gohook（当然也有类似的其他库）：可以在方法前插入一些逻辑。它是怎么做到的？通过反射找到方法的地址（指针），然后插入一段代码，执行完后再执行原方法。没有完全测试，不建议生产使用。
2. AST修改源码，认为所有的高级编程语言源码都可以抽象为一种语法树，即对代码进行结构化的抽象，这种抽象可以让我们更加简单地分析甚至操作源码。
我觉得可能还是Go太年轻了，Java之所以要用AOP，很大的原因是代码已经堆积如山，没法修改，历史包袱沉重，最小代价实现需求是首选，所以会选择AOP这种技术。反观Go还年轻，大多数项目属于造轮子期间，需要AOP的地方早就在代码中提前埋伏好了。我相信随着发展，一定也会出现一个生产可用Go AOP框架。

## 源码分析

[spring源码分析之——spring aop原理](http://michael-softtech.iteye.com/blog/814047) 

![](/public/upload/spring/spring_aop.png)


### 何时将 Pointcut替换为代理类

spring中aop namespace的handler是AopNamespaceHandler，可以看到`aop:config`标签的解析类是：ConfigBeanDefinitionParser
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
ConfigBeanDefinitionParser的功能大致有两块：

1. 注册一个AspectJAwareAdvisorAutoProxyCreator类型的bean， 本质是一个BeanPostProcessor
2. 解析主标签下面的advisor标签，并且注册advisor.


在bean初始化的时候（在AbstractAutoProxyCreator中），检查是否需要生成代理对象。如果需要，就生成代理。

```java
// AbstractAutoProxyCreator.java
public Object postProcessBeforeInstantiation(Class<?> beanClass, String beanName) throws BeansException {
    Object cacheKey = getCacheKey(beanClass, beanName);
    ...
    // Create proxy here if we have a custom TargetSource.
    // Suppresses unnecessary default instantiation of the target bean:
    // The TargetSource will handle target instances in a custom fashion.
    if (beanName != null) {
        TargetSource targetSource = getCustomTargetSource(beanClass, beanName);
        if (targetSource != null) {
            this.targetSourcedBeans.add(beanName);
            Object[] specificInterceptors = getAdvicesAndAdvisorsForBean(beanClass, beanName, targetSource);
            Object proxy = createProxy(beanClass, beanName, specificInterceptors, targetSource);
            this.proxyTypes.put(cacheKey, proxy.getClass());
            return proxy;
        }
    }
    return null;
}
```

### 创建代理对象

```java
// AbstractAutoProxyCreator.java
protected Object createProxy(
        Class<?> beanClass, String beanName, Object[] specificInterceptors, TargetSource targetSource) {
    ...
    ProxyFactory proxyFactory = new ProxyFactory();
    ...
    if (!proxyFactory.isProxyTargetClass()) {
        if (shouldProxyTargetClass(beanClass, beanName)) {
            proxyFactory.setProxyTargetClass(true);
        }
        else {
            evaluateProxyInterfaces(beanClass, proxyFactory);
        }
    }
    Advisor[] advisors = buildAdvisors(beanName, specificInterceptors);
    for (Advisor advisor : advisors) {
        proxyFactory.addAdvisor(advisor);
    }
    proxyFactory.setTargetSource(targetSource);
    customizeProxyFactory(proxyFactory);
    ...
    return proxyFactory.getProxy(getProxyClassLoader());
}
```
Spring通过AopProxy接口类把Aop代理对象的实现与框架的其它部分有效的分离开来。

```java
// ProxyFactory.java
public Object getProxy(ClassLoader classLoader) {
    // 实际是JdkDynamicAopProxy.getProxy(classLoader)
    return createAopProxy().getProxy(classLoader);
}
// ProxyCreatorSupport.java
protected final synchronized AopProxy createAopProxy() {
    ...
    // 返回JdkDynamicAopProxy
    return getAopProxyFactory().createAopProxy(this);
}
```
真正的代理对象靠AopProxy生成。AopProxy的getProxy()方法中调用`Proxy.newProxyInstance(xx,xx,Invocationhanlder)`创建代理对象，当然了，要为这个调用准备一个InvocationHanlder实现（AopProxy自身实现类同时也实现了InvocationHanlder接口）。



```java
final class JdkDynamicAopProxy implements AopProxy, InvocationHandler, Serializable {
    private final AdvisedSupport advised;
    public Object getProxy(ClassLoader classLoader) {
        ...
        Class<?>[] proxiedInterfaces = AopProxyUtils.completeProxiedInterfaces(this.advised, true);
        findDefinedEqualsAndHashCodeMethods(proxiedInterfaces);
        return Proxy.newProxyInstance(classLoader, proxiedInterfaces, this);
    }
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        ...
        // Get the interception chain for this method.
		List<Object> chain = this.advised.getInterceptorsAndDynamicInterceptionAdvice(method, targetClass);
        if (chain.isEmpty()) {
            ...
            Object[] argsToUse = AopProxyUtils.adaptArgumentsIfNecessary(method, args);
            retVal = AopUtils.invokeJoinpointUsingReflection(target, method, argsToUse);
        }else {
            // We need to create a method invocation...
            invocation = new ReflectiveMethodInvocation(proxy, target, method, args, targetClass, chain);
            // Proceed to the joinpoint through the interceptor chain.
            retVal = invocation.proceed();
        }
        ...
        return retVal;
    }
}
```

在invoke方法里，先`AdvisedSupport.getInterceptorsAndDynamicInterceptionAdvice(method,targetClass)` 获取拦截链（或者说pointcut 和advice 被组装成了一个 chain），触发目标对象方法对应的**拦截链**的执行。

### 拦截链的执行

虽然同是责任链模式，但aop拦截器链跟一般的责任链模式还是有所不同的。aop的拦截器分为前置，后置和异常时拦截。而在一般的责任链模式中，前置、后置和异常时拦截是通过代码实现来区分的。

```java
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
```

aop拦截器链的执行逻辑如下

1. 执行所有的前置通知，如果碰到后置通知，则方法入栈（递归调用）。
2. 执行目标方法
3. 执行后置通知（原来压栈的方法出栈）
4. 异常通知（与后置通知类似（都是在方法的后边执行嘛），不过，貌似一个方法的异常通知只能有一个）

## 其它

2019.12.21补充：考虑以下背景

1. 阿里有一个ARMS系统，相当于每个jvm有一个组件向一个中心汇报信息，同时中心可以下发指令给一个jvm执行， 从而实现 通过一个后台管理线上的所有jvm进程。
2. 公司有一个全链路检测系统，可以动态向某个jvm 注入一段指令，比如在某个方法执行前塞入一个`Thread.sleep` 来模拟该方法超时的效果。
3. 公司有一个日志采集监控系统，仅通过jvm 启动时加入agent，就可以获取jvm 运行时的各种信息，比如数据库的连接池大小等

一个jvm 在运行时，不管是动态的，还是静态的，我们都想在不影响原有代码的情况下，做点什么。


