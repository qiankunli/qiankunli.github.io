---

layout: post
title: spring aop 实现原理简述
category: 技术
tags: Spring
keywords: proxyFactoryBean

---

## 简介(待整理，过多细节，太罗嗦)

AOP是一套编程思想，是一种功能分解的方法，类似于责任链模式中的横切（具体的实现还真用了责任链模式）。在展开叙述之前，先介绍一些概念及名词。

AOP是一个编程模型，aspectj和jboss AOP对于Aop模型进行了具体的实现。Spring AOP则将aspectj的实现加入到Spring平台中，使得AOP与Spring可以更好的融合起来为开发提供方便的服务。具体的说，spring aop本身不做aop的事，只是提供一种手段（封装和引入），将aop与spring ioc整合在一起。

Aop的实现用到了动态代理技术，动态代理技术主要有两套实现：jdk和cglib。

[spring源码分析之——spring aop原理](http://michael-softtech.iteye.com/blog/814047) 从代码上看，Spring AOP的原理大致如下： 

实现一个InstantiationAwareBeanPostProcessor接口的bean。在每次bean初始化的时候找到所有advisor（spring ioc启动时，会采集类信息存储在BeanDefinition中），根据pointcut 判断是不是需要为将实例化的bean生成代理，如果需要，就把advice编制在代理对象里面。

## spring aop中的一些概念

1. advice:如拦截到对应的某个方法后，我们要做写什么？advice就是拦截后要执行的动作。 类似于j2ee开发中的filter，举个例子

        interface BeforeAdvice{
        	void before(Method method,Object[] args,Object target) throws Throwable;
        }


2. Pointcut：决定advice应该作用于哪个方法。举个例子

        class TruePointCut implements Pointcut{
            // PointCut中的关键就是MethodMatcher成员
        	public MethodMatcher getMethodMatcher(){
        	    return MethodMatcher.TRUE;
            }
        }
        MethodMatcher{
    		// 这会让任何目标方法都会被增强
    		public Boolean matcher(Method method,Class targetClass){
    	        return true;
            }
        }
        
3. advisor，pointcut和advice的结合，举个例子

        class DefaultPointcutAdvisor extends AbstractGenericPointcutAdvisor{
        	private Pointcut pointcut = Pointcut.TRUE;// advice成员在父类中
            public DefaultPointcutAdvisor(Pointcut pointcut, Advice advice) {
        		this.pointcut = pointcut;
        		setAdvice(advice);
            }
        }
        class AbstractGenericPointcutAdvisor{
        	private Advice advice;
        }
        
Pointcut和advice在spring aop中都是一套类图（较多的父子层级关系）。它们既是aop模型中的概念，也对应配置文件中为我们提供的数据。

像上面讲述推送项目一样，我们从两个方面讲spring aop的实现。

## 创建代理对象（假设配置信息都已加入如内存）

![Alt text](/public/upload/java/spring_aop.png)

代理对象由ProxyFactoryBean获取，ProxyFactoryBean是一个工厂bean，其getObject方法主要做了两件事：

1.	加载配置；
2.	创建并调用AopProxy返回代理对象。

Spring通过AopProxy接口类把Aop代理对象的实现与框架的其它部分有效的分离开来。ProxyFactoryBean倒像是一个桥梁，准备了必要的环境（比如将配置文件上的配置加载到属性上），真正的代理对象靠AopProxy生成。

AopProxy的getProxy()方法中调用Proxy.newProxyInstance(xx,xx,Invocationhanlder)创建代理对象，当然了，要为这个调用准备一个InvocationHanlder实现（AopProxy实现类同时也实现了InvocationHanlder接口）。在invoke方法里，触发目标对象方法对应的拦截链的执行。（拦截链的获取下文会提到）

AopProxy实例由AopProxyFactory创建，相关接口描述如下：

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

## spring aop的应用场景

将整体流程写好，把中间的个性化环节暴漏出来，这是框架常做的工作。而在spring中，一边是spring（确切的说，是spring的ioc倡导的pojo编程范式），一边是某个领域的具体解决方案（一般包含复杂的父子关系），两者如何整合？（我就曾经碰到过这样的问题，用spring写多线程程序，怎么看怎么怪）。

这个时候，我们就能看到aop的用武之地，spring的事务，rmi实现等都依赖的aop。

## 小结

本文简要概述了spring aop的实现原理，忽略了一些背景知识，如有疑问，欢迎评论中留言。


