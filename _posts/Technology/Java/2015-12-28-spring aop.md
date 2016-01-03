---

layout: post
title: spring aop实现原理简述
category: 技术
tags: Java
keywords: proxyFactoryBean

---

## 简介

AOP是一套编程思想，是一种功能分解的方法，类似于责任链模式中的横切（具体的实现还真用了责任链模式）。在展开叙述之前，先介绍一些概念及名词。

aspectj和jboss AOP对于Aop模型进行了具体的实现。Spring AOP则将aspectj的实现加入到Spring平台中，使得AOP与Spring可以更好的融合起来为开发提供方便的服务。具体的说，spring aop本身不做aop的事，只是提供一种手段（封装和引入），将aop与spring ioc整合在一起。

Aop的实现用到了动态代理技术，动态代理技术主要有两套实现：jdk和cglib。

## 从一个项目开始说起

笔者曾经实现过一个项目，根据用户的收听记录，为用户推送消息，以达到挽留用户的目的，具体情况是：

1.	根据用户的最后收听时间，将用户分为不同的组
2.	每个组有一个推送策略。比如用户已经7天没有登录app，则为用户推送一些文案，推送选项包括：订阅专辑更新、推荐专辑（根据机器学习得到）以及默认推送文案。
3.	推送策略的每个选项有优先级，假设“订阅专辑更新”优先级最高，则如果用户订阅的专辑有更新，为用户推送“亲爱的xx，您订阅的xx有更新了”。如果没有更新，则尝试其他推送选项。

我在实现该项目时，运用了工厂模式及责任链模式。具体流程如下：

1.	加载配置。将所有推送选项加载进来，根据策略配置组成推送链，并建立推送分组和推送链的映射关系（形成一个“推送策略工厂”）。
2.	推送过程。根据用户属性计算用户所属的分组，通过“推送策略工厂”返回该分组对应的推送链，触发推送链。

今天笔者拜读《Spring技术内幕》，看到spring aop的源码分析，其实现与笔者项目真是异曲同工（当然，多少还有点不一样），对应关系如下：

    推送项目	                    Spring AOP
    用户	                        目标对象要增强的方法
    推送选项，比如“订阅专辑更新”	    Advice
    推送策略	                    拦截链
    推送策略工厂	                拦截链工厂
    
希望这可以作为引子，可以让读者更容易理解下面的内容。

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
2.	创建并调用AopProxy返回代理对象。Spring通过AopProxy接口类把Aop代理对象的实现与框架的其它部分有效的分离开来。ProxyFactoryBean倒像是一个桥梁，准备了必要的环境，真正的代理对象靠AopProxy生成。
3.	AopProxy的getProxy()方法中调用Proxy.newProxyInstance(xx,xx,Invocationhanlder)创建代理对象，当然了，要为这个调用准备一个InvocationHanlder实现（AopProxy实现类同时也实现了InvocationHanlder接口）。在invoke方法里，AopProxy又把活交给了ReflectiveMethodInvocation对象（具体的说是该对象的proceed方法），在proceed方法中，触发目标对象方法对应的拦截链（AopProxy传入的）的执行。（拦截链的获取下文会提到）

AopProxy实例由AopProxyFactory创建，相关接口描述如下：

    Interface AopProxy{
    	Object getProxy();
    	Object getProxy(ClassLoader classLoader);
    }
    Interface AopProxyFactory{
    	AopProxy createAopProxy(AdvisedSupport config) throws AopConfigException;
    	// 根据config决定生成哪种AopProxy，包括JdkDynamicAopProxy和CglibProxyFactory(通过CglibProxyFactory创建Cglib2AopProxy)
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
    	根据interceptorNames拿到每一个interceptor（即advice）的名字，根据名字通过ioc拿到advice实例，并将其加入到advisors成员中。
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
            3.	遍历advisors（依据配置加载的所有advisors（advisor=advice + pointcut）），通过当前advisor中的pointcut的matches方法判断advisor是否使用这个method，使用则加入到chain中
            4.	返回chain并加入缓存
        }
    }

## spring aop的应用场景

将整体流程写好，把中间的个性化环节暴漏出来，这是框架常做的工作。而在spring中，一边是spring（确切的说，是spring的ioc倡导的pojo编程范式），一边是某个领域的具体解决方案（一般包含复杂的父子关系），两者如何整合？（我就曾经碰到过这样的问题，用spring写多线程程序，怎么看怎么怪）。

这个时候，我们就能看到aop的用武之地，spring的事务，rmi实现等都依赖的aop。

## 小结

本文简要概述了spring aop的实现原理，忽略了一些背景知识，如有疑问，欢迎评论中留言。


