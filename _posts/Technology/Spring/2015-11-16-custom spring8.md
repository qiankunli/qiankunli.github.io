---

layout: post
title: 自己动手写spring（八） 支持AOP
category: 技术
tags: Spring
keywords: Java Spring

---

## 前言

本文在前文的基础上提供对AOP的支持。

如果将Aop理解为一种类的横向分解技术，那么其实在AOP之前，有很多AOP-like的实现，比如j2ee的过滤器链。当然，这些都是静态的，java动态代理技术提供了一个更加优雅的实现。

因为大部分代码都是相通的，所以本文只以前置增强为例。在本例中，BeanB实现了InterfaceB的run方法，我们期望BeanB在执行run方法前，先执行BeanC的run方法。

## 相关的标签类

为了描述清楚BeanB和BeanC的关系，我们借鉴spring的aop标签，定义以下标签类（省略set和get方法）：
    
    // 增强
    public class Advice {
    	protected String method;   // 增强pointcut方法的方法
    	protected String pointcut; // 表示一个类的方法
    }
    // 前置增强
    public class Before extends Advice{
    }
    public class Aspect {
    	private String ref;
    	private String id;
    	private List<Advice> advices = new ArrayList<Advice>();
	}
    // 配置文件
    <bean id="beanC" class="org.lqk.lspring.bean.BeanC"></bean>
	<aspect ref="beanC">
		<before method="before" pointcut="org.lqk.lspring.bean.BeanB.run()"></before>
	</aspect>
	
AOP说的直观点，就是将本来在一个方法里的代码放在多个方法里，然后将多个方法组织在一起，"如何组织"交给配置文件或专门的类。这个"如何组织"其实有多种方案。

1. 以beanC类（增强类）为主，描述beanC的某个方法会被用在哪里（类似于上面的配置文件）
2. 以beanB类（目标类）为主，描述beanB的某个方法需要如何增强（可惜spring的作者偏重第一种方式，大家只有以人家的喜好为喜好了。从本文的实现方式看，这种方式更好些）

## AOPBeanProcessor

为支持了AOP技术，我们使用beanFactory.getBean("beanB")就不能返回beanB实例本身了，而是一个代理后的实例。（beanId与实例的映射保存在beanId2Class map中）那么，我们什么时候替换beanId2Class中"beanB"对应的实例呢？

结合java动态代理的相关代码，我们可以知道，代理类的实例也要在类本身实例的基础上生成，因此前文中生成类本身实例的代码可以复用。我们在一个新的processor中生成代理类，并替换beanId2Class中的实例。

    public class AOPBeanProcessor extends AbstractBeanProcessor {
    	// 要切入的方法与切入类和方法的映射
    	private Map<String, Map<String, List<Advice>>> pointcut2Aspect = new HashMap<String, Map<String, List<Advice>>>();
    	public AOPBeanProcessor(AbstractBeanFactory beanFactory) {
    		super(beanFactory);
    		parseAspect();
    	}
    	private void parseAspect() {
    		List<Aspect> aspects = this.beans.getAspects();
    		for (Aspect aspect : aspects) {
    			List<Advice> advices = aspect.getAdvices();
    			for (Advice advice : advices) {
    				String pointcut = advice.getPointcut();
    				String ref = aspect.getRef();
    				Map<String, List<Advice>> aspect2Advice = pointcut2Aspect.get(pointcut);
    				if (MapUtils.isEmpty(aspect2Advice)) {
    					aspect2Advice = new HashMap<String, List<Advice>>();
    					pointcut2Aspect.put(pointcut, aspect2Advice);
    				}
    				List<Advice> advcs = aspect2Advice.get(ref);
    				if (CollectionUtils.isEmpty(advcs)) {
    					advcs = new ArrayList<Advice>();
    					aspect2Advice.put(ref, advcs);
    				}
    				advcs.add(advice);
    			}
    		}
    	}
    	@Override
    	public void process(String beanId) throws Exception {
    		Class clazz = beanId2Clazz.get(beanId);
    		for (String pointcut : pointcut2Aspect.keySet()) {
    		    // 如果该实例需要增强
    			if (pointcut.startsWith(clazz.getName())) {
    				Map<String, List<Advice>> aspect2Advice = pointcut2Aspect.get(pointcut);
    				Object obj = beanId2Class.get(beanId); // 元对象
    				ProxyHandler proxyHandler = new ProxyHandler(obj, aspect2Advice);
    				Object proxyObj = Proxy.newProxyInstance(clazz.getClassLoader(),
    						clazz.getInterfaces(), proxyHandler);
    			    // 使用代理类实例替换
    				beanId2Class.put(beanId, proxyObj);
    				break;
    			}
    		}
    	}
    	public class ProxyHandler implements InvocationHandler {
    		private Object proxied;	// 元对象
    		private Map<String, List<Advice>> aspect2Advice;
    		public ProxyHandler(Object proxied, Map<String, List<Advice>> aspect2Advice) {
    			this.proxied = proxied;
    			this.aspect2Advice = aspect2Advice;
    		}
    		public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
    			// 处理前置方法
    			for (String ref : aspect2Advice.keySet()) {
    				List<Advice> advices = aspect2Advice.get(ref);
    				// 获取增强类的实例
    				Object adviceObj = beanFactory.getBean(ref);
    				// 执行配置的所有前置增强方法
    				for (Advice advice : advices) {
    					if (advice instanceof Before) {
    						ReflectUtil.invokeMethod(adviceObj, advice.getMethod(), null, null);
    					}
    				}
    			}
    			// 执行具体目标对象的方法
    			return method.invoke(proxied, args);
    			// 在转调具体目标对象之后，可以执行一些功能处理
    		}
    	}
    }

在前文中，我们使用beanId2Bean来组织从配置文件中得到的信息，为的就是根据beanId可以快速找到类相关的元信息。类似的，在AOPBeanProcessor中，我使用pointcut2Aspect，为的也是根据输入的beanId能够快速确定该实例需要哪些增强。剩下的就是按照java动态代理的相关逻辑，生成代理类实例。

## 类之间的关系图

    org.lqk.lspring.framework
        BeanOperator
        DisposableBean
        FactoryBean
        InitializingBean
    org.lqk.lspring.factory
        AbstractBeanFactory
        BeanFactory
        AnnotationBeanFactory
        XmlBeanFactory
        FacadeBeanFactory
    org.lqk.lspring.processor
        AbstractBeanProcessor
        BeanProcessor
        DisposableBeanProcessor
        InitializingBeanProcessor
        FactoryBeanProcessor
        AOPBeanProcessor
    org.lqk.lspring.annotation
        Component
        Value
    org.lqk.lspring.tag
        Beans
        Bean
        Scan
        Property
        Advice
        Before
        Aspect
    org.lqk.lspring.bean
        BeanA
        BeanB
        InterfaceB
        BeanC
    org.lqk.lspring.util
        StringUtil
        ReflectUtil