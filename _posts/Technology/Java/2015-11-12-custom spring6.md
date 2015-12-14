---

layout: post
title: 自己动手写spring（六） 支持FactoryBean
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

为什么需要工厂bean？回答这个问题，首先要回答下为什么需要工厂模式？简单说，就是不用`new xxx()`了。用`new xxx()`有什么不好呢？简单说就是：不能适应变化。对于传统的`new xxx()`,假设这个代码有10个地方使用，那么当`xxx()`改变时，你便要更改10个地方的代码（好吧，我承认，借助于ide，这个优势不是很明显）。比较简单的方式就是，提供一个专门的类负责构造这个bean，任何对这个bean构造逻辑的改变都只在这个类里完成。**这提供了一个很好的封装机制，比如封装连接池（注入的一个连接是FactoryBean从连接池中获取到的）、Proxy、RMI和JNDI等**。

FactoryBean也是一个bean，也有生命周期（比如实现InitializingBean和DisposableBean）。只是创建这个bean的目的不是使用这个FactoryBean本身，而是使用它的getObject方法得到一个bean。

这决定了，对于FactoryBean的处理，我们能复用前文的大部分过程。只是在保存beanId与对象实例的映射时，要做一些特别的操作。

## 使用工厂bean来构建一个bean

背景提示：通过对配置文件和注解信息的解析，我们得到两个Map

1. beanId2Bean，beanId对xml tag对象Bean的映射
2. beanId2Clazz，beanId对Class对象的映射

并使用beanId2Class（beanId与对象实例的映射）来保存已经创建好的对象实例。我们在使用getBean获取对象实例时，需要检查下当前对象实例是否实现了FactoryBean接口，如果是，则需要进行对象实例的替换。

    public class BeanFactory{
        public Object getBean(String beanId) throws Exception {
    		Object obj = beanId2Class.get(beanId);
    		if (obj != null) {
    			return obj;
    		}
    		// 尝试从配置文件中获取对象实例
    		obj = getBeanFromXml(beanId);
    		// 尝试从注解中获取对象实例（或者补充设置配置文件中对象实例的属性）
    		obj = getBeanFromAnnotation(beanId);
    		// 执行lspring管理的bean的init-method和afterPropertiesSet方法
    		InitializingBeanProcessor.process(obj, beanId2Bean.get(beanId),beanId2Clazz.get(beanId));
    		// 处理FactoryBean（如果beanId目前对应的实例是FactoryBean的话）
    		FactoryBeanProcessor.process(obj, beanId2Bean.get(beanId), beanId2Clazz.get(beanId),beanId2Class, beanId2Bean,beanId2Clazz);
    		return beanId2Class.get(beanId);
    	}
	}

    public class FactoryBeanProcessor {
	    public static void process(Object obj,Bean bean,Class clazz,Map<String, Object> beanId2Class,  Map<String, Bean> beanId2Bean,Map<String, Class> beanId2Clazz)
			throws Exception {
		Class[] intfs = clazz.getInterfaces();
    		for (Class intf : intfs) {
    			if("FactoryBean".equals(intf.getSimpleName())){
    				// 执行getObject方法拿到真实的对象实例
    				Method m = clazz.getMethod("getObject", null);
    				Object targetObj = m.invoke(obj, null);
    				// 替换beanId对应的对象
    				beanId2Class.put(bean.getId(), targetObj);
    				// 执行getObjectType方法拿到真实的对象type
    				Method m2 = clazz.getMethod("getObjectType", null);
    				Object targetClazz = m2.invoke(obj, null);
    				// 替换beanId中的clazz
    				beanId2Clazz.put(bean.getId(), (Class)targetClazz);
    				// 暂时先不处理Bean对象
    				beanId2Bean.remove(bean.getId());				
    				break;
    			}
    		}
	    }
	}
    


## 工厂bean本身的生命周期

我们提到，FactoryBean作为lspring管理的一个Bean，本身也是具备生命周期的。因为InitializingBeanProcessor.process位于FactoryBeanProcessor.process之前，这确保了FactoryBean的afterPropertiesSet可以被执行。可现在beanId2Clazz中beanId对应的实例已经被替换，FactoryBean的destroy方法如何被执行呢？DisposableBeanProcessor.process方法的执行基于beanId2Class、beanId2Bean和beanId2Clazz，因此，我们要为FactoryBean建立一套新的映射关系。


    public class FactoryBeanProcessor {
	    public static void process(Object obj,Bean bean,Class clazz,Map<String, Object> beanId2Class,  Map<String, Bean> beanId2Bean,Map<String, Class> beanId2Clazz)
			throws Exception {
    		Class[] intfs = clazz.getInterfaces();
    		for (Class intf : intfs) {
    			if("FactoryBean".equals(intf.getSimpleName())){
    			    // 省略其它代码
    				// 对于工厂bean，InitializingBean方法执行的时候，beanId还不是factoryBeanId
    				// 这样不影响工厂bean本身的DisposableBean接口的处理（InitializingBean接口已处理过）
    				String factoryBeanId = StringUtil.lowerCaseFirstChar(clazz.getSimpleName());
    				beanId2Clazz.put(factoryBeanId, clazz);
    				beanId2Class.put(factoryBeanId, obj);
    				beanId2Bean.put(factoryBeanId, bean);
    				break;
    			}
    		}
    	}
    }

## 小结

因为目前支持的特性较多，BeanFactory已经非常臃肿，下文我们将目前涉及到的所有类进行架构的上的调整，比如将BeanFactory部分方法剥离出来。

## 类之间的关系图

    org.lqk.lspring.framework
        BeanFactory
        DisposableBean
        DisposableBeanProcessor
        FactoryBean
        FactoryBeanProcessor
        InitializingBean
        InitializingBeanProcessor
    org.lqk.lspring.annotation
        Component
        Value
    org.lqk.lspring.tag
        Beans
        Bean
        Scan
        Property
    org.lqk.lspring.bean
        BeanA
        BeanB
    org.lqk.lspring.util
        StringUtil
        ReflectUtil