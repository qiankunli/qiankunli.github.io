---

layout: post
title: 自己动手写spring（五） bean的生命周期管理
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

众所周知，使用Class的newInstance方法，就算是调用了bean的构造方法。bean是不能控制自己何年何月何日生（死）的，但bean可以和bean工厂约定一些规则（也就是实现特定的方法），决定下生死时刻要做些什么。

bean生命周期管理，主要涉及到标签中的init-method,destroy-method以及InitializingBean,DisposableBean接口。

## InitializingBean

调用时机，当一个bean构造完成的时候调用。对前文和反射概念有一定了解的童鞋来说，实现起来不难。

    public class BeanFactory {

    	public Object getBean(String beanId) throws Exception {
    		Object obj = beanId2Class.get(beanId);
    		if (obj != null) {
    			return obj;
    		}
    		obj = getBeanFromXml(beanId);
    		obj = getBeanFromAnnotation(beanId);
    		// 为了不给BeanFactory类增加负担，我们将这个逻辑提出来
    		InitializingBeanProcessor.process(obj, beanId2Bean.get(beanId),beanId2Clazz.get(beanId));
    		return obj;
    	}
    }
    
    public class InitializingBeanProcessor {
    	public static void process(Object obj, Bean bean,Class clazz) throws Exception {		
    		Class[] intfs = clazz.getInterfaces();
    		for(Class intf : intfs){
    			if(intf.getSimpleName().equals("InitializingBean")){
    				Method m = clazz.getMethod("afterPropertiesSet", null);
    				m.invoke(obj, null);
    				break;
    			}
    		}
    	}
    }

## init-method

    <bean id="beanA" name="test" class="org.lqk.lspring2.bean.BeanA" init-method="init" destroy-method="close">
		<property name="beanB" ref="beanB" />
		<property name="title" value="studentA" />
	</bean>
	
为bean标签类中添加init-method和destroy-method属性，并调用它们，也不是难事。

    public class InitializingBeanProcessor {
    	public static void process(Object obj, Bean bean,Class clazz) throws Exception {
    	    // 处理init-method标签
    		String initMethodName = bean.getInitMethodName();
    		if(StringUtils.isNotEmpty(initMethodName)){
    			Method m = clazz.getMethod(initMethodName, null);
    			m.invoke(obj, null);
    		}
    		// 处理InitializingBean的afterPropertiesSet方法
    		Class[] intfs = clazz.getInterfaces();
    		for(Class intf : intfs){
    			if(intf.getSimpleName().equals("InitializingBean")){
    				Method m = clazz.getMethod("afterPropertiesSet", null);
    				m.invoke(obj, null);
    				break;
    			}
    		}
    	}
    }
    
## DisposableBean和destroy-method

何时调用DisposableBean的方法和destroy-method，当然是系统运行结束退出的时候，但系统什么时候退出呢？这就用到了shutdownHook（这可是java基础喔，在spring之前就有喔）。

    public class BeanFactory {
        private Thread shutdownHook;
        public BeanFactory() {
    		try {
    			loadXml();
    			loadAnnotation();
    			registerShutdownHook();
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
	    }
        public void registerShutdownHook() {
    		if (this.shutdownHook == null) {
    			// No shutdown hook registered yet.
    			this.shutdownHook = new Thread() {
    				@Override
    				public void run() {
    					destroyBeans();
    				}
    			};
    			Runtime.getRuntime().addShutdownHook(this.shutdownHook);
    		}
    	}
    	protected void destroyBeans() {
    		try {
    			DisposableBeanProcessor.process(beanId2Class, beanId2Bean,beanId2Clazz);
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
    	}
    }
    
DisposableBeanProcessor与InitializingBeanProcessor的逻辑类似，此处不在赘述。

下文，我们将讨论工厂bean的实现。