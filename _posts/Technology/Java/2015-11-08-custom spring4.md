---

layout: post
title: 自己动手写spring（四） 整合xml与注解方式
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

本文总结前文的内容，将两种方式整合在一起。

再次强调，要实现bean工厂功能，首先要将bean信息加载到内存，由配置文件或注解方式转化为“以类的方式”存储，并以map的形式组织起来（为方便查询）。具体的说就是

    public class BeanFactory {
    	private Beans beans;
    	// 实现id到bean对象的映射
    	private Map<String,Bean> beanId2Bean = new HashMap<String,Bean>();
    	// 实现id到clazz的映射
    	private Map<String,Class> beanId2Clazz = new HashMap<String,Class>();
    	
    	// 实现id到对象的映射，保存创建好的bean实例
    	private Map<String, Object> beanId2Class = new HashMap<String, Object>();
    	
    	public BeanFactory() {
    		try {
    			loadXml();
    			loadAnnotation();
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
    	}
    	// getBean时，先尝试从xml中获取一下，如果没有，再尝试从注解信息中获取一下  	
        public Object getBean(String beanId) throws Exception{
    		Object obj = beanId2Class.get(beanId);
    		if (obj != null) {
    			return obj;
    		}	
    		obj = getBeanFromXml(beanId);
    		if(obj == null){
    			obj = getBeanFromAnnotation(beanId);
    		}
    		if (obj != null) {
    			beanId2Class.put(beanId, obj);
    		}
		    return obj;
	    }
    }

代码中的loadXml,loadAnnotation,getBeanFromXml,getBeanFromAnnotation可从前文的load和getBean方法中得到。

## 增加tag类

前文中，扫描使用注解类的包的包名是写死在代码中，此处将其写在配置文件中。为此，要调整一些标签类（省略get和set方法）。

    // 添加scan类
    public class Scan {
    	private String packageName;	
    }
    调整Beans类
    public class Beans {
    	private List<Bean> beans = new ArrayList<Bean>();
    	// 事实上可能有多个scan，但多个scan类已不是难点所在，此处不再赘述
    	private Scan scan;
    }
    // beans.xml配置文件
    <?xml version="1.0" encoding="UTF-8"?>
    <beans>
    	<bean id="beanA" name="test" class="org.lqk.lspring1.bean.BeanA">
    		<property name="beanB" ref="beanB"/>
    		<property name="title" value="studentA"/>
    	</bean>
    	<scan package="org.lqk.lspring1.bean"/>
    </beans>
    
## 到此为止了么？

在示例中，beanA使用了xml文件方式，beanB使用了注解方式，BeanFactory正常工作啦。但旋即，笔者注意到一个问题，**粒度**。目前的粒度，是以Bean为单位，一个Bean要么全注解方式，要么全配置文件方式。但经常使用spring的人，会知道有一种情况：在配置文件中声明bean，然后在类中使用`@autowire`注入属性，因此我们要调整getBean方法。

	public Object getBean(String beanId) throws Exception{
		Object obj = beanId2Class.get(beanId);
		if (obj != null) {
			return obj;
		}	
		// 此处，不再一判空就返回
		obj = getBeanFromXml(beanId);
		obj = getBeanFromAnnotation(beanId);
		return obj;
	}
	
    public Object getBeanFromAnnotation(String beanId) throws Exception{
		Class clazz = beanId2Clazz.get(beanId);
		if(null == clazz){
			return null;
		}
		Object obj = beanId2Class.get(beanId);
		// getBeanFromXml没有构建该bean时，创建bean。否则，复用bean。
		if (obj == null) {
			obj = clazz.newInstance();
		}
		// 省略前文中getBean代码
	}	

## 其它

到这里，我们就实现了spring的基本功能之一，依赖注入。其实主要就是两个步骤：

1. 将元信息从配置文件和注解中加载到类中
2. 根据元信息，使用反射构建bean


当然，按照代码重构的逻辑，BeanFactory的很多方法，可以独立为一个组件，这样BeanFactory就不会很臃肿。

在下文中，我们将提供对bean生命周期的管理。

## 类之间的关系图

    org.lqk.lspring.framework
        BeanFactory
    org.lqk.lspring.annotation
        Component
        Value
    org.lqk.lspring.tag
        Beans
        Bean
        Property
        Scan
    org.lqk.lspring.bean
        BeanA
        BeanB