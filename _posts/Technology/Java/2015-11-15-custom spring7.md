---

layout: post
title: 自己动手写spring（七） 类结构设计调整
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

前文的类设计比较简单粗暴，只为实现功能，本文对类设计做一些调整，避免一个类出现过多的代码。

## 分解BeanFactory

目前，BeanFactory主要有以下成员

    public Class BeanFactory{
        private Beans beans;
        // 实现id到bean对象的映射
    	private Map<String, Bean> beanId2Bean = null;
    	// 实现id到对象的映射
    	private Map<String, Object> beanId2Class = null;
    	// 实现id到clazz的映射
    	private Map<String, Class> beanId2Clazz = null;
    	private Thread shutdownHook;
    
        // 加载类信息
        public void loadXml();
        public void loadAnnotation();
        // 获取Bean对象
        pubic Object getBean(String beanId);
        pubic Object getBeanFromXml(String beanId);
        pubic Object getBeanFromAnnotation(String beanId);
        // 退出的时候，执行DisposableBeanProcessor.process
        public void registerShutdownHook();
        public void destroyBeans();
        
    }
    
简单的，我们可以将BeanFactory成员归为以下几类：

1. 实现BeanFactory功能的基本数据，比如beanId2Bean，beanId2Class和beanId2Clazz
2. 与xml相关的数据，比如beans，loadXml，getBeanFromXml
3. 与Annotation相关的数据，比如loadAnnotation，getBeanFromAnnotation
4. 其它功能，比如shutdownHook，registerShutdownHook和destroyBeans

很明显，第二类和第三类也可以说是一个Bean工厂，只不过其类信息的来源不同。因此，我们将BeanFactory一分为三

1. XmlBeanFactory      
2. AnnotationBeanFactory
3. FacadeBeanFactory，负责统筹XmlBeanFactory和AnnotationBeanFactory的使用

到这里，按照java的习惯，我们要定义一个BeanFactory接口，大家约定一个getBean方法。因为beanId2Bean，beanId2Class和beanId2Clazz三个map在三个BeanFactory中都会用到，所以定义一个AbstractBeanFactory来包含这三个成员。XmlBeanFactory、AnnotationBeanFactory和FacadeBeanFactory继承AbstractBeanFactory即可。

    public abstract class AbstractBeanFactory implements BeanFactory{
    	// 实现id到bean对象的映射
    	protected Map<String, Bean> beanId2Bean = null;
    	// 实现id到对象的映射
    	protected Map<String, Object> beanId2Class = null;
    	// 实现id到clazz的映射
    	protected Map<String, Class> beanId2Clazz = null;
    	public AbstractBeanFactory(){
    		beanId2Bean = new HashMap<String, Bean>();
    		beanId2Class = new HashMap<String, Object>();
    		beanId2Clazz = new HashMap<String, Class>();
    	}
    	// 省略set和get方法
    	public abstract Object getBean(String beanId) throws Exception;
    }

其中需要注意的是，对于XmlBeanFactory和AnnotationBeanFactory来说，在执行getBean方法时，会递归调用getBean方法（这个方法在语义上指的是FacadeBeanFactory的getBean方法）。因此，XmlBeanFactory和AnnotationBeanFactory应该维护一个FacadeBeanFactory的引用。以XmlBeanFactory为例

    public class XmlBeanFactory extends AbstractBeanFactory{
    	private AbstractBeanFactory parentBeanFactory = null;
    	public XmlBeanFactory(AbstractBeanFactory parentBeanFactory){
    		this.parentBeanFactory = parentBeanFactory;
    		this.beanId2Bean = parentBeanFactory.getBeanId2Bean();
		    this.beanId2Clazz = parentBeanFactory.getBeanId2Clazz();
		    this.beanId2Class = parentBeanFactory.getBeanId2Class();
    	}
    }

## 处理Processor类

Processor目前都是以工具类的形式存在，可以观察到其process方法都包含beanId2Bean，beanId2Class和beanId2Clazz参数，为了减少参数传递，我们提供BeanProcessor接口和AbstractBeanFactory抽象类。

    public abstract class AbstractBeanProcessor implements BeanProcessor{
        // 实现id到bean对象的映射
    	protected Map<String, Bean> beanId2Bean = null;
    	// 实现id到对象的映射
    	protected Map<String, Object> beanId2Class = null;
    	// 实现id到clazz的映射
    	protected Map<String, Class> beanId2Clazz = null;
    	public AbstractBeanProcessor(AbstractBeanFactory beanFactory){
    		this.beanId2Bean = beanFactory.getBeanId2Bean();
    		this.beanId2Class = beanFactory.getBeanId2Class();
    		this.beanId2Clazz = beanFactory.getBeanId2Clazz();
    	}
    	// 省略set和get方法
    	// BeanProcessor的两个方法
    	public abstract void process(String beanId) throws Exception;
    	public abstract void process() throws Exception;
    }
    
读者想必已经发现，AbstractBeanProcessor类的三个成员与AbstractBeanFactory的三个成员重复。因此，我们再提取出一个BeanOperator类，来容纳这三个成员及其set和get方法。于是，AbstractBeanProcessor变成了

    public abstract class AbstractBeanProcessor extends BeanOperator implements BeanProcessor{
    	public AbstractBeanProcessor(AbstractBeanFactory beanFactory){
    		this.beanId2Bean = beanFactory.getBeanId2Bean();
    		this.beanId2Class = beanFactory.getBeanId2Class();
    		this.beanId2Clazz = beanFactory.getBeanId2Clazz();
    	}
    	public abstract void process(String beanId) throws Exception;
    	public abstract void process() throws Exception;
    }

## 小结

我在拆分这段代码时，感觉真是痛快淋漓。为了提高框架的感觉，读者还可以自定义异常类，来封装处理过程中的各种错误。因为这还不是现阶段的重点，因此本文不再分析。

重新设计之后，虽然可以让类结构更加清晰，分工更加明确，但也掩盖了最初的想法和思路，导致代码不再直观。

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