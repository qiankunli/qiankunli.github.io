---

layout: post
title: 自己动手写spring（二） 创建一个bean工厂
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

前文虽然以spring配置文件的样式，根据配置文件搞了一个类出来，但离工具化的应用还很远，本文再进一步。

## 处理类之间的关系

前文的例子只创建了较为简单的beanB类，现在我们创建beanA类。从模仿spring beanFactory的getBean方法开始。getBean方法体现了spring的一个点：bean只有在使用时才会真正被创建。

getBean的基本思路是，根据bean的“元信息”创建bean的实例，所以这个“元信息”必须提前从配置文件中采集并加载到内存中，此处，我们以map的形式来管理它。

    public class Main3 {
        // 根据beans类构建id到bean对象（注意不是beanA和beanB）的映射。不然根据id获取Bean对象太麻烦
	    public static Map<String, Bean> beanMap = new HashMap<String, Bean>();
	    public static void load(){
	        // 前文的所有代码
	    }
        public static Object getBean(String beanId) throws Exception {
    		Object obj = null;
    		Bean bean = beanMap.get(beanId);
    		// 创建类对象
    		Class clazz = Class.forName(bean.getClassName());
    		obj = clazz.newInstance();
    		List<Property> props = bean.getProps();
    		for (Property prop : props) {
    			String methodName = "set" + bigCaseFirstChar(prop.getName());
    			// 为一般属性赋值
    			if (null != prop.getValue()) {
    				Method m = clazz.getMethod(methodName, String.class);
    				m.invoke(obj, prop.getValue());
    			}
    			// 为对象属性赋值
    			if (null != prop.getRef()) {
    				Bean chileBean = beanMap.get(prop.getRef());
    				Class childClazz = Class.forName(chileBean.getClassName());
    				Method m = clazz.getMethod(methodName, childClazz);
    				// 递归调用
    				Object childObj = getBean(chileBean.getId());
    				m.invoke(obj, childObj);
    			}
    		}
    		return obj;
    	}
    	public static void main(String[] args) {
    		try {
    			load();
    			System.out.println("test=================================================>");
    			BeanA beanA = (BeanA) getBean("beanA");
    			System.out.println(beanA.getTitle());
    			System.out.println(beanA.getBeanB().getTitle());
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
	    }
    }
    
是不是已经有spring的感觉啦，不过，我们还需要将创建好的bean保存起来，同时，将上述代码包装为一个util类。这里，比较重要的一个思想就是：Bean是执行getBean时，才开始尝试构建的。

    public class BeanUtil {
        public static Beans beans;
    	// 保存id到bean对象的映射
    	public static Map<String, Bean> beanMap = new HashMap<String, Bean>();
    	// 保存id到对象的映射，这个map就很有容器的意思啦
    	public static Map<String, Object> objMap = new HashMap<String, Object>();
    	static {
    		try {
    			load();
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
    	}
    	public static Object getBean(String beanId) throws Exception {
    	    // 先看下map中有没有
    		Object obj = objMap.get(beanId);
    		if (obj != null) {
    			return obj;
    		}
    		// 跟上面代码重复，省略
    		if (obj != null) {
    			objMap.put(beanId, obj);
    		}
    		return obj;
    	}
    	public static void main(String[] args) {
    		try {
    			System.out.println("test=================================================>");
    			BeanA beanA = (BeanA) BeanUtil.getBean("beanA");
    			System.out.println(beanA.getTitle());
    			System.out.println(beanA.getBeanB().getTitle());
    		} catch (Exception e) {
    			e.printStackTrace();
       		}
	    }
    }

这个beanUtil就是一个简单的bean工厂啦，有时间，我提供下支持注解的版本。
