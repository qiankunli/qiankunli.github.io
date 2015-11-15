---

layout: post
title: 自己动手写spring（三） 支持注解方式
category: 技术
tags: Java
keywords: Java Spring annotation

---

## 前言

本文使用注解方式来读取bean类，并解决bean之间的依赖。其中Component和Value注解直接copy自Spring，Resource注解来自javax。

## 扫描包中的所有类

还是以前文提到的beanA和beanB两个model类为例（省略set和get方法）。

    @Component
    public class BeanA {
    	@Value("studentA")
    	private String title;
    	@Resource
    	private BeanB beanB;
    }
    
    @Component
    public class BeanB {
    	@Value("studentB")
    	private String title;    	
    }

根据前文，我们可以发现，要实现bean工厂功能，首先要将bean信息加载到内存（由配置文件方式转化为“以类的方式”存储）。
在本文中，虽然配置文件没了，但思路是一致的。我们要根据注解，采集bean信息，存储在map中。注解和配置文件方式的一个不同是：配置文件是配什么bean加载什么，而注解方式，则需要我们扫描目标pacakge中的所有bean。


    public class Main {
    	private static Map<String,Class> beanId2Clazz = new HashMap<String,Class>();
    	// 将一个字符串的首字符小写
    	private static String smallCaseFirstChar(String prop){
    		char ch = (char) (prop.charAt(0) + 32);
    		return prop.replaceFirst(prop.charAt(0) + "", ch + "");
    	}
    	public static void main(String[] args) throws Exception {
    		// 扫描包中包含的类（即扫描目的包所在文件的子文件）
    		String packageName = "org.lqk.lspring.bean";
    		String rootPath = Main3.class.getResource("/").getPath();
    		File file = new File(rootPath + File.separator +  packageName.replace(".", File.separator));
    		String[] fileNames = file.list();
    		// 扫描
    		for(String fileName : fileNames){
    			String className = fileName.substring(0,fileName.length() - ".class".length());
    			String fullClassName = packageName + "." + className;
    			Class clazz = Class.forName(fullClassName);
    			Component cop = (Component) clazz.getAnnotation(Component.class);
    			if(null != cop){
    				String beanId = cop.value();
    				if(StringUtils.isEmpty(beanId)){
    					beanId = smallCaseFirstChar(className);
    				}
    				beanId2Clazz.put(beanId, clazz);
    			}
    		}
    	}
    }
    
关于扫描一个包中所有的类，有现成的`org.reflections`包，此处为了减少##读者的理解曲线，就不提了。

当加载完bean的信息后，整个步骤已经跟前文很像了。


## 加载beanB类

    public class Main {
    	private static Map<String,Class> beanId2Clazz = new HashMap<String,Class>();
    	public static void load() throws ClassNotFoundException {
    	    // 即为上述的main类
    	}
    	public static Object getBean(String beanId) throws Exception{
    		Class clazz = beanId2Clazz.get(beanId);
    		if(null == clazz){
    			return null;
    		}
    		Object obj = clazz.newInstance();
    		Field[] fields = clazz.getDeclaredFields();
    		// 处理带value注解的属性
    		for(Field field : fields){
    			Value v = field.getAnnotation(Value.class);
    			if(null == v){
    				continue;
    			}
    			String propertyValue = v.value();
    			Method m = clazz.getMethod("set" + bigCaseFirstChar(field.getName()), String.class);
    			m.invoke(obj, propertyValue);
    		}
    		return obj;
    	}
    	public static void main(String[] args) throws Exception {
    		load();
    		BeanB beanB = (BeanB)getBean("beanB");
    		System.out.println(beanB.getTitle());
    	}
    }
    
## 加载beanA类

    public class Main2 {
    	private static Map<String,Class> beanId2Clazz = new HashMap<String,Class>();
    	public static void load() throws ClassNotFoundException {
             // 不再赘述
    	}
    	public static Object getBean(String beanId) throws Exception{
    		Class clazz = beanId2Clazz.get(beanId);
    		if(null == clazz){
    			return null;
    		}
    		Object obj = clazz.newInstance();
    		Field[] fields = clazz.getDeclaredFields();
    		for(Field field : fields){
    			Method m = clazz.getMethod("set" + bigCaseFirstChar(field.getName()), field.getType());
    			Value v = field.getAnnotation(Value.class);
    			// 处理value注解
    			if(null != v){
    				String propertyValue = v.value();
    				m.invoke(obj, propertyValue);
    			}
    			Resource r = field.getAnnotation(Resource.class);
    			// 处理resource注解
    			if(null != r){
    				String propertyBeanId = r.name();
    				if(StringUtils.isEmpty(propertyBeanId)){
    					propertyBeanId = smallCaseFirstChar(field.getType().getSimpleName());
    				}
    				// 递归处理
    				Object propertyObj = getBean(propertyBeanId);
    				m.invoke(obj, propertyObj);
    			}
    		}
    		return obj;
    	}
    	public static void main(String[] args) throws Exception {
    		load();
    		BeanA beanA = (BeanA)getBean("beanA");
    		System.out.println(beanA.getBeanB().getTitle());
    	}
    }

至此，我们已经完全使用注解方式创建了一个bean工厂，下文将会尝试把注解和配置文件两种方式整合到一块儿。

## 类之间的关系图

    org.lqk.lspring
        BeanUtil
        Main
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