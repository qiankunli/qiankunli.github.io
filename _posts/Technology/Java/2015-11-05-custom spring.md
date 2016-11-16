---

layout: post
title: 自己动手写spring（一） 使用digester
category: 技术
tags: Java
keywords: Java Spring

---

## 前言

本来想熟悉下digester的使用，写着写着发现，可以搞一个类似spring的东西的来，将过程记录下来，与大家分享。

例子中很多代码没有优化，基本没有异常处理和判空操作，这是我的一个坏习惯，先奔着目的去，实现成功后再重构。但这样的代码，更易看懂，优化后的代码反而掩盖了很多思路和思想，尤其容易让人分不清重点（因为算是技巧的地方太多了）。

项目源码的git clone地址：`git@github.com:qiankunli/lspring.git`

## 使用digester读取一个配置文件

digester入门的例子通常是

    <?xml version="1.0" encoding="UTF-8"?>
    <books>
    	<book>
    		<author>Author 2</author>
    		<title>His One Book</title>
    	</book>
    </books>


digester代码

    public class BookMain {
    	public static void main(String[] args) throws IOException, SAXException {
    		  Digester digester = new Digester();
              digester.setValidating(false);
              // 创建对象
              digester.addObjectCreate("books", Books.class);
              digester.addObjectCreate("books/book", Book.class);
              // 设置属性
              digester.addBeanPropertySetter("books/book/author", "author");
              digester.addBeanPropertySetter("books/book/title", "title");
              // 处理对象之间的关系
              digester.addSetNext("books/book", "addBook");
              InputStream in = ClassLoader.getSystemResourceAsStream("books.xml");
              Books bs = (Books) digester.parse(in);
              System.out.println(bs.getBooks().get(0));
    	}
    }
    
这个xml文件的配置跟我们常用的不太一样，很明显，常见的是下面这种：

    <books>
    	<book author="author" title="title" class="org.lqk.digester.book.Book"/>
    </books>
    
对应的处理类

    public class BookMain2 {
    	public static void main(String[] args) throws IOException, SAXException {
    		  Digester digester = new Digester();
              digester.setValidating(false);
              digester.addObjectCreate("books", Books.class);
              digester.addObjectCreate("books/book", "org.lqk.digester.book.Book","class");
              digester.addSetProperties("books/book");
              digester.addSetNext("books/book", "addBook");
              InputStream in = ClassLoader.getSystemResourceAsStream("books2.xml");
              Books bs = (Books) digester.parse(in);
              System.out.println(bs.getBooks().get(0));
    	}
    }

这个跟常用的就像很多了。从这个例子中可以看到，根据配置文件，我们是可以搞出一些类来，并处理它们之间的关系。说的抽象点，我们可以将配置文件描述的信息，映射到jvm的类中。

## spring样式的配置文件

那对于一个spring样式的配置文件，兴许也能搞出点东西来。

    <beans>
    	<!-- 目前的局限 
    		1. 必须包括id和class两个属性
    	-->
    	<!-- <bean>标签中的name属性对Bean类有意义，但对beanA和beanB确没有意义 -->
    	<bean id="beanA" name="test" class="org.lqk.container.bean.BeanA">
    		<property name="beanB" ref="beanB"/>
    		<property name="title" value="studentA"/>
    	</bean>
    	<bean id="beanB" class="org.lqk.container.bean.BeanB">
    		<property name="title" value="studentB"/>
    	</bean>
    </beans>
    
我们先来尝试读取基本信息。首先是创建三个model，将标签信息对象化（你该不会想直接用digester生成beanA吧!）,model中省略set和get方法（但是必须要有的喔）。

    public class Beans {
    	private List<Bean> beans = new ArrayList<Bean>();
    	public void addBean(Bean bean){
    		this.beans.add(bean);
    	}
    }

    public class Bean {
    	private String className;
    	private String id;
    	private List<Property> props = new ArrayList<Property>();
    	public void addProperty(Property property){
    		this.props.add(property);
    	}
    }	
    
    public class Property {
    	private String name;
    	private String value;
    	private String ref;
    }

    public class Main {
    	public static void main(String[] args) throws IOException, SAXException {
    		Digester digester = new Digester();
    		digester.setValidating(false);
    		digester.addObjectCreate("beans", Beans.class);
    
    		// 如果配置文件中有多个bean，add一次即可
    		digester.addObjectCreate("beans/bean", Bean.class);
    
    		// 设置bean的属性<bean name="",id="">中的id和name。默认属性名和类中的属性名一样，不同的要特殊配置
    		digester.addSetProperties("beans/bean", "class", "className");
    		digester.addSetProperties("beans/bean");
    
    		digester.addObjectCreate("beans/bean/property", Property.class);
    		// 将标签中的属性赋给对象
    		digester.addSetProperties("beans/bean/property");
    
    		// 设置对象间的关系
    		digester.addSetNext("beans/bean/property", "addProperty");
    		digester.addSetNext("beans/bean", "addBean");
    
    		InputStream in = ClassLoader.getSystemResourceAsStream("beans.xml");
    		Beans beans = (Beans) digester.parse(in);
    		List<Bean> beanList = beans.getBeans();
    		for (Bean bean : beanList) {
    			System.out.println("bean =================================================>");
    			System.out.println("    id ==> " + bean.getId());
    			List<Property> props = bean.getProps();
    			for (Property prop : props) {
    				System.out.println("    property =================================================>");
    				System.out.println("        name ==> " + prop.getName());
    				System.out.println("        ref ==> " + prop.getRef());
    				System.out.println("        value ==> " + prop.getValue());
    			}
    		}
    	}
    }


现在信息从配置文件转到了jvm中，以类的形式存在。我们可以通过`Class.forName(className)`加载真正的beanA进入jvm。柿子捡软的捏，我们先创建一个beanB出来。

    public class Main2 {	
    	public static void main(String[] args) throws Exception {
    		// 省略上述代码
    		for (Bean bean : beanList) {
    			if("org.lqk.container.bean.BeanB".equals(bean.getClassName())){
    				// 加载类并创建对象
    				Class clazz = Class.forName("org.lqk.container.bean.BeanB");
    				Object obj = clazz.newInstance();
    				// 为对象属性赋值
    				List<Property> props = bean.getProps();
    				String methodName = "set" + bigCaseFirstChar(props.get(0).getName());
    				Method m = clazz.getMethod(methodName, String.class);
    				m.invoke(obj, props.get(0).getValue());
    				// 使用对象
    				BeanB beanB = (BeanB)obj;
    				System.out.println(beanB.getTitle());
    				break;
    			}
    		}
    	}
    	// 将该字符串的首字母大写
        public static String bigCaseFirstChar(String prop) {
    		char ch = (char) (prop.charAt(0) - 32);
    		return prop.replaceFirst(prop.charAt(0) + "", ch + "");
	    }
    }

我们真的弄了一个类出来，虽然对经常用spring的人来说有点大惊小怪，但亲身体验一把，还是很有意义的。

## 类之间的关系图

    org.lqk.lspring
        Main
    org.lqk.lspring.tag
        Beans
        Bean
        Property
    org.lqk.lspring.bean
        BeanA
        BeanB
    