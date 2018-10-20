---

layout: post
title: mybatis学习
category: 技术
tags: Data
keywords: mysql mybatis

---

## 前言

What is MyBatis?

MyBatis is a first class persistence framework with support for custom SQL, stored procedures and advanced mappings. MyBatis eliminates almost all of the JDBC code and manual setting of parameters and retrieval of results. MyBatis can use simple XML or Annotations for configuration and map primitives, Map interfaces and Java POJOs (Plain Old Java Objects) to database records.

1. 一般业务模型直接体现在数据库表设计，表设计好了，页面设计确认了，剩下的code就是一个匹配的过程。我以前用持久化框架，都是先创建数据库表，一直忽视从class persistence framework 的角度来看问题，这才是实际的数据流动的视角。
2. 基本隐藏了jdbc 实现，条件查询（包括结果返回）也简化了很多
3. 使用xml 或 xml 控制java 类和 database record 的映射过程


[Mybatis framework learning](http://www.codeblogbt.com/archives/303221)

1. Mybatis It is characterized by the use of Xml files to configure SQL statements. Programmers write their own SQL statements. When the requirements change, we only need to modify the configuration files, which is more flexible. 提取 xml  来隔离sql 的变化

2. In addition, Mybatis transfer to the SQL statement can be passed directly into the Java object without the need for a parameter to correspond with the placeholder; the query result set is mapped to a Java object.  sql 的输入输出 都可以是对象

![](/public/upload/data/mybatis_framework.jpg)

1. The global configuration file SqlMapConfig.xml configuring the running environment of Mybatis. The Mapper.xml file is the SQL map file, and the SQL statement in the file is configured in the file. This file needs to be in SqlMLoad in apConfig.xml.
2. SqlSessionFactory is constructed through configuration information such as Mybatis environment, that is, a conversation factory. 这个图的走势很形象，尤其是从配置文件 建builder的感觉。
3. SqlSession is created by a conversational factory, and the operation database needs to be done through SqlSession.
4. Mybatis The bottom layer customize the Executor actuator interface to operate the database, and the Executor executor calls the specific MappedStatement object to perform the database operation.
5. MappedStatement Mybatis is also a low-level encapsulation object, which encapsulates Mybatis configuration information and SQL mapping information. A SQL in the Mapper.xml file corresponds to a Mapped Statement object, SQL.ID is the ID of the Mapped statement.
6. Executor The input Java object is mapped to SQL by MappedStatement before the SQL is held, and the meaning of the input parameter mapping is to set the parameters for PreparedStatement in JDBC programming. ExecutOr maps the output results to the Java object after executing SQL through the MappedStatement, and the output result mapping process is equivalent to the resolution process for the results in JDBC programming.


## 直接感受

mybatis 最重要的接口是 SqlSession，它是应用程序与持久层之间执行交互操作的一个单线程对象，具体的说

1. SqlSession对象完全包含以数据库为背景的所有执行SQL操作的方法,它的底层封装了JDBC连接,可以用SqlSession实例来直接执行被映射的SQL语句.
2. SqlSession是线程不安全的，每个线程都应该有它自己的SqlSession实例.
3. 使用完SqlSeesion之后关闭Session很重要,应该确保使用finally块来关闭它.

	public interface SqlSession extends Closeable {
  		<T> T selectOne(String statement);
  		<T> T selectOne(String statement, Object parameter);
  		int insert(String statement);
  		int insert(String statement, Object parameter);
  		int delete(String statement);
  		int delete(String statement, Object parameter);
  	}
  	
  	
从操作接口可以看到，基本是符合class persistence framework 的感觉的

1. 输入是statement（sql 的id），对象参数
2. 返回值是对象
3. framework 负责屏蔽sql 的变化，输入数据的序列化，输出数据的反序列化。（一不留神有点io通信框架的感觉了）


SqlSession  由 SqlSessionFactory 创建

	public interface SqlSessionFactory {
	  	SqlSession openSession();//这个方法最经常用,用来创建SqlSession对象.
	  	SqlSession openSession(boolean autoCommit);
	  	SqlSession openSession(Connection connection);
	  	SqlSession openSession(TransactionIsolationLevel level);
	  	SqlSession openSession(ExecutorType execType);
	  	SqlSession openSession(ExecutorType execType, boolean autoCommit);
	  	SqlSession openSession(ExecutorType execType, TransactionIsolationLevel level);
	  	SqlSession openSession(ExecutorType execType, Connection connection);
	  	Configuration getConfiguration();
	}

![](/public/upload/data/mybatis_work.png)

关于SqlSessionFactory和SqlSession两个对象给一个具体的使用过程: 

	public class TestQuickStart {
	    @Test
	    public void testQueryBlogById() throws Exception {
	        // 1. Load the SqlMapConfig.xml configuration file
	        InputStream inputStream = Resources.getResourceAsStream("SqlMapConfig.xml");
	        // 2. Create SqlSessionFactoryBuilder objects
	        SqlSessionFactoryBuilder sqlSessionFactoryBuilder = new SqlSessionFactoryBuilder();
	        // 3. Create SqlSessionFactory objects
	        SqlSessionFactory sqlSessionFactory = sqlSessionFactoryBuilder.build(inputStream);
	        // 4. Create SqlSession objects
	        SqlSession sqlSession = sqlSessionFactory.openSession();
	        // 5. Execute SqlSession object execution query and get result Blog13         // The first parameter is the ID of statement of BlogMapper.xml. The second parameter is the parameter needed to execute SQL.
	        Blog blog = sqlSession.selectOne("queryBlogById", 1);
	        // 6. Print results
	        System.out.println(blog);
	        // 7. Release resources
	        sqlSession.close();
	    }
	}

从中可以看到：

1. mybatis 最重要的是SqlSessionFactory和SqlSession两个对象， 尤其以SqlSession 最为核心，其interface 定义最体现了class persistence framework 的精髓
2. 很多框架，在实现其基本逻辑后， 一般会提供给一个xx-spring 封装以与spring 整合。所以在学习mybatis 时，**我们自己要分清楚哪些是框架本身的部分（里儿），哪些是皮儿。**

## 再进一步


作者在做hibernate 与mybatis 的对比时提到：

1. Hibernate The degree of automation is high, it only needs to configure OR mapping relations, and does not need to write Sql statements. hibernate完全可以通过对象关系模型实现对数据库的操作，拥有完整的JavaBean对象与数据库的映射结构来自动生成sql。而mybatis仅有基本的字段映射，对象数据以及对象实际关系仍然需要通过手写sql来实现和管理。
2. Mybatis Although mapping relations between Sql and Java objects, programmers need to write Sql themselves.


github [baomidou/mybatis-plus](https://github.com/baomidou/mybatis-plus)重点：

1. 只做增强不做改变，这一点对理解 mybatis的使用很重要
2. 启动即会自动注入基本 CURD，Mapper 对应的 XML 支持热加载，对于简单的 CRUD 操作，甚至可以无 XML 启动
3. 内置代码生成器：采用代码或者 Maven 插件可快速生成 Mapper（替换原来的dao） 、 Model 、 Service 、 Controller 层代码
4. 支持 Lambda 形式调用


