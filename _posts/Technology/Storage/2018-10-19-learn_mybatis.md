---

layout: post
title: mybatis学习
category: 技术
tags: Storage
keywords: mysql mybatis

---

## 前言

What is MyBatis?

MyBatis is a first class persistence framework with support for custom SQL, stored procedures and advanced mappings. MyBatis eliminates almost all of the JDBC code and manual setting of parameters and retrieval of results. MyBatis can use simple XML or Annotations for configuration and map primitives, Map interfaces and Java POJOs (Plain Old Java Objects) to database records.

1. 一般业务模型直接体现在数据库表设计，表设计好了，页面设计确认了，剩下的code就是一个匹配的过程。我以前用持久化框架，都是先创建数据库表，一直忽视从class persistence framework 的角度来看问题，这才是实际的数据流动的视角。
2. 基本隐藏了jdbc 实现，条件查询（包括结果返回）也简化了很多
3. 使用xml 或 xml 控制java 类和 database record 的映射过程

## 使用手感

原生mybatis 代码

```java
String resource = "mybatis/mybatis-config.xml";
InputStream inputStream = Resources.getResourceAsStream(resource);
SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
try (SqlSession session = sqlSessionFactory.openSession()) {
    List<User> users = session.selectList("mybatis.mapper.UserMapper.getAll");
    System.out.println(String.format("result:%s", users));
}
```
```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE configuration
  PUBLIC "-//mybatis.org//DTD Config 3.0//EN" "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
  <environments default="development">
    <environment id="development">
      <transactionManager type="JDBC"/>
      <dataSource type="POOLED">
        <property name="driver" value="com.mysql.jdbc.Driver"/>
        <property name="url" value="jdbc:mysql://localhost:3306/test"/>
        <property name="username" value="root"/>
        <property name="password" value="root"/>
      </dataSource>
    </environment>
  </environments>
  <mappers>
    <mapper resource="mybatis/mapper/UserMapper.xml"/>
  </mappers>
</configuration>
```
## 整体架构

![](/public/upload/data/mybatis_work.png)

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

## 源码分析

![](/public/upload/java/mybatis_object.png)

mybatis 最重要的接口是 SqlSession，它是应用程序与持久层之间执行交互操作的一个单线程对象，具体的说

1. SqlSession对象完全包含以数据库为背景的所有执行SQL操作的方法,它的底层封装了JDBC连接,可以用SqlSession实例来直接执行被映射的SQL语句.
2. SqlSession是线程不安全的，每个线程都应该有它自己的SqlSession实例.
3. 使用完SqlSeesion之后关闭Session很重要,应该确保使用finally块来关闭它.

自上而下，逐渐分解

|抽象|参数|方法|备注|
|---|---|---|---|
|SqlSession|mapper xml 文件中的statement  id|增删改查|
|Executor|MappedStatement, 根据parameterObject 可以得到真正待执行的BoundSql|query/update|额外处理缓存、批量等逻辑|
|StatementHandler|java.sql.Statement|query/update|

![](/public/upload/java/mybatis_select.png)

```java
// DefaultSqlSession
public void select(String statement, Object parameter, RowBounds rowBounds, ResultHandler handler) {
    try {
        MappedStatement ms = configuration.getMappedStatement(statement);
        executor.query(ms, wrapCollection(parameter), rowBounds, handler);
    } catch (Exception e) {
        throw ExceptionFactory.wrapException("Error querying database.  Cause: " + e, e);
    } finally {
        ErrorContext.instance().reset();
    }
}
// SimpleExecutor
public <E> List<E> doQuery(MappedStatement ms, Object parameter, RowBounds rowBounds, ResultHandler resultHandler, BoundSql boundSql) throws SQLException {
    Statement stmt = null;
    try {
        Configuration configuration = ms.getConfiguration();
        StatementHandler handler = configuration.newStatementHandler(wrapper, ms, parameter, rowBounds, resultHandler, boundSql);
        stmt = prepareStatement(handler, ms.getStatementLog());
        return handler.query(stmt, resultHandler);
    } finally {
        closeStatement(stmt);
    }
}
// SimpleStatementHandler
public <E> List<E> query(Statement statement, ResultHandler resultHandler) throws SQLException {
    String sql = boundSql.getSql();
    statement.execute(sql);
    return resultSetHandler.handleResultSets(statement);
}
```

## 学到的技巧

StatementHandler 包含多个子类，分别对应不同的场景。一般情况下 会使用工厂模式，mybatis 实现了一个RoutingStatementHandler（上层代码无需关心具体实现），根据参数 创建一个delegate 实现 并将请求转发给它。 **是一个用对象 替换if else 的典型实现**。

```java
public class RoutingStatementHandler implements StatementHandler {
    private final StatementHandler delegate;
    public RoutingStatementHandler(Executor executor, MappedStatement ms, Object parameter, RowBounds rowBounds, ResultHandler resultHandler, BoundSql boundSql) {
        switch (ms.getStatementType()) {
        case STATEMENT:
            delegate = new SimpleStatementHandler(executor, ms, parameter, rowBounds, resultHandler, boundSql);
            break;
        case PREPARED:
            delegate = new PreparedStatementHandler(executor, ms, parameter, rowBounds, resultHandler, boundSql);
            break;
        case CALLABLE:
            delegate = new CallableStatementHandler(executor, ms, parameter, rowBounds, resultHandler, boundSql);
            break;
        default:
            throw new ExecutorException("Unknown statement type: " + ms.getStatementType());
        }
    }
}
```

## 再进一步

作者在做hibernate 与mybatis 的对比时提到：

1. Hibernate The degree of automation is high, it only needs to configure OR mapping relations, and does not need to write Sql statements. hibernate完全可以通过对象关系模型实现对数据库的操作，拥有完整的JavaBean对象与数据库的映射结构来自动生成sql。而mybatis仅有基本的字段映射，对象数据以及对象实际关系仍然需要通过手写sql来实现和管理。
2. Mybatis Although mapping relations between Sql and Java objects, programmers need to write Sql themselves.


github [baomidou/mybatis-plus](https://github.com/baomidou/mybatis-plus)重点：

1. 只做增强不做改变，这一点对理解 mybatis的使用很重要
2. 启动即会自动注入基本 CURD，Mapper 对应的 XML 支持热加载，对于简单的 CRUD 操作，甚至可以无 XML 启动
3. 内置代码生成器：采用代码或者 Maven 插件可快速生成 Mapper（替换原来的dao） 、 Model 、 Service 、 Controller 层代码
4. 支持 Lambda 形式调用


