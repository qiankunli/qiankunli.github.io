---

layout: post
title: Spring 创建Bean的年代变迁
category: 技术
tags: Spring
keywords: springboot

---

## 简介

* TOC
{:toc}

![](/public/upload/spring/ioc_overview.png)

本文主要讲下图中左侧的内容

## 根据xml, annotations or java code创建Bean

[History of Spring Framework and Spring Boot](https://www.quickprogrammingtips.com/spring-boot/history-of-spring-framework-and-spring-boot.html)It currently consists of a large number of modules providing a range of services. These include a component container, aspect oriented programming support for building cross cutting concerns, security framework, data access framework, web application framework and support classes for testing components. **All the components of the spring framework are glued together by the dependency injection architecture pattern**. Dependency injection(also known as inversion of control) makes it easy to design and test loosely coupled software components. 依赖注入的关键就是有一个component container/IOC container，它持有所有对象的实例，负责所有对象的创建和销毁问题，在创建对象时可以夹一点自己的私货，比如

1. 使用BeanFactory 创建复杂对象
1. 将某个类实例 掺加一点业务逻辑改为 一个代理类

Over years spring framework has grown substantially. Almost all infrastructural software components required by a java enterprise application is now available in spring framework. However **collecting all the required spring components together and configuring them in a new application requires some effort**. This involves setting up library dependencies in gradle/maven and then configuring the required spring beans using xml, annotations or java code. Spring developers soon realized that it is possible to automate much of this work. Enter spring boot!互联网应用有很多基础设置，比如kafka、mybatis，他们实现了对应领域的基本功能，但它们并不直接与spring 相容，需要提供 spring-kafka、spring-mybatis 等去封装他们的入口对象，通过xml、注解、配置类等方式将其注入到spring 的ioc中。**所谓与spring 整合，就是融入spring的ioc中**。

spring 迭代过程中，跟配置有关的部分，配置最终也是为了创建Bean。

1. spring 1.0
2. spring 2.0

    1. extensible XML configuration which was used to simplify XML configuration,
    3. additional IoC container extension points
3. spring 2.5, 
    1. support for annotation configuration
    2. component auto-detection in classpath
4. spring 3.0, java based bean configuration(JavaConfig)
5. spring 3.1, introduced the annotation @Profile
6. Spring 4.0, provide some updates to Spring 3.x @Profile


## 根据@Profile创建Bean ==> 一次打包即可跨环境运行

[@Profile Annotation Improvements in Spring 4](https://javapapers.com/spring/profile-annotation-improvements-in-spring-4/)

@Configuration标注在类上，相当于把该类作为spring的xml配置文件中的`<beans>`，作用为：配置spring容器(应用上下文)。When our Spring application starts, Spring IOC container loads all beans defined in this class into application context so that other components can utilize them.
@Bean标注在方法上(返回某个实例的方法)，等价于spring的xml配置文件中的`<bean>`，作用为：注册bean对象


Before Spring 3.1, if we want to create Environment based application build, we should use Maven Profiles（并且一个环境要打一次包）. Then, Spring framework introduced similar kind of concept with @Profile annotation.

    @Configuration
    @Profile("dev")
    public abstract class DevEmployeeConfig{	  
        @Bean
        public DataSource dataSource() {
            return new DevDatabaseUtil();
        }	 
    }
    @Configuration
    @Profile("prod")
    public abstract class ProdEmployeeConfig{	  
        @Bean
        public DataSource dataSource() {
            return new ProductionDatabaseUtil();
        }
    }

spring 从环境变量、jvm参数等各个渠道通过`spring.profiles.active` 和`spring.profiles.default` 确定profile的值。

In Spring 3.1, we can use the @Profile annotation only at the class level. We cannot use at method level. From Spring 4.0 onwards, we can use @Profile annotation at the class level and the method level. 两个@Bean 其实定义了一个dataSource，根据@Profile 决定

    @Configuration
    publicclass EmployeeDataSourceConfig {
        @Bean(name="dataSource")
        @Profile("dev")
        public DataSource getDevDataSource() {
            returnnew DevDatabaseUtil();
        }	
        @Bean(name="dataSource")
        @Profile("prod")
        public DataSource getProdDataSource() {
            returnnew ProductionDatabaseUtil();
        }	
    }


## 根据任意条件创建Bean

[Spring @Conditional Annotation](https://javapapers.com/spring/spring-conditional-annotation/)

Spring 4.0 has introduced a new annotation @Conditional. It is used to develop an “If-Then-Else” type of conditional checking for **bean registration**. 

Spring 4.0 @Conditional annotation is at more higher level when compared to @Profile annotation.Difference between @Conditional and @Profile Annotations

1. Spring 3.1 @Profiles is used to write conditional checking based on Environment variables only. Profiles can be used for loading application configuration based on environments.
2. Spring 4 @Conditional annotation allows Developers to define user-defined strategies for conditional checking. @Conditional can be used for conditional bean registrations.


    public class LinuxCondition implements Condition {
        public boolean matches(ConditionContext context,AnnotatedTypeMetadata metadata) {
            return context.getEnvironment().getProperty("os.name").contains("Linux");
        }
    }
    @Configuration
    public class ConditionConifg {
        @Bean
        @Conditional(WindowsCondition.class) 
        public ListService windowsListService() {
            return new WindowsListService();
        }
        @Bean
        @Conditional(LinuxCondition.class) 
        public ListService linuxListService() {
            return new LinuxListService();
        }
    }

[Spring 4.0 的条件化注解](https://qidawu.github.io/2017/06/05/spring-conditional-bean/)

    public interface Condition {
        boolean matches(ConditionContext var1, AnnotatedTypeMetadata var2);
    }

其中，通过 Condition 接口的入参 ConditionContext，我们可以做到如下几点：

1. 借助 getRegistry() 返回的 BeanDefinitionRegistry 检查 bean 定义；
2. 借助 getBeanFactory() 返回的 ConfigurableListableBeanFactory 检查 bean 是否存在，甚至探查 bean 的属性；
3. 借助 getEnvironment() 返回的 Environment 检查环境变量是否存在以及它的值是什么；
4. 读取并探查 getResourceLoader() 返回的 ResourceLoader 所加载的资源；
5. 借助 getClassLoader() 返回的 ClassLoader 加载并检查类是否存在。

## 有条件创建Bean + 一点变通 ==> 自动配置

    public class JdbcTemplateCondition implements Condition {
        @Override
        public boolean matches(ConditionContext context,AnnotatedTypeMetadata metadata) {
            try {
                // 加载成功即存在，加载失败跑异常即不存在
                context.getClassLoader().loadClass("org.springframework.jdbc.core.JdbcTemplate");
                return true;
            } catch (Exception e) {
                return false;
            } 
        }
    }

Springboot 定义了很多 @Conditional，比如`ConditionalOnClass("org.springframework.jdbc.core.JdbcTemplate")` 便等同了上述代码。