---

layout: post
title: 《Tomcat底层源码解析与性能调优》笔记
category: 技术
tags: Java
keywords: tomcat

---

## 简介（未完成）

* TOC
{:toc}

[tomcat8源码maven方式组织](https://github.com/emacsist/tomcat-source-maven)

使用golang 语言 实现一个http server，只需几行代码即可，为何用java 实现如何“沉重”呢？这背后tomcat 是一个什么角色呢？

    package main
    import (
        "io"
        "net/http"
    )
    func helloHandler(w http.ResponseWriter, req *http.Request) {
        io.WriteString(w, "hello, world!\n")
    }
    func main() {
        http.HandleFunc("/", helloHandler)
        http.ListenAndServe(":12345", nil)
    }

## tomcat是一个Servlet 容器？


单纯的思考一下这句话，我们可以抽象出来这么一段代码：

    class Tomcat {
        List<Servlet> sers;
    }

如果Tomcat就长这样，那么它肯定是不能工作的，所以，Tomcat其实是这样：

    class Tomcat {
        Connector connector; // 连接处理器
        List<Servlet> sers;
    }

### Servlet规范与tomcat实现

![](/public/upload/java/servlet_tomcat_object.png)

在tomcat server.xml 中体现的也非常直观

    <Server port="8005" shutdown="SHUTDOWN">
        <Listener className="org.apache.catalina.core.ThreadLocalLeakPreventionListener" />
        <Service name="Catalina">
            <Connector port="8080" protocol="HTTP/1.1"
                connectionTimeout="20000"
                redirectPort="8443" />
            <Connector port="8009" protocol="AJP/1.3" redirectPort="8443" />
        </Service>
        <Engine name="Catalina" defaultHost="localhost">
            <Realm className="org.apache.catalina.realm.UserDatabaseRealm"
                resourceName="UserDatabase"/>
            <!--可以另外创建一个host，使用不同的appBase-->
            <Host name="localhost"  appBase="webapps"
                unpackWARs="true" autoDeploy="true">
                <!--可以配置 Context>
            </Host>
        </Engine>
    </Server>

## 从各个视角看tomct


![](/public/upload/java/tomcat_war.png)

tomcat 的功能简单说 就是让 一堆class文件+web.xml  可以对外支持http

![](/public/upload/java/tomcat_sample.png)

![](/public/upload/java/tomcat_overview.png)

### connector 架构

![](/public/upload/java/tomcat_connector.png)

### io 和线程模型

![](/public/upload/java/tomcat_connector_object.png)

1. 异步accept(), 并将得到的socket 注册到poller中

## 启动过程

`/usr/java/jdk1.8.0_191/bin/java -Dxx  -Xxx org.apache.catalina.startup.Bootstrap start`

![](/public/upload/java/tomcat_start.png)

分别启动连接管理部分和业务处理部分

![](/public/upload/java/tomcat_object_overview.png)


webapps 下没有war包 也可以启动。有了war 包，通过事件 触发war 包的解压和加载

## 一次请求的处理

![](/public/upload/java/tomcat_handle_request.png)

## Request 和 Response 是如何读取和写回数据的

## 其它

### tomcat 为什么运行war 而不是jar

如果一个项目打成jar包，那么tomcat 在启动时 就要去分析下 这个jar 是一个web项目还是一个 普通二方库。 

