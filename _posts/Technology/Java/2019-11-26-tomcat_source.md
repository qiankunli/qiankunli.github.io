---

layout: post
title: Tomcat源码分析
category: 技术
tags: Java
keywords: tomcat

---

## 简介

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

本文整体结构受周瑜《Tomcat底层源码解析与性能调优》培训视频的启发。

## 从各个视角看tomct

### tomcat是一个Servlet 容器？


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

绿色的类定义 在servlet-api 包中，其它类除自定义外在tomcat 包中

### 整体架构

![](/public/upload/java/tomcat_war.png)

tomcat 的功能简单说 就是让 一堆class文件+web.xml  可以对外支持http

![](/public/upload/java/tomcat_sample.png)

![](/public/upload/java/tomcat_overview.png)

## 启动过程

`/usr/java/jdk1.8.0_191/bin/java -Dxx  -Xxx org.apache.catalina.startup.Bootstrap start`

![](/public/upload/java/tomcat_start.png)

分别启动连接管理部分和业务处理部分

![](/public/upload/java/tomcat_object_overview.png)

业务处理部分中，各个类的关系 在tomcat server.xml 中体现的也非常直观

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


## io处理

### connector 架构

![](/public/upload/java/tomcat_connector.png)

### io 和线程模型

![](/public/upload/java/tomcat_connector_object.png)

同样一个颜色的是内部类的关系

![](/public/upload/java/tomcat_handle_request_io.png)

1. Http11NioProtocol start 时会分别启动poller 和 acceptor 线程
2. acceptor 持有ServerSocket/ServerSocketChannel， 负责监听新的连接，并将得到的Socket 注册到Poller 上
3. Poller 持有Selector， 负责`selector.select()` 监听读写事件，将新的socket 注册到selector上，以及其它通过addEvent 加入到Poller中的event
4. Http11NioProcessor 封装了 http 1.1 的协议处理部分，比如parseRequestLine，连接出问题时response设置状态码为503 或400 等。以读事件为例， 最终会将 数据读取到 Request 对象的inputBuffer 中

线程数量

    public class NioEndpoint extends AbstractEndpoint<NioChannel> {

        private Executor executor = new ThreadPoolExecutor(getMinSpareThreads(), getMaxThreads(), 60, TimeUnit.SECONDS,taskqueue, tf);
        
        private int pollerThreadCount = Math.min(2,Runtime.getRuntime().availableProcessors()); // new Thread().start() 的方式
        protected int acceptorThreadCount = 0;      // new Thread().start() 的方式

        // poller  内部除了 selector.select() 逻辑外，一般通过executor 异步执行
        // acceptor 就是简单的 accept 一个socket 并将其 加入到poller 的event 队列中（ 以将socket 注册到selector）所以没有用到executor
        
    }


## 业务处理

### container 架构

![](/public/upload/java/tomcat_container.png)

为了更清晰一点，上图只画出了Host 类族，Engine、Context、Wrapter 与Host 类似。黄色部分组成了一个pipeline，可以看到Engine、Context、Wrapter 和Host 作为容器，并不亲自“干活”，而是交给对应的pipeline。

    public class CoyoteAdapter implements Adapter {
        // 有读事件时会触发该操作
        public boolean event(org.apache.coyote.Request req,
            org.apache.coyote.Response res, SocketStatus status) {
            ...
            // 将读取的数据写入到 request inputbuffer 
            request.read();
            ...
            // 触发filter、servlet的执行
            connector.getService().getContainer().getPipeline().getFirst().event(request, response, request.getEvent());
            ...
        }
    }

pipeline 逐步传递请求直到Servlet

![](/public/upload/java/tomcat_handle_request_container.png)

## 其它

### tomcat 为什么运行war 而不是jar

如果一个项目打成jar包，那么tomcat 在启动时 就要去分析下 这个jar 是一个web项目还是一个 普通二方库。 

### tomcat 与热部署和热加载

热部署和热加载是类似的，都是在不重启Tomcat的情况下，使得应用的最新代码生效。热部署表示重新部署应用，它的执行主体是Host，表示主机。热加载表示重新加载class，它的执行主体是Context，表示应用。

[Tomcat热部署与热加载](https://www.yuque.com/renyong-jmovm/kb/emk7gt) 值得细读

### Sprint Boot如何利用Tomcat加载Servlet？（待补充）

### Tomcat如何支持异步Servlet？（待补充）

### 安全

如果你在Servlet代码中直接 加入`System.exit(1)` 你会发现，仅仅是作为一个tomcat 上层的一个“业务方”，却有能力干掉java进程，即tomcat的运行。

    public class XXServlet extends HttpServlet {
        protected void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws ServletException, IOException{
            System.exit(1);
            xxx
        }
    }

