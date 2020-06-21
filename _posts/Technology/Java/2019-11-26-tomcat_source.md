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

Tomcat 独立部署的模式下，我们通过 startup 脚本来启动 Tomcat，Tomcat 中的 Bootstrap 和 Catalina 会负责初始化类加载器，并解析server.xml和启动这些组件。

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

## tomcat的类加载

[Tomcat热部署与热加载](https://www.yuque.com/renyong-jmovm/kb/emk7gt) 值得细读

tomcat并没有完全遵循类加载的双亲委派机制，考虑几个问题：

1. 如果在一个Tomcat内部署多个应用，多个应用内使用了某个类似的几个不同版本，如何互不影响？org.apache.catalina.loader.WebappClassLoader
2. 如果多个应用都用到了某类似的相同版本，是否可以统一提供，不在各个应用内分别提供，占用内存呢？common ClassLoader 其实质是一个指定了classpath（classpath由catalina.properties中的common.loader 指定`common.loader="${catalina.base}/lib","${catalina.base}/lib/*.jar","${catalina.home}/lib","${catalina.home}/lib/*.jar"`）的URLClassLoader

![](/public/upload/java/tomcat_classloader.jpg)

    public final class Bootstrap {
        ClassLoader commonLoader = null;
        ClassLoader catalinaLoader = null;
        public void init() throws Exception {
            initClassLoaders();
            Thread.currentThread().setContextClassLoader(catalinaLoader);
            SecurityClassLoad.securityClassLoad(catalinaLoader);
            ...
        }
        private void initClassLoaders() {
            ...
            commonLoader = createClassLoader("common", null);
            ...
            catalinaLoader = createClassLoader("server", commonLoader);
        }
    }

![](/public/upload/java/tomcat_loader.png)

热部署和热加载是类似的，都是在不重启Tomcat的情况下，使得应用的最新代码生效。热部署表示重新部署应用，它的执行主体是Host，表示主机。热加载表示重新加载class，它的执行主体是Context，表示应用。

## Sprint Boot如何利用Tomcat加载Servlet？

在内嵌式的模式下，Bootstrap 和 Catalina 的工作就由 Spring Boot 来做了，Spring Boot 调用了 Tomcat 的 API 来启动这些组件。

tomcat 源码中直接提供Tomcat类，其java doc中有如下表述：**Tomcat supports multiple styles of configuration and startup** - the most common and stable is server.xml-based,implemented in org.apache.catalina.startup.Bootstrap. Tomcat is for use in apps that embed tomcat. 从Tomcat类的属性可以看到，该有的属性都有了，内部也符合Server ==> Service ==> connector + Engine ==> Host ==> Context ==> Wrapper 的管理关系，下图绿色部分是通用的。

![](/public/upload/java/tomcat_minimal.png)

所以 Minimal 情况下 new 一个tomcat 即可启动一个tomcat。

    Tomcat tomcat = new Tomcat
    tomcat.setXXX
    tomcat.start();

所以spring-boot-starter-web 主要体现在 创建 并配置Tomcat 实例，具体参见[SpringBoot 中内嵌 Tomcat 的实现原理解析](http://www.glmapper.com/2019/10/06/springboot-server-tomcat/)

## Tomcat如何支持异步Servlet？

从上文类图可知，NioEndpoint中有一个Executor，selector.select 之后，Executor 异步处理 `Socket.read`  + 协议解析 + `Servlet.service`，如果Servlet中的处理逻辑耗时越长就会导致长期地占用Executor，影响Tomcat的整体处理能力。 为此一个解决办法是

    public class AsyncServlet extends HttpServlet {
        Executor executor = xx
        public void doGet(HttpServletRequest req, HttpServletResponse res) {
            AsyncContext asyncContext = req.startAsync(req, res);
            executor.execute(new AsyncHandler(asyncContext));
        }
    }
    public class AsyncHandler implements Runnable {
        private AsyncContext ctx;
        public AsyncHandler(AsyncContext ctx) {
            this.ctx = ctx;
        }
        @Override
        public void run() {
            //耗时操作
            PrintWriter pw;
            try {
                pw = ctx.getResponse().getWriter();
                pw.print("done!");
                pw.flush();
                pw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            ctx.complete();
        }
    }

startAsync方法其实就是创建了一个异步上下文AsyncContext对象，该对象封装了请求和响应对象。然后创建一个任务用于处理耗时逻辑，后面通过AsyncContext对象获得响应对象并对客户端响应，输出“done!”。完成后要通过complete方法告诉Tomcat已经处理完，Tomcat就会请求对象和响应对象进行回收处理或关闭连接。

![](/public/upload/java/tomcat_async.png)

    public class Request
        implements HttpServletRequest {
        public AsyncContext startAsync(ServletRequest request,
                ServletResponse response) {
            ...
            asyncContext = new AsyncContextImpl(this);
            ...
            asyncContext.setStarted(getContext(), request, response,
                    request==getRequest() && response==getResponse().getResponse());
            asyncContext.setTimeout(getConnector().getAsyncTimeout());
            return asyncContext;
        }
    }

写回数据由Response 完成，从代码看，AsyncContextImpl.complete 方法表示 tomcat 可以重新开始关注该socket read事件了（之前一直在等socket 写回客户端数据）。

![](/public/upload/java/tomcat_async_complete.png)

## 其它

### tomcat为什么运行war 而不是jar

如果一个项目打成jar包，那么tomcat 在启动时 就要去分析下 这个jar 是一个web项目还是一个 普通二方库。 

### 安全

如果你在Servlet代码中直接 加入`System.exit(1)` 你会发现，仅仅是作为一个tomcat 上层的一个“业务方”，却有能力干掉java进程，即tomcat的运行。

    public class XXServlet extends HttpServlet {
        protected void doGet(HttpServletRequest req, HttpServletResponse resp)
            throws ServletException, IOException{
            System.exit(1);
            xxx
        }
    }

