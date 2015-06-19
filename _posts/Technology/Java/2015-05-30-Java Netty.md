---

layout: post
title: java netty
category: 技术
tags: Java
keywords: JAVA netty

---

## 一 前言 ##

netty和mina其实是一个人写的，所以风格非常类似。而在了解了netty和mina之后，笔者真是了解了Java框架的“高大全”。框架嘛，就是将通用的部分固定下来，我们在固定的位置填自己的逻辑代码就可以了。

## 二 netty架构

从使用上将，netty最后该来的“效果”很像http编程（据说tomcat的实现也跟netty有关，至少跟java nio有关）。

![Alt text](/public/upload/java/netty.png) 


## 三 普通的java web开发与Netty的对比

如果不谈struts或spring mvc等上层组件，使用最原始的servlet来构建web项目，我们通常会用到servlet、listener和filter等三个组件。

    public class HelloServlet extends HttpServlet  {
    	@Override
    	protected void doGet(HttpServletRequest req, HttpServletResponse resp)
    			throws ServletException, IOException {
    		PrintWriter out = resp.getWriter();
    		out.write("hello world");
    	}
    }

我们知道，http的底层实现仍然是tcp，亦或者说，browser与web server底层仍然是socket通信。而通过j2ee，呈现在我们面前的是一个servlet。从HttpServletRequest可以拿到请求数据，通过HttpServletResponse可以写回响应。至于数据的encode与decode、socket通信、socket阻塞等细节完全不用关心。

与此同时，j2ee也提供基于事件的机制，那就是listener。

    public class HelloListener implements ServletContextListener{
    	public void contextDestroyed(ServletContextEvent arg0) {
    		System.out.println("HelloListener contextDestroyed");
    	}
    	// web项目启动后会触发该方法
    	public void contextInitialized(ServletContextEvent arg0) {
    		System.out.println("HelloListener contextInitialized");
    	}
    }
    
对数据处理的粒度也是可以细化的，比如在真正处理数据前，先将其转换为json格式等，这些可以通过filter来完成。

    public class HelloFilter implements Filter{
    	public void destroy() {
    		// TODO Auto-generated method stub
    	}
    	public void doFilter(ServletRequest arg0, ServletResponse arg1,
    			FilterChain arg2) throws IOException, ServletException {
    		// TODO Auto-generated method stub
    	}
    	public void init(FilterConfig arg0) throws ServletException {
    		// TODO Auto-generated method stub
    	}
    }
    
所以j2ee主要实现了以下效果：

1. 屏蔽了底层tcp的通信细节。（因为操作中看不到一点socket的痕迹，很容易让人认为web服务器是一个“独立的”高大上的技术）
2. 规范了数据处理。（数据接收，数据转换，数据处理，数据输出）
3. 提供了事件通知等机制，可以让代码及时通信转台变化做出反应。

总之，我们有很多办法实现两个系统间的相互访问，比如http client访问http server。但http并没有覆盖所有场景，比如无法处理大文件、近实时消息（比如聊天或财务数据）。socket肯定可以实现上述功能，但它太麻烦了。而netty作为一个“网络通讯框架”，就是用来解决这个问题。

netty中的IOHandler组件，便类似上述三个组件。

    public class MyHandler extends ChannelHandlerAdapter{
        // 连接建立时
    	@Override
    	public void channelActive(ChannelHandlerContext ctx) throws Exception {
    		super.channelActive(ctx);
    	}
    	// 连接断开时
    	@Override
    	public void channelInactive(ChannelHandlerContext ctx) throws Exception {
    		super.channelInactive(ctx);
    	}
    	// 接收到数据时（msg为接收到的数据，输出数据通过ctx写出）
    	@Override
    	public void channelRead(ChannelHandlerContext ctx, Object msg)
    			throws Exception {
    		super.channelRead(ctx, msg);
    	}
    	// channelRead方法执行完毕时
    	@Override
    	public void channelReadComplete(ChannelHandlerContext ctx) throws Exception {
    		super.channelReadComplete(ctx);
    	}
    }
    
handler可以形成一个pipeline，依次对数据进行处理。比如在pipeline中可以添加一个encoder（也继承自ChannelHandlerAdapter），在对数据进行处理前，先进行编码。

说了这些，其实想结合http server和netty（两者其实本质上是一样的东西）说明一下，网络通信框架在做什么，一般提供什么样的封装等等。

## 其它

netty既然可以实现tcp数据的接收和发送，自然也可以实现其它在tcp基础上的各种协议，比如http、websocket和rpc（hadoop中的rpc组件，便是基于netty实现的）等。

![Alt text](/public/upload/java/netty_http.png) 

基于netty实现的http协议与tomcat等web容器有以下优势：

1. 不必遵守servlet规范

    - 接收数据
    - 解析符合http协议格式的请求数据
    - 处理请求
    - 将处理结果编码为符合http协议规范的格式
    - 返回响应

2. 可以使用异步io模型

我们见到的一些软件（比如docker，当然docker是基于go语言实现的），接受restful api，并且数据部分采用json格式。如果用java实现，便可以在上述netty+http的基础上，增加对请求数据与json的转换过程。


http只是一种半双工协议，在实际的分布式应用中，我们也可以实现一个全双工的协议，包括以下部分：

1. 数据通讯格式
2. 提高可靠性，比如每隔一定时间发送心跳包
3. 增加安全机制，在处理客户端发来的请求时，先对客户端信息进行验证。

## 四 引用

[Netty初步][]

[Netty原理和使用][]


[Netty初步]: http://xpenxpen.iteye.com/blog/2041781
[Netty原理和使用]: http://www.jdon.com/concurrent/netty.html