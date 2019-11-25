---

layout: post
title: 《how tomcat works》笔记
category: 技术
tags: Java
keywords: tomcat

---

## 简介

对一个java web应用来说，Tomcat是首，SSM是中，JVM是尾。我们通常对于SSM是比较了解的，而忽略了首尾，而Tomcat在目前的网络编程中是举足轻重的。如果能够掌握Tomcat的原理，那么是非常有用的，比如：

1. Tomcat到底是如何处理一个请求的？这对于针对Tomcat的性能调优是必备的。
2. 目前Spring Boot和Dubbo等框架中都是使用的内嵌Tomcat，那么一个内嵌的Tomcat到底是如何运行的？
3. Tomcat的架构设计其实非常优秀的，如果能明白Tomcat为什么要那么设计，那么对于Tomcat的原理和自己的架构设计思维都能有很大提升。

《how tomcat works》的书写方式类似于 从0到1 写tomcat，从一个入门级程序猿的demo 代码开始，逐渐演化 出一个web 容器。

## 代码的演化
        
新手的直观感觉

	While(true){
		Socket socket = serverSocket.accept();
		inputStream in = xxx
		OutputStream out = xx
		按照http协议读取和写入
	}

进化

	Whle(true){
		Socket socket = serverSocket.accept();
		inputStream in = xxx
		OutputStream out = xx
		Request request= new Request(in)
		Response reponse = new Response (out)
		HttpProcessor processor = new HttpProcessor(request,response)
	}

进化 

	class Bootstrap{
		public static void main(String[] args){
			HttpConnector connector = new HttpConnector();
			connector.start();
		}
	}

	HttpConnector implements Runnalbe{
		public void run(){
			socket = serversSocket.accept();
			while(){
				HttpProcessor processor = new HttpProcessor(this);
				processor.process(socket)
			}
		}	
	}

	
	class HttpProcessor{
		public void process(Socket socket){
			request = xxx
			response = xxx
			if(request.getRequestURI().startsWith(“/servlet/”)){
				ServletProcessor processor = new ServletProcessor();
				Processor.process(request,response);
			}else{
				StaticResourceProcessor processor = new StaticResourceProcessor();
				processor.process(request,response)
			}
		}
	}
   
   
httpConnector，接受socket，传给HttpProcessor，解析request和response。HttpProcessor传给Container，`container.invoke(request,response)`实现具体的业务逻辑。

	Class Container{
		Invoke(request,response){
			Pipeline.invoke(request,response)
		}
	}

更准确的说：[详解tomcat的连接数与线程池](https://www.cnblogs.com/kismetv/p/7806063.html) Connector的主要功能，是接收连接请求，创建Request和Response对象用于和请求端交换数据；然后分配线程让Engine（也就是Servlet容器）来处理这个请求，并把产生的Request和Response对象传给Engine。当Engine处理完请求后，也会通过Connector将响应返回给客户端。

Container容器（tomcat 中叫 Catalina ）有父子关系，有四种容器：

1. engine（引擎） , 为一个域名找到 合适的 host
2. host（主机） ,  为一个域名下的url返回处理它的context
3. context（上下文），代表一个web 项目，加载配置文件，调用sessionManager 等
4. wrapper（包装器），包装一个servlet

父子关系，就是分层，在不同的层次解决不同的问题。

对于每一个连接，连接器都会调用关联容器的 invoke 方法。接下来容器调用它
的所有子容器的 invoke 方法。但容器并不是invoke方法的简单包装，为了invoke方法可以正常执行，容器必须“加载servlet（对于wrapper容器来讲，适当时机执行servlet.init,servlet.destroy），加载配置文件（对于context来讲）等”外围工作，不同级别的容器，所做的外围工作不同。

## 过滤器

== 接口定义 ==

	public interface Filter {   
        .....          
        //执行过滤   
        public void doFilter ( ServletRequest request, ServletResponse response, FilterChain chain ) throws IOException, ServletException;   
	}   
  
	public interface FilterChain {   
	    public void doFilter ( ServletRequest request, ServletResponse response ) throws IOException, ServletException;   
	} 

== 实现类 ==

	class ApplicationFilterChain implements FilterChain {   
		private int pos=0;
		public void doFilter ( ServletRequest request, ServletResponse response ) throws IOException, ServletException{
		   //pos为当前filter的所在位置,n为filters数组的长度   
		   if (pos < n) {   
	            //pos++执行后，把filterchain的当前filter指向下一个   
	            ApplicationFilterConfig filterConfig = filters[pos++];   
	            Filter filter = null;   
	            try {   
	                filter = filterConfig.getFilter();   
	                //filter执行过滤操作   
	                filter.doFilter(request, response, this);   
	            }   
	            ...   
		   }
	   }
	}   
  
	class SampleFilter implements Filter {   
	      ........   
	      public void doFilter(ServletRequest request, ServletResponse response,FilterChain chain)   
	        throws IOException, ServletException {   
	         //do something    
	         .....   
	         //request, response传递给下一个过滤器进行过滤   
	         chain.doFilter(request, response);   
	    }   
	}


chain.doFilter 就是一个内部递归，只是分散在了两个对象上执行。

== 我当时想起来的方式 ==

	Filter{
		Boolean doFilter(req,resp);
	}
	FilterChain{
		doFilter(req,resp){
			Filters = xxx;
			for(int i=0;i<n;i++){
				if(! filters[i].doFilter(req,resp)){
					break;
				}
			}
			servlet.service(req,resp)
		}
	}

这种方式有以下缺点

1.	处理完filter，还得单独调用一下servlet。而tomcat的模式，filter和servlet是兼容的
2.	这种方式只能实现前向过滤，不能实现后置过滤，要知道，在tomcat的filter中可以

## 线程池

[多线程](http://qiankunli.github.io/2014/10/09/Threads.html)

1. HttpProcessor 会被封装成 runnable 交給Executor 执行。 **所以，所谓丢弃 连接，或者服务端执行 超时，都要从线程池 提交任务 这个事情来理解**
2. 线程池的核心 就两个事儿：核心线程数、等待队列。因此，tomcat中 也会对应有 最小线程数、最大线程数、队列长度（tomcat 中叫acceptCount）等配置。可见，tomcat 某一个时刻能处理的最大请求数 由最大线程数 + 队列长度 决定的。
1. 线程池 线程数 是有限的，超过线程数 会在队列中等待。如果队列已满，则会执行reject 策略。 默认策略是：线程池 拒绝 HttpProcessor 为主体的 runnable，服务端关闭 socket，抛异常。客户端感知到 连接被关闭了 connection refused。
2. 如果 一个线程 执行超时，则客户端会断开连接。此时，服务端线程 仍然继续 持有 socket 并做运算，只是最终向socket 写入数据时（`connector.OutputBuffer.realWriteBytes(OutputBuffer.java:393)`），会报Broken pipe异常（当然，这只是引起Broken pipe 原因之一）。

		org.apache.catalina.connector.ClientAbortException: java.io.IOException: Broken pipe  
	    at org.apache.catalina.connector.OutputBuffer.realWriteBytes(OutputBuffer.java:393)  
	    at org.apache.tomcat.util.buf.ByteChunk.flushBuffer(ByteChunk.java:426)  
	    at org.apache.catalina.connector.OutputBuffer.doFlush(OutputBuffer.java:342)  
	    at org.apache.catalina.connector.OutputBuffer.close(OutputBuffer.java:295)  
	    at org.apache.catalina.connector.Response.finishResponse(Response.java:453)  
	    at org.apache.catalina.core.StandardHostValve.throwable(StandardHostValve.java:378)  

	客户端tcp 连接关闭时，服务端会有大量的CLOSE_WAIT 状态的连接，检查服务端CLOSE_WAIT 连接数 也是定位问题的手段之一。