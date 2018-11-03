---

layout: post
title: marathon-client 源码分析
category: 技术
tags: Mesos
keywords: marathon java client

---


## 简介

源码位置

[mesosphere/marathon-client](https://github.com/mesosphere/marathon-client)

Feign is used as the underlying REST library. 

也就是以restful api 的方式与 marathon server 交流

## 使用示例

	String endpoint = "<Marathon's endpoint>";
	Marathon marathon = MarathonClient.getInstance(endpoint);
	
根据操作对象 Marathon 可以操作marathon 提供的所有v2 http 接口，比如

	App app = new App();
	app.setId("echohisleepbye-app");
	app.setCmd("echo hi; sleep 10; echo bye;");
	app.setCpus(1.0);
	app.setMem(16.0);
	app.setInstances(1);
	marathon.createApp(app);

## 源码分析

主要包结构

    mesosphere.marathon.client
	 	auth
	 	model
	 		v2
	 			各类请求和相应的model
	 	Marathon
	 	MarathonClient
	 	MarathonException
	 	
我们来看 创建 Marathon 操作对象的代码

public static Marathon getInstance(String endpoint, RequestInterceptor... interceptors) {

	Builder b = Feign.builder()
				.encoder(new GsonEncoder(ModelUtils.GSON))
				.decoder(new GsonDecoder(ModelUtils.GSON))
				.errorDecoder(new MarathonErrorDecoder());
		if (interceptors != null)
			b.requestInterceptors(asList(interceptors));
		String debugOutput = System.getenv(DEBUG_JSON_OUTPUT);
		if ("System.out".equals(debugOutput)) {
			System.setProperty("org.slf4j.simpleLogger.logFile", "System.out");
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");
			b.logger(new Slf4jLogger()).logLevel(Logger.Level.FULL);
		} else if (debugOutput != null) {
			b.logger(new Logger.JavaLogger().appendToFile(debugOutput)).logLevel(Logger.Level.FULL);
		}
		b.requestInterceptor(new MarathonHeadersInterceptor());
		return b.target(Marathon.class, endpoint);
	}
	
比较一栋，再看下Marathon.class

	public interface Marathon {
		// Apps
		@RequestLine("GET /v2/apps")
		@Headers(HeaderUtils.MARATHON_API_SOURCE_HEADER)
		GetAppsResponse getApps() throws MarathonException;
	
		@RequestLine("GET /v2/apps")
		@Headers(HeaderUtils.MARATHON_API_SOURCE_HEADER)
		GetAppsResponse getApps(@QueryMap Map<String, String> queryMap) throws MarathonException;
	
		@RequestLine("GET /v2/apps/{id}")
		@Headers(HeaderUtils.MARATHON_API_SOURCE_HEADER)
		GetAppResponse getApp(@Param("id") String id) throws MarathonException;
		...
	}
	
也很易懂，所以重点就在` b.target(Marathon.class, endpoint);` 上了，全称是`feign.Builder.target(Class<T> apiType, String url);`

	public <T> T newInstance(Target<T> target) {
		Map<String, MethodHandler> nameToHandler = targetToHandlersByName.apply(target);
		Map<Method, MethodHandler> methodToHandler = new LinkedHashMap<Method, MethodHandler>();
		List<DefaultMethodHandler> defaultMethodHandlers = new LinkedList<DefaultMethodHandler>();
	
		for (Method method : target.type().getMethods()) {
		  if (method.getDeclaringClass() == Object.class) {
		    continue;
		  } else if(Util.isDefault(method)) {
		    DefaultMethodHandler handler = new DefaultMethodHandler(method);
		    defaultMethodHandlers.add(handler);
		    methodToHandler.put(method, handler);
		  } else {
		    methodToHandler.put(method, nameToHandler.get(Feign.configKey(target.type(), method)));
		  }
		}
		InvocationHandler handler = factory.create(target, methodToHandler);
		T proxy = (T) Proxy.newProxyInstance(target.type().getClassLoader(), new Class<?>[]{target.type()}, handler);
	
		for(DefaultMethodHandler defaultMethodHandler : defaultMethodHandlers) {
		  defaultMethodHandler.bindTo(proxy);
		}
		return proxy;
	}
	
这段代码最好倒过来看

1. `T proxy = (T) Proxy.newProxyInstance(target.type().getClassLoader(), new Class<?>[]{target.type()}, handler); ` 通过 一个InvocationHandler 可以运行时构造一个 `interface Marathon`实例。
2. InvocationHandler，java 源码上如此表述 Each proxy instance has an associated invocation handler. When a method is invoked on a proxy instance, the method invocation is encoded and dispatched to the invoke method of its invocation handler. 每一个代理类都要有一个InvocationHandler，当执行被代理类的方法时，执行逻辑会转入到InvocationHandler 的invoke 方法上。
3. 根据 interface Marathon 方法上的信息，可以拿到请求uri、参数、返回值等 一个http 请求的基本数据，然后对Marathon 的每一个方法 生成一个MethodHandler，最终将它们聚合成 一个InvocationHandler



## 类似场景对比

[我们能用反射做什么](http://qiankunli.github.io/2018/01/23/reflect.html) 提到： **dynamic proxies 总让人跟代理模式扯上关系，但实际上说dynamic interface implementations 更为直观。**


### 自定义spring namespace handler

spring 自定义一个namespace ，比如`<custom name="",config="",interface="CustomIface">`，然后spring 将其序列化为一个对象，并自动注入到代码中

	@Component
	Class A {
		@Autowire
		private CustomIface custom;
	}

两者的共同点是： 

1. 项目中都没有 CustomIface 的具体实现代码
2. 代理类都是动态生成的，`<custom>` 或者是处理成 spring的BeanDefinition，或者处理成一个FactoryBean， 最终需要 `Proxy.newProxyInstance` 来返回一个代理类。

不同的是：

1. CustomIface 中的方法一般很少，大部分只有一个
2. Marathon.class 的方法很多，而InvocationHandler 只是笼统的拦截了方法的执行，针对每一个方法的具体逻辑，都写在InvocationHandler 中是不现实的。必然要 抽象一个MethodHandler ，而InvocationHandler 中只是将 调用分发到MethodHandler 上。这也是InvocationHandlerFactory 的作用： Controls reflective method dispatch.


所以，我来看下 Feign 提供的效果

1. 以一个接口来描述restful api配置，描述url、请求参数、返回值。
2. 接口实现类完全靠 动态代理实现
3. 整个过程没有spring的参与

[OpenFeign/feign](https://github.com/OpenFeign/feign) 将这种方式称为Interface Annotations,Feign annotations define the Contract between the interface and how the underlying client should work.
