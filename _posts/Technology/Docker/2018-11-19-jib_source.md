---

layout: post
title: jib源码分析及应用
category: 技术
tags: Docker
keywords: jib

---

## 简介

* TOC
{:toc}

阅读本文前，建议事先了解下 [docker环境下的持续构建](http://qiankunli.github.io/2018/11/18/ci_in_docker.html#)

## 基本使用

### 直接通过代码做镜像

	Jib.from("busybox")
	   .addLayer(Arrays.asList(Paths.get("helloworld.sh")), AbsoluteUnixPath.get("/")) 
	   .setEntrypoint("sh", "/helloworld.sh")
	   .containerize(
	       Containerizer.to(RegistryImage.named("gcr.io/my-project/hello-from-jib")
	                                     .addCredential("myusername", "mypassword")));

### 集成到maven

#### 集成到pom

[Google开源其Java容器化工具Jib，简化镜像构建全流程](https://mp.weixin.qq.com/s/KwmVoFeUG8gJCrgh5AFkvQ)

`mvn compile jib:build` 从中可以看到


	[INFO] Retrieving registry credentials for harbor.test.ximalaya.com...
	[INFO] Getting base image harbor.test.ximalaya.com/test/jdk8-tomcat8...
	[INFO] Building dependencies layer...
	[INFO] Building resources layer...
	[INFO] Building classes layer...
	[INFO] Retrieving registry credentials for harbor.test.ximalaya.com...
	[INFO] Finalizing...
	[INFO] 
	[INFO] Container entrypoint set to [java, -cp, /app/libs/*:/app/resources/:/app/classes/, org.apache.catalina.startup.Bootstrap]
	[INFO] 
	[INFO] Built and pushed image as harbor.xx/test/jib-demo


1. 与常规的将代码及依赖 打成一个jar 包作为一个layer 不同，jib 将dependencies、resources、 classes（即项目代码） 分别打成一个layer， 在项目实践中，dependencies、resources 变化不多 ，因此能够复用相当一部分空间。

2. maven pom.xml 配置 针对插件的 0.9.9 版本

		<plugin>
			<groupId>com.google.cloud.tools</groupId>
			<artifactId>jib-maven-plugin</artifactId>
			<version>0.9.9</version>
			<configuration>
				<allowInsecureRegistries>false</allowInsecureRegistries>
				<from>
					<image>harbor.test.xxx.com/test/jdk8</image>
					<auth>
						<username>xxx</username>
						<password>xxx</password>
					</auth>
				</from>
				<to>
					<image>harbor.test.xxx.com/test/jib-springboot-demo</image>
					<auth>
						<username>xxx</username>
						<password>xxx</password>
					</auth>
				</to>
				<container>
					<mainClass>com.xxx.springboot.demo.DockerSpringbootDemoApplication</mainClass>
				</container>
			</configuration>
		</plugin>




还有一种方案  [Optimizing Spring Boot apps for Docker](https://openliberty.io/blog/2018/06/29/optimizing-spring-boot-apps-for-docker.html)

#### 通过mvn调用

## 打tag

To tag the image with a simple timestamp, add the following to your pom.xml:

	<properties>
	  <maven.build.timestamp.format>yyyyMMdd-HHmmssSSS</maven.build.timestamp.format>
	</properties>
	Then in the jib-maven-plugin configuration, set the tag to:
	
	<configuration>
	  <to>
	    <image>my-image-name:${maven.build.timestamp}</image>
	  </to>
	</configuration>
	
## jib 优化

从目前来看，我们感觉针对 java 项目来说，jib 是最优的。jib 貌似可以做到，不用非得在 pom.xml 中体现。`mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build -Dimage=<MY IMAGE>`

对以后的多语言，可以项目中 弄一个deploy.yaml 文件。这个yaml 文件应该是 跨语言的，yaml 文件的执行器 应能根据 yaml 配置 自动完成 代码 到 镜像的所有工作。

	language:java
	param:
	creator: zhangsan
	
然后，编写一个 build 程序

1. 碰到java 执行 `mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build -Dimage=<MY IMAGE>` 。这里 build 程序要做的工作 是根据 deploy.yaml 文件的用户参数 拼凑 `mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build -Dimage=<MY IMAGE>`  命令 并执行。 **本质上还是用maven，只是在jenkins 和 maven 之间加了一层**，加了一层之后，就可以更方便的支持用户的个性化（或者说，用户可以在项目代码上 而不是jenkins 上做个性化配置）
2. 碰到golang 执行 go build。  go 语言中 一定有类似  jib 的框架在。

## 源码分析

### maven 插件的 基本抽象

[博客园首页联系订阅管理
随笔 - 90  文章 - 0  评论 - 234
Maven提高篇系列之（六）——编写自己的Plugin](http://www.cnblogs.com/davenkin/p/advanced-maven-write-your-own-plugin.html)

### 主干代码

`mvn compile jib:build` 触发 BuildImageMojo execute 方法执行

中间有一个分步执行 框架

BuildStepsRunner 包括一个BuildSteps 属性，外界通过  `BuildStepsRunner.build` ==> `BuildSteps.run` 触发build 过程的执行

  	BuildStepsRunner(BuildSteps buildSteps) {
    	this.buildSteps = buildSteps;
  	}
  	public void build(HelpfulSuggestions helpfulSuggestions){
  	
  	}

BuildSteps  主要是 触发 `StepsRunnerConsumer.accept` ==> StepsRunner 的执行序列
  	
	private BuildSteps(
		String description,
		BuildConfiguration buildConfiguration,
		SourceFilesConfiguration sourceFilesConfiguration,
		Caches.Initializer cachesInitializer,
		String startupMessage,
		String successMessage,
		StepsRunnerConsumer stepsRunnerConsumer){
			...
		}
		
StepsRunner 是有一个基本的构造函数之后

	class StepsRunner{
		private 各种Step
		private 基本属性(主要是上下文参数)
    	public StepsRunner(
      		BuildConfiguration buildConfiguration,
      		SourceFilesConfiguration sourceFilesConfiguration,
      		Cache baseLayersCache,
      		Cache applicationLayersCache) {
      			...
      		}
    }

执行runxxx 方法，runxx方法 构造一个对应的xxxStep 赋给  StepsRunner 对应xxStep 成员。new xxStep 时， 构造方法中触发了xxStep 逻辑的异步执行    	

	stepsRunner
        .runRetrieveTargetRegistryCredentialsStep()
        .runAuthenticatePushStep()
        .runPullBaseImageStep()
        .runPullAndCacheBaseImageLayersStep()
        .runPushBaseImageLayersStep()
        .runBuildAndCacheApplicationLayerSteps()
        .runBuildImageStep(getEntrypoint(buildConfiguration, sourceFilesConfiguration))
        .runPushContainerConfigurationStep()
        .runPushApplicationLayersStep()
        .runFinalizingPushStep()
        .runPushImageStep()
        .waitOnPushImageStep()
        
        
 每一个AsyncStep 的大致组成是
 
 	class xxStep implements AsyncStep<Void>, Callable<Void>{
 		private 完成本Step所需基本属性
 		private 依赖Step
 		private final ListenableFuture<Void> listenableFuture;
 		xxStep(基本属性,依赖Step){
	 		赋值code
	 		// 依赖任务执行完毕后，执行本Step 的call 方法
	 		listenableFuture = Futures.whenAllSucceed(
	                依赖Step.getFuture(),
	                依赖Step.getFuture())
	            .call(this, listeningExecutorService);
 		}
 	}
 	
 
最优意思的部分就是， 本来十几个step 具有复杂的依赖关系，有的需要同步执行，有的可以异步执行。而通过代码的腾挪， 表面调用起来却是平铺直叙的。
 
从另一个角度说，代码调用可以是顺序的，但业务不是顺序的。代码呈现的感觉跟实际的执行 不是一回事（也可以说，我们以前的方法太笨了）。

再换一个角度说，我们看下 rxnetty 的一些代码，充分体现“程序=逻辑+控制”，逻辑与控制的分离。 

	RxNetty.createHttpGet("http://localhost:8080/error")
	               .flatMap(response -> response.getContent())
	               .map(data -> "Client => " + data.toString(Charset.defaultCharset()))
	               .toBlocking().forEach(System.out::println);
      	


个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)