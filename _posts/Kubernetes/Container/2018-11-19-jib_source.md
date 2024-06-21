---

layout: post
title: jib源码分析及应用
category: 技术
tags: Container
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


	[INFO] Retrieving registry credentials for harbor.test.xx.com...
	[INFO] Getting base image harbor.test.xx.com/test/jdk8-tomcat8...
	[INFO] Building dependencies layer...
	[INFO] Building resources layer...
	[INFO] Building classes layer...
	[INFO] Retrieving registry credentials for harbor.test.xxx.com...
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

假设存在一个jib-demo的web项目，则可以在项目目录下（与项目pom.xml平级）执行

	mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build \
		-Djib.from.image=xx/common/jdk8-tomcat8 \
	    -Djib.from.auth.username=zhangsan \
	    -Djib.from.auth.password=lisi \
		-Djib.to.image=xx/test/jib-demo \
	    -Djib.to.auth.username=zhangsan \
	    -Djib.to.auth.password=lisi
	    
也就是所有的pom配置都可以转换为命令行配置，使用这种方式的好处是开发无感知。

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
	


## 源码分析

针对jib 0.10.1

环境准备

1. 从[https://github.com/GoogleContainerTools/jib](https://github.com/GoogleContainerTools/jib) 拉取源文件，主要分为三个部分

		docs
		jib-core
		jib-gradle-plugin
		jib-maven-plugin
		jib-plugins-common
2. `jib-core` `jib-gradle-plugin` `jib-maven-plugin` 是单独的项目（使用ide 单独打开），后两者都用到了 `jib-core`，等于说基本实现 靠 `jib-core`，然后包了一个gradle 或 maven 的壳
3. 引入gradle 项目，idea 一般要先 将 `mavenLocal()` 加入到 build.gradle 的 repositories 中，并执行`gradlew build`，将相关依赖下载完毕，然后再用idea 打开一次即可。


### 主干代码

要梳理两个事情

1. 流程如何驱动，中间有一个分步执行 框架
2. build 一个image，要干哪些事情，有哪些基本抽象


![](/public/upload/docker/jib_process.png)
  	
### 基本抽象
  	  	
从jib-core 的代码demo 看，有一个包挺重要，那就是`com.google.cloud.tools.jib.api` 包括几个类，划分一下

	Jib						// jib 对外暴漏的操作对象类，实质操作JibContainerBuilder
	Containerizer			// 看样子啥都没干，聚合了一堆参数
	JibContainer
		JibContainerBuilder
	TargetImage
		DockerDaemonImage
		RegistryImage
		TarImage
	SourceImage
		RegistryImage
		
也就是基本概念其实就四个：Jib、Containerizer、JibContainer、SourceImage 和 TargetImage。

	interface SourceImage {
	  	ImageConfiguration toImageConfiguration();
	}
	interface TargetImage {
  		ImageConfiguration toImageConfiguration();
  		BuildSteps toBuildSteps(BuildConfiguration buildConfiguration);
	}
	
### 信息采集

从一个较高的角度来说，Jib 干了什么事儿呢？

	Jib.from("busybox")
	   .addLayer(Arrays.asList(Paths.get("helloworld.sh")), AbsoluteUnixPath.get("/")) 
	   .setEntrypoint("sh", "/helloworld.sh")
	   .containerize(
	       Containerizer.to(RegistryImage.named("gcr.io/my-project/hello-from-jib")
	                                     .addCredential("myusername", "mypassword")));
	                                     
JibContainerBuilder 是 和 jib 平级的入口对象，Jib 对象的唯一作用就是引出JibContainerBuilder，之所以是Jib 而不是JibContainerBuilder 作为第一入口对象，估计是为了可读性。

![](/public/upload/docker/jib_JibContainerBuilder.png)

但JibContainerBuilder 也不是主角，所做的一切都是为了构造BuildConfiguration，我们看下 BuildConfiguration 的成员

![](/public/upload/docker/jib_BuildConfiguration_2.png)

可以对 BuildConfiguration 涉及的所有配置列出一个层次关系

![](/public/upload/docker/jib_BuildConfiguration_1.png)


此处有几点

1. JibContainerBuilder 是 JibContainer 的  Builder，常规来说Builder 类中会有很多属性，组后通过build 方法将其转换为Builder 目标对象。但JibContainerBuilder 估计是  属性太多了，所以其内部将属性归类，又套了一层Builder：ContainerConfiguration.Builder、BuildConfiguration.Builder。
2. Containerizer 也像是一个 参数聚合类。换句话说，当一个流程有很多参数要配置时，你可以使用Builder 模式（甚至Builder 套Builder），也可以传入配置类。**为什么要玩这么多花活儿呢？ 将配置类分门别类，使其更符合语义。**
3. BuildConfiguration 聚合了各种配置类，它才是所有配置参数的集中地。此外，其不仅指定了静态的配置， 还指定了eventDispatcher 以及 ExecutorService 等对象，动静结合，使得BuildSteps 只关注 build step 本身的串联。



### 流程驱动

有个问题

1. Step 如何串到一起
2. Step 如何执行，BuildSteps.run ==> StepRunner.run 如上图

从代码呈现的调用顺序来看，Step 之间的先后顺序如下图：

![](/public/upload/docker/jib_BuildSteps_process.png)

相邻的两个同色表示没有依赖关系，不同色表示有依赖关系。上图只展现了相邻Step的并行度，实际执行时，并发度可以更高

![](/public/upload/docker/jib_BuildSteps_parallel.png)

看到这张图，我们埋几个疑问：

1. 常规情况下 这种Step 可组合式的逻辑如何实现？责任链模式，pipeline
2. jib 为什么没有选择常规方式实现？
	 
![](/public/upload/docker/jib_BuildSteps.png)
		
![](/public/upload/docker/jib_StepsRunner.png)

BuildSteps 和 StepRunner 都分为构造和执行两个部分

1. BuildSteps 分别针对 DockerDaemonImage、RegistryImage、TarImage 等TargetImage 类型，提供了对应的静态构造/工厂方法。
2. StepsRunner 针对每一个步骤 提供了静态构造方法，但StepsRunner更像一个builder，只不过一般builder 类每次setXXX 是设置属性，StepsRunner 每次setXX 是扩充其持有的 stepsRunnable （Runnable 实现类），也就是扩充Runnable 的逻辑内容。stepsRunnable 是一个runnable 引用， 每一次setXX 都会将其指向一个更复杂的runnable 匿名实现类。

![](/public/upload/docker/jib_StepsRunner_process.png)

以步骤比较少的 `BuildSteps.forBuildToDockerDaemon` 为例

	public static BuildSteps forBuildToDockerDaemon(DockerClient dockerClient, BuildConfiguration buildConfiguration) {
	    return new BuildSteps(
	        DESCRIPTION_FOR_DOCKER_DAEMON,
	        buildConfiguration,
	        StepsRunner.begin(buildConfiguration)
	            .pullBaseImage()
	            .pullAndCacheBaseImageLayers()
	            .buildAndCacheApplicationLayers()
	            .buildImage()
	            .finalizingBuild()
	            .loadDocker(dockerClient));
	}

StepsRunner 部分实现如下      

	public class StepsRunner {
	  	private final Steps steps = new Steps();	// 此处的steps 就是一个holder，StepsRunner 设定某个Step时，用以检查其依赖的前置Step 是否已被设置
		private Runnable stepsRunnable = () -> {};
		public StepsRunner pullBaseImage() {
			// 这个匿名runnable 干了两件事：1. 给steps成员赋值 2. PullBaseImageStep 构造方法会触发 Step 的执行
	    	return enqueueStep(() -> steps.pullBaseImageStep = new PullBaseImageStep(...));
	  	}
		private StepsRunner enqueueStep(Runnable stepRunnable) {
		    Runnable previousStepsRunnable = stepsRunnable;
		    // 扩容一个runnable 逻辑
		    stepsRunnable =
		        () -> {
		          previousStepsRunnable.run();
		          stepRunnable.run();
		        };
		    stepsCount++;
		    return this;
		}
	}


注意 每一个 XXStep 都是一个 AsyncStep 实现， `new PullBaseImageStep(...)` 便触发了该Step的实际执行。

那么问题来了，既然是AsyncStep，若是依赖 前置Step的执行结果，而前置Step 还未执行完毕怎么办？每一个AsyncStep 的大致组成是
 
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
 		public XX call() throws ExecutionException{...}
 	}
 	
 
最有意思的部分就是， 本来十几个step 具有复杂的依赖关系，有的需要同步执行，有的可以异步执行。而通过代码的腾挪， 表面调用起来却是平铺直叙的。
 
从另一个角度说，代码调用可以是顺序的，但业务不是顺序的。代码呈现的感觉跟实际的执行 不是一回事（也可以说，我们以前的方法太笨了）。


### 回顾下流程驱动

AsyncStep 接口官方注释：Holds the future for an asynchronously-running step. Implementations should:

1. Be immutable
2. Construct with the dependent AsyncSteps and submit a Callable to the ListeningExecutorService to run after all its dependent  AsyncSteps (for example,by using  `Futures.whenAllSucceed`)
3. Have getFuture return the submitted future

也就是说，在AsyncStep 实现类的构造方法中，并已经有了以下逻辑

1. 等待依赖的Step 执行完毕
2. 实现类自己是一个Callable，将自己提交给listeningExecutorService，使其执行自己的call 方法。换句话，实现类的call 方法被执行时，所有的依赖Step 已经执行完毕了。

也就是BuildSteps.run ==> StepsRunner.run ==> runnable.run，runnable.run 的本质是

	{
		new XXStep();
		new XXStep();
		new XXStep(buildConfiguration,dependent AsyncSteps, ListeningExecutorService)
		...
	}

所有的Step 都是异步执行，但因为持有了dependent AsyncSteps 的应用，造成了半同步半异步的效果。

从这个角度看，BuildSteps 和 StepsRunner 的分工还蛮合理的，一个对象只干一点事情

1. BuildSteps 负责 根据上层业务 指定 需要的Step，跟业务关系比较大
2. StepsRunner 负责 将Step 组装在一起，并指明Step 的依赖关系（依赖关系本身与业务无关）
3.  runnable.run 负责实际的驱动执行

[系统设计的一些体会](http://qiankunli.github.io/2018/09/28/system_design.html) 提到：要分得清楚访问代码、业务代码、存储代码、胶水代码各自应在哪些层级，它们应该是什么角色。**在这里，所有Step 串行调用异步执行是本质**（异步执行有调用线程和执行线程之分，所以串行调用和异步执行不冲突，BuildSteps 和 StepsRunner 的静态构造方法和Builder 模式是访问或胶水代码，提高了可读性。

每次`new XXStep()`可以理解为另起线程 执行一个Step。就像`ExecutorService.submit(()-> System.out.println("run in new thread"))` 是一样的。


再换一个角度说，我们看下 rxnetty 的一些代码，充分体现“程序=逻辑+控制”，逻辑与控制的分离。 

	RxNetty.createHttpGet("http://localhost:8080/error")
	               .flatMap(response -> response.getContent())
	               .map(data -> "Client => " + data.toString(Charset.defaultCharset()))
	               .toBlocking().forEach(System.out::println);
	               
	               
在这个例子中，我直觉上的实现是不同的 TargeImage 对应一个Step数组，BuildSteps.run 就是 循环执行`ExecutorService.submit(step)`。**在这个步骤中，每个Step 是客体，等着被构造、被初始化、被执行。而jib 则是将ExecutorService 作为Step的成员，Step 有着更强的把控力， 对外也隐藏了异步的感觉**。优劣还需进一步体会。 
      	

###  和maven 集成

[博客园首页联系订阅管理
随笔 - 90  文章 - 0  评论 - 234
Maven提高篇系列之（六）——编写自己的Plugin](http://www.cnblogs.com/davenkin/p/advanced-maven-write-your-own-plugin.html)

`mvn compile jib:build` 触发 BuildImageMojo execute 方法执行



## 从Jib 中学到的

1. jib 重度使用了Builder 模式， 还Builder 套Builder（Builder 分层），本质是解决 当配置项过多时，通过将配置归类等方式 使得框架入口更易懂
2. 当一个流程有多个Step

	1. 如何聚合这些Step
	2. 若是支持Step 异步执行的话，如何处理它们之间的依赖关系
3. 摸清代码的意图，是理解源码的第一步



## 一些实践

以jib-demo 项目为例，执行

	mvn com.google.cloud.tools:jib-maven-plugin:0.10.1:build -Djib.from.image=harbor.test.xxx.com/common/runit-jdk8-tomcat8 -Djib.from.auth.username=xx -Djib.from.auth.password=xx -Djib.to.image=harbor.test.xxx.com/test/jib-demo:20181206-154143 -Djib.to.auth.username=xx -Djib.to.auth.password=xx -f=pom.xml -Djib.useOnlyProjectCache=true -Djib.container.appRoot=/usr/local/tomcat/webapps/jib-demo
	
输出为：

	[INFO] Getting base image harbor.test.xx.com/common/runit-jdk8-tomcat8...
	[INFO] Building dependencies layer...
	[INFO] Building resources layer...
	[INFO] Building classes layer...

如果jib-demo 依赖一些snapshots jar，输出为

	[INFO] Getting base image harbor.test.xxx.a.com/common/runit-jdk8-tomcat8...
	[INFO] Building dependencies layer...
	[INFO] Building snapshot dependencies layer...
	[INFO] Building resources layer...
	[INFO] Building classes layer...


如果我们分别查看 `docker history harbor.test.xx.com/test/jib-demo:20181206-154143` 以及  `docker history  harbor.test.xx.com/common/runit-jdk8-tomcat8` 会发现两者大部分相似，只有最后的三个部分不同

	[root@docker1 ~]# docker history harbor.test.xx.com/test/jib-demo:20181206-172214
	IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
	8317485ce8ec        48 years ago        jib-maven-plugin:0.10.1                         846B                classes
	<missing>           48 years ago        jib-maven-plugin:0.10.1                         5.6kB               resources
	<missing>           48 years ago        jib-maven-plugin:0.10.1                         6.25MB              dependencies
	<missing>           3 days ago          /bin/sh -c chmod +x /etc/service/tomcat/run     406B		
	
这正是jib `在harbor.test.xx.com/common/runit-jdk8-tomcat8` 之上添加的dependencies 、resources  和 classes layer。

个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)