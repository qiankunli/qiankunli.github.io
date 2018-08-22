---

layout: post
title: 测试环境docker化实践2
category: 技术
tags: Docker
keywords: Docker,macvlan

---


## 简介（未完成）

[Docker在沪江落地的实践](https://hujiangtech.github.io/tech/%E5%90%8E%E7%AB%AF/2017/03/21/Docker.html)文中列举了docker 方案落地要解决的几个问题。


将一个web app部署到docker 容器上。除了编排问题之外

1. 网络模型的选择。公司测试环境遍布两三个办公楼，macvlan直接依赖两层，对交换机配置要求较高。
2. docker+os 对swap 分区的支持。现在的表现是，同样规模的物理机，直接部署比通过docker部署支持的服务更多，后者比较吃资源。
3. marathon + mesos 方案通过zk 沟通，mesos-slave 有时无法杀死docker-slave

mesos+marathon方案也取得很多经验：

1. macvlan 直接在ip 上打通 容器与物理机的访问，使用起来比较便捷。
2. marathon app + instance + cpu + memory 对app的抽象与控制比较直观，用户输入与实际marathon.json 映射比较直观。

应用背景：web项目，将war包拷贝到tomcat上，启动tomcat运行


## docker build 比较慢


jenkins流程

1. maven build war
2. 根据war build image
3. push image
4. 调度marathon

问题

docker build 慢，docker push 慢
	
build 慢的原因：

1. level 多
2. war 包大

build 慢的解决办法

1. 在jenkins机器上，docker run -d, docker cp,docker cimmit 的方式替代dockerfile
2. 先调度marathon在目标物理机上启动容器，然后将war包拷到目标物理机，进而拷贝到容器中，启动tomcat

	* 优点：完全规避了docker build
	* 缺点：每个版本的war包没有镜像，容器退化为了一个执行机器， 镜像带来的版本管理、回滚等不再支持

	
[Docker 持续集成过程中的性能问题及解决方法](http://oilbeater.com/docker/2016/01/02/use-docker-performance-issue-and-solution.html)

另一种方案：

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


打tag


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

## 性能不及物理机(未完成)

表现为过快的耗尽物理机资源：

cpu设置问题

[Docker: 限制容器可用的 CPU](https://www.cnblogs.com/sparkdev/p/8052522.html)

[Docker 运行时资源限制](http://blog.csdn.net/candcplusplus/article/details/53728507)

## 镜像的清理

起初，集群规模较小，harbor 和其它服务部署在一起。后来镜像文件 越来越多，于是将harbor 单独部署。但必然，单机磁盘无法承载 所有镜像，此时

1. 集群改造
2. 清理过期不用的镜像

	清理时，应根据当前服务使用镜像的情况，保留之前几个版本。笔者曾粗暴的清理了一些镜像，刚好赶上断网，直接导致部分服务重新部署时拉不到镜像。

## 网络方案选择与多机房扩展问题（未完成）

