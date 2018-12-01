---

layout: post
title: docker环境下的持续构建
category: 技术
tags: Docker
keywords: 持续交付

---

## 简介

* TOC
{:toc}

建议先看下 常规自动化构建的 理念和历史发展 [maven/ant/gradle使用](http://qiankunli.github.io/2015/07/13/maven.html)

![](/public/upload/docker/ci_in_docker.png)


## 几个问题

从上图可以看到，”代码 ==> 镜像“涉及到多个步骤，构建过程的过程中涉及到几个基本问题：

1. 用不用jenkins? 怎么用jenkins？是否容器化jenkins？
2. dockerfile 和代码是否在一起？
3. 提交代码要不要自动构建？
4. 构建过程要不要放到容器？
5. 构建和发布的结合部 

	* 发布时构建
	* 平时代码提交即构建，发布时从已构建好的镜像进行部署。[基于容器的自动构建——Docker在美团的应用](https://www.jianshu.com/p/a1f371d9e0c5)

一个完整的构建过程 有两个考虑维度

1. 用不用一些组件
2. 怎么用一些组件

	* 比如maven，是直接调用，还是放在docker中使用；
	* 比如docker build，是直接调用，还是docker in docker。


## 传统的基于jenkins构建

流程

1. maven build war
2. 根据war build image
3. push image
4. 调度marathon/k8s

该方案的问题：

1. build 慢
2. push 慢
3. 发布是一个有明显高峰低谷的操作，高峰时jenkins 压力很大
	* 中午吃饭前
	* 下午五六点：干了一天了，基本出活儿了

### docker build 慢

build 慢的原因：

1. layer 多
2. war 包大
3. jenkins 单机瓶颈，同时执行十几个`docker build -t ` docker 也会很卡

build 慢的解决办法

1. 在jenkins机器上，docker run -d, docker cp,docker cimmit 的方式替代dockerfile
2. 先调度marathon在目标物理机上启动容器，然后将war包拷到目标物理机，进而拷贝到容器中，启动tomcat

	* 优点：完全规避了docker build
	* 缺点：每个版本的war包没有镜像，容器退化为了一个执行机器， 镜像带来的版本管理、回滚等不再支持

3. 将git 代码 变成镜像的事情，可以交给专门的容器做，理由

	* 破解jenkins 单点问题
	* 集群资源通常是富余的，而build 任务具有明显的临时性特征，可以充分的将富余的资源利用起来。

### docker push 慢
	
从两个方面解决：

1. dockerfile 技巧，比如减少layer等  [Docker 持续集成过程中的性能问题及解决方法](http://oilbeater.com/docker/2016/01/02/use-docker-performance-issue-and-solution.html)
2. 镜像预热，p2p 镜像分发，[美团容器平台架构及容器技术实践](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651749434&idx=1&sn=92dcd59d05984eaa036e7fa804fccf20&chksm=bd12a5778a652c61f4a181c1967dbcf120dd16a47f63a5779fbf931b476e6e712e02d7c7e3a3&mpshare=1&scene=23&srcid=11183r23mQDITxo9cBDHbWKR%23rd)

## 如何将代码变成image

### 步骤

1. 代码 ==> 可执行文件，比如war/jar/二进制文件等
2. 生成dockerfile
3. docker build 得到镜像
4. docker push 镜像到仓库

难点在于

1. 每个步骤都需要配置
2. 将步骤驱动起来，成为一个完整的步骤

### 可用方案

这是一个难题，难点不在如何变成jar，难在如何让一群完全没有docker 经验的人，按照指示，将代码变成jar/war，再变成image

1. jenkins + 变量方案，jenkins 执行maven 代码变成jar/war 之后，用变量指定jar/war 位置，根据变量构建Dockerfile，制作镜像。该方案在实施过程中碰到以下问题
	
	* 新手习惯于克隆老手的已有项目，大量的配置错配
	* 大量的变量散落在各个jenkins 项目中，无法被统一管理
	* 有些项目生成的jar/war 没有包含依赖jar，而依赖jar目录各式各样
	* 每一个步骤都容器化时，比较复杂。尤其是 build 和push 过程，还需要 docker in docker
	
2. 阿里云效方案，用户在代码中附属一个配置文件，由命令根据文件打成jar/war，再制作为image
3. google jib 方案，使用maven 插件，将过程内置到 maven build 过程中，并根据image registry 格式，直接push到registry 中。  jib 应用参见 [jib源码分析及应用](http://qiankunli.github.io/2018/11/19/jib_source.html)
4. 假设一个是maven项目，项目根目录下放上Dockerfile、marathon.json/xxx-pod.yaml 文件，自己写一个脚本（比如叫deploy.sh) 用户`maven package` 之后，执行`deploy.sh` 。该方案有以下问题

	* 直接暴露Dockfile 和 marathon.json 对于一些新手来说，难以配置，可能要将配置文件“封装”一下

灵活性和模板化的边界在哪里？可以参见下 CNI 的设计理念。

从目前来看，针对 java 项目来说，jib 是最优的。在项目目录下直接执行 `mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build -Djib.to.image=<MY IMAGE>` 便可以将代码转换为镜像。

### 如何驱动这些步骤

#### jenkins 作为驱动器的问题

1. 权限管理比较弱
2. 用户配置、操作无法拦截，也就不能做错误校验（或者很麻烦）。用户配置分散在各个jenkins中，也没有手段可以批量修改。
3. jenkins 一般作为一个公司的基础系统，集中式，项目很多，会很卡。当然，可以用docker、集群的方式解决。
4. 和流水线即代码的理念一对比，就显得很臃肿。

[流水线即代码](http://insights.thoughtworkers.org/pipeline-as-code/)以Jenkins为例，暂且不谈1.0版本无法直接支持流水线这一问题，为了支持构建、测试和部署等，我们一般会先手工安装所需插件，在多个文本框中粘贴大量shell/batch脚本，下载依赖包、设置环境变量等等。久而久之（实际上用不了多久），这台Jenkins服务器就变成无法替代（特异化）的“怪兽”了，因为没人清楚到底对它做了哪些更改，也不知道这些更改对系统产生了哪些影响，这时的Jenkins服务器就腐化成了Martin Fowler口中的雪花服务器(snowflake server)。雪花服务器有两点显著的特征：

1. 特别难以复现
2. 几乎无法理解

第一点是由于以往所做的更改并没有被记录下来（手工操作产生的配置漂移(configuration drift)），第二点则是由于绝大部分情况下散乱的配置是没有文档描述的，哪部分重要、哪部分不重要已经无从知晓，改动的风险很大。

当前实现[流水线即代码](http://insights.thoughtworkers.org/pipeline-as-code/)的CI/CD工具大体遵循了两种模式：

1. 版本控制
2. DSL（领域特定语言），基于groovy 或yaml 的，Jenkins2.0 允许我们在项目的特定目录下放置一个Jenkinsfile的文件，使用Groovy实现了一套描述流水线的DSL。

#### 流水线即代码

可以项目中 弄一个deploy.yaml 文件。yaml 文件的解释/执行器 应能根据 yaml 配置 自动完成 代码 到 镜像的所有工作。

	language:java
	creator: zhangsan
	war: target/xxx.war
	mem: 2048
	
对于yaml 执行器，根据yaml文件，执行对应的编译命令。比如对于java，拼凑 `mvn compile com.google.cloud.tools:jib-maven-plugin:0.10.0:build -Djib.to.image=<MY IMAGE>`  命令并执行

网上说的各种好处一大堆，笔者个人比较大的体会主要有：

1. 可见性，比如上例中，开发都和运维都清楚 服务被部署后，内存只有2G
2. 简单性，配置文件只描述目标，不关心实现，自然也就省掉了跟具体实现相关的配置，也不会有操作顺序不一致带来的问题。
3. 易推广，因为真的很多开发连jenkins是啥都不知道

[流水线即代码](http://insights.thoughtworkers.org/pipeline-as-code/)流水线即代码(Pipeline as Code)通过编码而非配置持续集成/持续交付(CI/CD)运行工具的方式定义部署流水线。

扩展：如果有一个配置，使得运维和开发约定好 如何处理路径映射，运维将这个配置同步到nginx中，一定很好玩。

## 其它细节

[Crafting perfect Java Docker build flow](https://codefresh.io/docker-tutorial/java_docker_pipeline/)

* Choosing a base Docker image for Java Application. glibc 问题
* Choosing the right Java Application server. Originally, Java server-side deployment assumes you have pre-configured a Java Web Server (Tomcat, WebLogic, JBoss, or other) and you are deploying an application WAR (Web Archive) packaged Java application to this server and run it together with other applications, deployed on the same server.Lots of tools are developed around this concept, allowing you to update running applications without stopping the Java Application server, route traffic to the new application, resolve possible class loading conflicts and more.



个人微信订阅号

![](/public/upload/qrcode_for_gh.jpg)