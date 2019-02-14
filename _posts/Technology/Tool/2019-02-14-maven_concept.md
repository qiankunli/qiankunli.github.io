---

layout: post
title: maven的基本概念
category: 技术
tags: Tool
keywords: maven

---

## 简介

* TOC
{:toc}

本文主要是针对 [How does a maven repository work?](https://blog.packagecloud.io/eng/2017/03/09/how-does-a-maven-repository-work/#snapshot-repositories) 的整理

文中以 client 作为贯穿通篇的example

	<dependency>
		<groupId>io.packagecloud</groupId>
		<artifactId>client</artifactId>
		<version>3.0.0-SNAPSHOT</version>
	</dependency>

## 一个Artifacts 的各种形态/表示

1. Maven Coordinates(坐标)，consist of groupId, artifactId, and version fields,allow you precisely specify a particular dependency in an absolute way like geographical coordinates（地理坐标）.
2. How does maven locate and resolve dependencies? Unlike other repository formats (APT, YUM Rubygems, there is no main index file that enumerates all possible artifacts available for that repository. **Maven uses the coordinates values for a given dependency to construct a URL** according to the maven repository layout. **URL construction scheme** 格式： `/$groupId[0]/../${groupId[n]/$artifactId/$version/$artifactId-$version.$extension`

以guava为例

    <dependency>
		<groupId>com.google.guava</groupId>
		<artifactId>guava</artifactId>
		<version>27.0.1-jre</version>
	</dependency>

其对应maven 中央仓库的 url 地址为  `http://central.maven.org/maven2/com/google/guava/guava/27.0.1-jre/`。guava-27.0.1-jre.jar   的具体下载地址就是上文提到的URL construction scheme`http://central.maven.org/maven2/com/google/guava/guava/27.0.1-jre/guava-27.0.1-jre.jar`

浏览器打开该url，可以看到如下目录结构

	../
	guava-27.0.1-jre-javadoc.jar                      2018-11-19 19:05   6642684  
	guava-27.0.1-jre-javadoc.jar.asc                  2018-11-19 19:05       488  
	guava-27.0.1-jre-javadoc.jar.md5                  2018-11-19 19:05        32  
	guava-27.0.1-jre-javadoc.jar.sha1                 2018-11-19 19:05        40  
	guava-27.0.1-jre-sources.jar                      2018-11-19 19:05   1633014  
	guava-27.0.1-jre-sources.jar.asc                  2018-11-19 19:05       488  
	guava-27.0.1-jre-sources.jar.md5                  2018-11-19 19:05        32  
	guava-27.0.1-jre-sources.jar.sha1                 2018-11-19 19:05        40  
	guava-27.0.1-jre.jar                              2018-11-19 19:05   2746650  
	guava-27.0.1-jre.jar.asc                          2018-11-19 19:05       488  
	guava-27.0.1-jre.jar.md5                          2018-11-19 19:05        32  
	guava-27.0.1-jre.jar.sha1                         2018-11-19 19:05        40  
	guava-27.0.1-jre.pom                              2018-11-19 19:05      8282  
	guava-27.0.1-jre.pom.asc                          2018-11-19 19:05       488  
	guava-27.0.1-jre.pom.md5                          2018-11-19 19:05        32  
	guava-27.0.1-jre.pom.sha1                         2018-11-19 19:05        40      

可以看到 针对 javadoc、sources、pom 和 jar 都有多个校验和文件。jar 文件本身称为Primary Artifacts，javadocs 和 sources 称为Secondary artifacts or “attached artifacts”, 在`<dependency>` 中体现为 classifier, unlike a primary artifact, a secondary artifact is not expected to have a remote pom and has thus never has any dependencies.（也即是说 Primary Artifacts 有喽）

因为maven 官方中央仓库不存放snapshot jar， 对于snapshot jar，除了上述文件之外，还会有一个maven-metadata.xml 文件

	maven-metadata.xml                                  
	maven-metadata.xml.md5                              
	maven-metadata.xml.sha1   

**也就是当你知道一个 文件的坐标（也就是`<dependency>` 内容），再加上一个Repository url，你就可以得到 这个jar 相关的一切（jar、pom、sources等文件及校验和）**。 

Just how your own Maven project has a pom.xml file listing its main dependencies, those dependencies also have a remote pom file serving a similar purpose. Maven uses this file to figure out what other dependencies to download. 假设你的项目依赖guava，maven 会下载guava的 pom文件，解析guava 的意思，进而下载guava 依赖的所有jar。

## Repository

**A Maven repository is wherever these constructed artifact URLs live**（Repository 就是给一个url 能够拿到的文件的地方）. Most of the time, this is a Web server with a `/maven2` document root, but it can actually be any protocol Maven has a transport plugin for.

1. The local repository, Before Maven attempts to download a particular artifact from a remote repository it checks the local repository. This is usually located at `$HOME/.m2/repository`. **The local repository follows the same standard repository layout as remote repositories**（也就是url 除了前缀不同，其它都是一样的）.
2. Remote repositories, Remote repositories are defined in your project’s pom.xml file under the <repositories/> section.  猜测 maven Repository setting.xml 和 项目的pom.xml 是合并的关系

there are two features that can be enabled on repositories, even at the same time.

1. Release repositories, his is enabled by default on all defined repositories , These are artifacts that once published to a coordinate, must not be changed.
2. SNAPSHOT repositories，多了maven-metadata.xml 和 snapshotVersion 的概念

	1. 每一个 version （`3.0.0-SNAPSHOT`）会对应 多个artifact 文件，每个文件对应一个 snapshotVersion（对应3.0.0-20161003.234325-2），Using the value of that `<snapshotVersion>` as the $version in our URL construction scheme(`/$groupId[0]/../${groupId[n]/$artifactId/$version/$artifactId-$version.$extension`)，比如获取jar的url变成了 `/io/packagecloud/client/3.0.0-SNAPSHOT/client-3.0.0-20161003.234325-2.jar` 。可以看作`3.0.0-SNAPSHOT` 中的SNAPSHOT 在应用时会被替换为 时间戳

	2. maven-metadata.xml, In order to determine the the latest artifact to download for a particular SNAPSHOT version, Maven uses the Standard Repository Layout to locate a `maven-metadata.xml` file for that dependency. As more snapshot artifacts are pushed to 3.0.0-SNAPSHOT, **the maven-metadata.xml will always get updated to reflect the latest `<snapshotVersion>` to use.** maven-metadata.xml永远指向 最新的snapshotVersion

There are two snapshot “styles” that Maven can use.

1. Unique Snapshots, These are the snapshot versions detailed in the example above, they use a high resolution timestamp as a version and clients must a maven-metadata.xml file to resolve the latest. This is the only snapshot style supported by Maven 3.
2. Non-Unique Snapshots, When this behavior is selected, there is no maven-metadata.xml file that is used, The artifact is resolved just like any other（可以理解为跟release的处理机制一样）. 


回顾一下

1. 上文主要提到了几个概念：Coordinates、Artifacts（Primary、Secondary）、Checksums 、artifact URL、Repository等
2. 基于这些基本概念，Repository 对Artifact进行了组织
3. 基于这些基本概念和约定，maven客户端 与 Repository 进行协作

## maven客户端与 Repository 的协作——maven 更新snapshot 的原理

[跟踪Maven更新Snapshot依赖包时的操作](https://www.cnblogs.com/zhangqingsh/archive/2013/04/08/3006723.html) 文中以 构建 `com.my.testu:testu:1.0.1-SNAPSHOT` 为example

运行`mvn package -U`

1. 从公司的Maven服务器上下载maven-metadata.xml，重命名为“maven-metadata-<RepositoryID>.xml”，并保存到本地仓库相应目录。
2. 比较maven-metadata-local.xml与maven-metadata-<RepositoryID>.xml中的lastUpdated时间戳的值。
3. 如果maven-metadata-local.xml中的时间戳比较大，则终止。
4. 如果maven-metadata-<RepositoryID>.xml中的时间戳较大，则从公司Maven服务器上下载最新版本。即：testu-1.0.1-20130407.081828-34.jar。这个过程分两步：

	1. 下载testu-1.0.1-20130407.081828-34.jar到本地Maven仓库。
	2. 将testu-1.0.1-20130407.081828-34.jar复制一份，覆盖掉原先的testu-1.0.1-SNAPSHOT.jar。
	
	也就是说，如果Maven从远程仓库下载了最新的SNAPSHOT发布包的话，那么最新的待时间戳的包和xxx-SNAPSHOT包是完全一样的。

## 实践

笔者碰到过一个问题：假设存在开发人员A和B，一个abc-0.0.1-snapshot.jar（A负责维护） 和demo项目（B负责维护），demo项目依赖abc-0.0.1-snapshot.jar。A改动 abc代码且未升版本号，mvn deploy 到artifactory.ximalaya.com 后。B  “mvn package” 编译 demo 项目时，如何 自动拉取到 最新的 abc-0.0.1-snapshot.jar？各种材料显示 “mvm package --update-snapshots” 可以做到，但在笔者的工作环境中update-snapshots 貌似无效。

经过各种求助，最终确认是maven remote Repository 的Maven Snapshot Version Behavior 设置为Non-Unique 的缘故，将其改为Unique 即可。

