---

layout: post
title: IntelliJ IDEA使用
category: 技术
tags: Tool
keywords: IntelliJ IDEA

---

## 中文乱码

C:\Program Files (x86)\JetBrains\IntelliJ IDEA Community Edition 14.1.2\bin\idea.exe.vmoptions

添加`-Dfile.encoding=UTF-8`

file ==> settings 搜索 encoding，找到file encoding 改为utf-8


## 快捷键的选择

1. 习惯eclipse的同学可以使用Eclipse
2. 在Ubuntu环境下的同学可以使用Default for GNOME

## 使用maven创建项目

在maven命令行模式下，我们一般用`mvn archetype:generate`创建项目的目录结构，比如`mvn archetype:generate -DgroupId=org.lqk -DartifactId=demo -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false -DarchetypeCatalog=local`
其中配置` -DarchetypeCatalog=local`的原因是`mvn archetype:generate`运行时会下载`http://repo1.maven.org/maven2/archetype-catalog.xml`，而国内无法下载，因此需要提前下载到本地，放在`～/.m2`目录下。并告诉maven使用本地archetype-catalog.xml件。

在Intellij中，则需要添加配置

![Alt text](/public/upload/tool/intellij_maven.png) 

通常情况下，先用maven命令行创建完项目，再使用intellij导入，也是比较方便的。

## ubuntu下输入法无法显示问题

参见`http://dachengxi.blog.51cto.com/4658215/1747124`，注意脚本插入的位置。

## 配置jdk

有时我们的pc装了多个jdk，并且默认的JAVA_HOME并不适合intellij。

intellij在寻找可用的jdk时会**按序**查找多个环境变量，因此设置某个环境变量的值，即可设置intellij使用的jdk。参见`https://intellij-support.jetbrains.com/hc/en-us/articles/206544879-Selecting-the-JDK-version-the-IDE-will-run-under`

## 配置jdk vm

idea64.exe.vmoptions

    -Xms512m
    -Xmx2048m
    -XX:MaxPermSize=1024m
    -XX:ReservedCodeCacheSize=240m
    -XX:+UseConcMarkSweepGC
    -XX:SoftRefLRUPolicyMSPerMB=50
    -ea
    -Dsun.io.useCanonCaches=false
    -Djava.net.preferIPv4Stack=true
    -XX:+HeapDumpOnOutOfMemoryError
    -XX:-OmitStackTraceInFastThrow
    
前三个值要调大一点，否则你intellij会慢的要死。
    
## 配置go语言开发环境

基本环境，windows下如果使用goxxx.msi安装，则会自动添加GOROOT和GOPATH环境变量，这两个变量很重要，作用参见其它文档。

intellij 安装 go语言插件，这个网上有很多，可以参见`http://blog.csdn.net/qinxiandiqi/article/details/50319953`

插件安装完毕后，创建一个go project，并创建相关目录。

    projectA
        src
        pkg
        bin

进入project structure,将src设置为Sources目录

将projectA的目录添加到该project的GOPATH变量中，`project settting ==> Go ==> Go Libraries ==> Project libraries ==> + ==> projectA dir`此时，该GOPATH值只对projectA生效。
