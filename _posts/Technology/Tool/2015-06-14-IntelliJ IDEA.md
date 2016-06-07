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