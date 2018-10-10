---

layout: post
title: 活用linux 命令
category: 技术
tags: Linux
keywords: linux命令 

---

## 简介（持续更新）

## 删除符合正则规则的文件

假设存在以下文件


	album-listener.log.2018-01-30  
	album-facade.log             
	album-facade.log.2017-12-26  
	album-facade.log.2018-01-23  
	album-facade.log.2018-01-24  
	album-facade.log.2018-05-23  
	album-facade.log.2018-07-01  
	album-facade.log.2018-07-15  
	album-facade.log.2018-07-16  
	album-facade.log.2018-07-31

因为磁盘空间有限，先输出2017年的日志

1. 直观想法是rm，但rm 不支持正则表达式
2. 则可以先找到文件，在通过xargs 串联 rm，`ls | grep ".*2018-03.*" | xargs rm -rf`


