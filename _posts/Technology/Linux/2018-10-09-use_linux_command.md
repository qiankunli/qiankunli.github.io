---

layout: post
title: 活用linux 命令
category: 技术
tags: Linux
keywords: linux命令 

---

## 简介（持续更新）

## yum 升级回退

[在RHEL/CentOS系统上使用YUM history命令回滚升级操作](http://os.51cto.com/art/201801/563966.htm)

	yum history undo 13

## 找到第一次报错的位置 并显示前后几行

	grep ERROR  xxx.log -m 1 -A 10

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

后来试验了一下，`rm -rf *2018-03*` 也是可以的。

## 在一个目录下寻找包含某个关键字的文件

`grep -rn "交换机" dir`