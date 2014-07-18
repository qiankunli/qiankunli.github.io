# haproxy快速入门 #

## 前言 ##

本文是关于我对docker的一些理解，将持续更新，如有错误和建议，请及时反馈到qiankun.li@qq.com。具体的细节请参见官方文档 [http://www.haproxy.org/](http://www.haproxy.org/)

## 概述 ##

HAProxy是一款提供高可用性、负载均衡以及基于TCP（第四层）和HTTP（第七层）应用的代理软件，HAProxy是完全免费的、借助HAProxy可以快速并且可靠的提供基于TCP和HTTP应用的代理解决方案。


## 安装 ##

### 源码安装 ###

1. 准备
  * linux
  * haproxy-1.4.8.tar.gz 　　download website [http://www.haproxy.org/#down](http://www.haproxy.org/#down)

2. 安装过程
    
    以安装到`/usr/local/`下为例
    
	    $ cd /usr/local
    	$ cp ../haproxy-1.4.8.tar.gz .
    	$ tar -zxvf haproxy-1.4.8.tar.gz
		$ cd haproxy-1.4.8
		$ uname -a      # check version of kernel
		$ make TARGET=linux26 PREFIX=/usr/local/haproxy 	#TARGET是内核版本，2.6就写作26
		$ make install PREFIX=/usr/local/haproxy

	到/usr/local/haproxy/sbin目录下执行`haproxy`,如果能够看到帮助代码，说明安装成功。

## 一个简单地小例子 ##

我们先讲一个简单地例子，由此建立对haproxy的感性认识，然后讲述haproxy的一些配置  

