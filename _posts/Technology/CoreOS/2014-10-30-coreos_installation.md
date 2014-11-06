---
layout: post
title: CoreOS安装
category: 技术
tags: CoreOS
keywords: CoreOS Docker
---

## CoreOS是什么 ##

### 从docker讲起

笔者最早学docker的时候，为在windows上使用docker，需要下载安装boot2docker.exe,这个应用程序在virtualbox上创建一个boot2docker-vm,这个vm其实就是一个极精简的linux，并安装了docker而已，好像这个linux就是为docker准备的。这是笔者第一次接触这样的linux系统。

后来听说有一个coreos，也是这样一个极简的linux系统，并且更加专业。比如它有很多管理docker container的工具，诸如fleet等。所以，我开始尝试了解coreos。

### 从下一代服务器操作系统讲起


经过一定的了解，将coreos理解为一种为docker而生的linux非常不公平。恰恰相反，docker是coreos实现其野心的重要手段。随着技术的发展，客户端操作系统越来越浏览器化，笔者及身边的很多朋友已经开始使用在线IDE以及图表工具了。与传统软件相比，省去了繁杂的安装过程。而服务器端操作系统则在走这样一段路：

1. 
