---
layout: post
title: virtualbox 使用
category: 技术
tags: Tool
keywords: virtualbox 虚拟机
---

## 使用VBoxManage.exe ##

VBoxManage.exe一般在virtualbox 安装目录下，以为virtualbox vm 添加sharedfolder 为例，

`VBoxManage sharedfolder add <vmname> --name <name> --hostpath <hostpath> [--transient] [--readonly] [--automount]`

这样，便可以使用脚本为virtualbox中的某一个虚拟机配置sharedfolder。

## import和export vm ##

如果你想将自己PC上virtualbox 的某一个 vm 复制给他人使用，便可以先将该vm export为`.ova`文件，复制到他人PC后，使用virtualbox import该ova文件即可。**注意**，操作过程中须勾选`reinitalize the MAC address of all network cards`选项。


## 克隆 vm ##

如果你需要对某个vm进行某些操作，但这些操作会影响vm现有的环境，便可以选择先克隆该vm，在新vm上进行必要的操作。

**注意**，操作过程中须勾选`reinitalize the MAC address of all network cards`选项。

## Vagrant

其实将vagrant写在本章是不公平的，Vagrant是一个创建和配置轻量级，可复制，可移植开发环境的工具。我知道的用途主要有：

1. 假如你有一个virtualbox虚拟机软件，自己配置了一个cenots vm，你对这个vm做一些特殊配置。这一切步骤，都可以通过Vagrantfile**描述出来**，并在其他人的电脑上重现你所做的所有工作。
2. 你已安装virtualbox，想create一个centos vm

    - 传统方式，你需要下载一个iso，然后一步步安装，配置
    - 使用vagrant，使用vagrant命令从网络上下载镜像，使用vagrant一键安装和登陆