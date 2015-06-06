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


## host和vm共享文件夹

![Alt text](/public/upload/tool/share_folder1.png) 

![Alt text](/public/upload/tool/share_folder2.png) 

mount命令格式: mount [-参数][设备名称][挂载点]

sudo mount -t vboxsf git /home/docker/git