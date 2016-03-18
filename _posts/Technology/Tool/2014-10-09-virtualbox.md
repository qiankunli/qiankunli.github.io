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

## 虚拟机网络设置

### nat + host-only

- 实现主机与虚拟机互访，则需要配置host-only
- 实现虚拟机访问互联网，则需要配置nat

所以，在virtualbox中，虚拟机一般配置双网卡（nat + host-only），bridge虽然是两者皆可，但会占用现有局域网的ip资源，不建议使用。

![Alt text](/public/upload/tool/network.png) 

virtualbox host-only的网络地址一般为`192.168.56.0/24`，并且按照上图的配置，virtualbox**有时**会将虚拟机的默认网关设置为`192.168.56.1`，导致vm无法访问互联网，所以需要更改路由表设置，ubuntu下`/etc/network/interfaces`文件内容如下所示：

    # This file describes the network interfaces available on your system
    # and how to activate them. For more information, see interfaces(5).
    # The loopback network interface
    auto lo
    iface lo inet loopback
    # The primary network interface
    auto eth0
    iface eth0 inet dhcp
    gateway 10.0.2.2
    netmask 255.255.255.0
    
    # 如果路由表配置正常，则不需要以下两行配置
    # 删除默认网关
    up sudo route del default
    # 添加nat网关
    up sudo route add default gw 10.0.2.2
    
    auto eth1
    iface eth1 inet static
    address 192.168.56.151
    gateway 192.168.56.1
    netmask 255.255.255.0
    

`192.168.56.1 ` 貌似不是`192.168.56.0/24`网络的网关
    
### internal network

internal network让各台虚拟机处于隔离的局域网内，只让它们相互通信，与外界（包括宿主机）隔绝。

虚拟机向virtualbox内置的dhcp服务请求ip，默认得到的地址是ipv6的

可以执行`dhclient eth0`或`dhcpcd eth0`来获取ipv4地址

http://superuser.com/questions/237057/how-do-i-make-ubuntu-server-get-ipv4-address

## ubuntu 下使用Host-only网络

ubuntu下使用virtualbox，默认是无法直接使用Host-only网络的，需要

![Alt text](/public/upload/tool/add_host_only_driver.png) 



