---
layout: post
title: CoreOS 安装
category: 技术
tags: CoreOS
keywords: CoreOS Docker
---

## 前言

本文旨在windows + virtualbox 环境下搭建一个coreos集群，自娱自乐，如有错误，欢迎大家指正。

下面提到的某些步骤需要翻墙。

官网上明确表示"It's highly recommended that you set up a cluster of at least 3 machines — it's not as much fun on a single machine"。而通常一个分布式系统，都会有一个分布式存储系统来信息共享，从而将集群各个节点紧密联系起来。在CoreOS中，etcd便担此重任。

从某种方式讲，创建一个CoreOS集群，工作量占大头的就是配置好各节点的etcd服务，或者说是创建一个etcd集群。

创建集群的一个关键是，说清楚集群都包含哪些节点，有以下几种方式：

1. 每个节点都有一个配置文件，列出当前集群的节点（IP或主机名）；
2. 指定一个节点，除该结点外，集群其它节点都向这个节点汇报自己的信息；
3. 大家都有一个身份标识，到第三方服务那里注册。

这几种方式各有利弊，第一种方式简单直接，但如果集群节点经常变化，那么第一种方式就不太合适了。而第二种和第三种，在etcd中分别被称为Static方式（通过制定 peers的IP 和端口创建）与 Discovery方式 （通过一个发现服务创建）。 

而搭建CoreOS集群，有以下几种办法：

1. 先为每个节点安装CoreOS系统，然后配置etcd服务（可以通过命令行，也可以修改配置文件）；
2. 在安装CoreOS时，就配置好etcd服务。即static方式或discovery方式。


## Static方式（通过ISO安装）

### 文件准备

1. 下载 [ISO][]
2. 使用puttygen等工具生成一对密钥。
3. 编写cloud-config.yaml

    假设需要配置三台主机，IP地址分别是：
    
    coreos1 172.17.8.101
    
    coreos2 172.17.8.102
    
    coreos3 172.17.8.103
    
    以coreos1的配置为例：

        #cloud-config
        hostname: coreos1 
        coreos:  
          etcd:
            name: coreos1
            peers: 172.17.8.102:7001 ## IP为172.17.8.102的主机不用配置该选项
            addr: 172.17.8.101:4001
            peer-addr: 172.17.8.101:7001
          units:
            - name: etcd.service
              command: start
            - name: fleet.service
              command: start
            - name: static.network
              content: |
                [Match]
                Name=enp0s8
                [Network]
                Address=172.17.8.101/24
                Gateway=172.17.8.1
                DNS=172.17.8.1
        users:  
          - name: core
            ssh-authorized-keys: 
              - 你的公钥 ##此处可以让三台主机使用同样的公钥
          - groups:
              - sudo
              - docker

编辑完毕后，请到[http://codebeautify.org/yaml-validator][]校验下yaml文件格式是否正确。

### 每台虚拟机安装过程

1. 根据下载ISO在virtualbox上创建一个虚拟机。并为其添加一个Host-Only网卡（网络地址要跟`cloud-config.yaml`中的一致）。
2. 虚拟机启动后，使用`sudo passwd root`命令配置root用户密码。
3. virtualbox新创的虚拟机默认使用NAT方式。因此可以配置端口映射，然后使用putty等shell工具登录该虚拟机。IP为localhost，端口是你刚刚端口映射配的，用户名是root，密码是你第二步配置的。
4. 现在我们以root用户登录了，使用`export http_proxy=http://xxxx`命令配置http_proxy代理，让你的机器可以访问外网。也可以不配置，但速度真的慢的要死。
5. 创建前面提到的cloud-config.yaml文件，运行`coreos-install -d /dev/sda -c ./cloud-config.yaml -v`命令（注意，不同虚拟机所用`cloud-config.yaml`文件不同），将coreos安装到虚拟硬盘，完毕后关闭虚拟机。
6. 配置虚拟机从硬盘启动（默认是从光盘启动的）。
7. 使用putty登录虚拟机，用户名为core，记得配置私钥喔。


## Discovery方式（通过Vagrant安装）

Discovery方式需要用到etcd官方的Discovery服务，当然，我们也可以自己搭建一个私有的Discovery服务。这种方式的主要优点是，能够自动感知到集群节点的增减。

### 文件准备

1. 下载[Vagrant][]并安装。
2. 获取coreos集群的Token
    
    `curl https://discovery.etcd.io/new`
   
3. 通过 Git下载官方的 Vagrant 仓库

    在git bash下执行`git clone https://github.com/coreos/coreos-vagrant.git`
    
    3.1. 进入coreos-vagrant目录，将 user-data.sample 和 config.rb.sample 两个文件各拷贝一份，并去掉 .sample 后缀，得到 user-data 和 config.rb 文件。
    
    3.2. 修改 config.rb 文件，配置 $num_instances 和 $update_channel 这两个参数。比如：
    
            $num_instances=3  ## 表示我们要创建3台主机
            $update_channel='stable'
   
    3.3. 修改user-data文件（类似于上述的cloud-config.yaml文件）

             #cloud-config 
                coreos:
                  etcd:
                      discovery: https://discovery.etcd.io/你的Token
                      addr: $public_ipv4:4001
                      peer-addr: $public_ipv4:7001
                  fleet:
                      public-ip: $public_ipv4
                  units:
                    - name: etcd.service
                      command: start
                    - name: fleet.service
                      command: start

        编辑完毕后，请到[http://codebeautify.org/yaml-validator][]校验下yaml文件格式是否正确。
     
     3.4. 将Vagrantfile文件中的“config.vm.box_version = ">= 308.0.1"”注释掉

4. 下载[box文件][]（本来不下载也可以的，但国内因为各种原因速度较慢，所以推荐用户使用浏览器等工具事先下载该文件，使用迅雷也是个不错的选择），将其放在coreos-vagrant目录下，并在其目录下执行

        vagrant box add coreos-stable coreos_production_vagrant.box
   

### 安装

进入coreos-vagrant目录，在git bash下执行`vagrant up`即可在virtualbox创建一个coreos集群了。

如何登录某台主机呢？执行`vagrant ssh-config --host core-01`，你可以看到core-01密钥文件被vagrant放到了哪里，然后就可以使用putty工具登录了。

## 推荐方式

上述第一种方式，安装过程中要下载文件（需访问外网），速度太慢。而discovery方式，需要自己搭建discovery服务（公有的discovery服务也需要能访问外网）。所有有了这个“使用vagrant + static”的方式。

### 准备过程

1. 下载[Vagrant][]并安装。
2. 通过 Git下载官方的 Vagrant 仓库

    在git bash下执行`git clone https://github.com/coreos/coreos-vagrant.git`
    
     2.1. 进入coreos-vagrant目录，将 user-data.sample 和 config.rb.sample 两个文件各拷贝一份，并去掉 .sample 后缀，得到 user-data 和 config.rb 文件。
    
     2.2. 修改 config.rb 文件，配置 $num_instances 和 $update_channel 这两个参数。比如：
    
            $num_instances=1
            $update_channel='stable'
        
      即，使用vagrant每次安装一台主机。
        
    2.3. 修改user-data文件，针对安装的主机，采用第一种方式的cloud-config.yaml文件内容，每装一台，修改一次。
    
3. 修改Vagrantfile文件

    3.1. 注释`config.vm.box_version = ">= 308.0.1"`
    
    3.2. 修改`config.vm.define vm_name = "core-%02d" % i do |config|`  为`config.vm.define vm_name = "core-01" do |config|`，其中core-01为要安装主机的主机名，每装一台，修改一次。
    
    3.3. 修改`ip = "172.17.8..#{i+100}"`为自己适合的网段（如果有必要的话）。

4.  下载[box文件][]并安装，`vagrant box add coreos-stable coreos_production_vagrant.box`。（参见第二种方式第四步）
5.  执行`vagrant up`，安装完毕后，使用putty登录，用户名为core，记得配置私钥。

当然，你也可以安装好一个CoreOS VM后，Clone下一个CoreOS VM，然后登陆并修改其相关配置即可。


## 小结

安装完毕后，可以进入某一台主机，使用`fleetctl list-machines`测试集群是否正常启动。本文涉及到比较多的知识点，比如：

1. cloud-config.yaml文件的作用及如何进行个性化配置
2. Static方式的工作原理
3. vagrant的工作机制

因为笔者现在也不是完全清楚，所以没有细写，欢迎大家交流。

## 引用

本文引用了其他博客或stack overflow上的成果，算是一个总结补充帖。但因为时间原因，一些已经忘记，部分引用如下：

[http://wiselyman.iteye.com/blog/2152167][]

[http://www.infoq.com/cn/articles/coreos-analyse-etcd][]

本文已授权在[dockerpool][]上发表，欢迎大家参与讨论！

[ISO]: https://coreos.com/docs/running-coreos/platforms/iso/
[http://codebeautify.org/yaml-validator]: http://codebeautify.org/yaml-validator
[Vagrant]: https://www.vagrantup.com/downloads.html
[box文件]: http://storage.core-os.net/coreos/amd64-usr/alpha/coreos_production_vagrant.box
[http://wiselyman.iteye.com/blog/2152167]: http://wiselyman.iteye.com/blog/2152167
[http://www.infoq.com/cn/articles/coreos-analyse-etcd]: http://www.infoq.com/cn/articles/coreos-analyse-etcd
[http://qiankunli.github.io/]: http://qiankunli.github.io/
[qiankun.li@qq.com]: qiankun.li@qq.com
[dockerpool]: http://www.dockerpool.com/