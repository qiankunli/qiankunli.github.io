---

layout: post
title: Docker网络二,libnetwork
category: 技术
tags: Network
keywords: Docker,libnetwork

---

## 前言

我们搭建一个网络环境，一般遵循一定的网络拓扑结构。由于Linux可以模拟相应的网络设备，并可以创建“虚拟机”（也就是容器），因此在Linux系统内部，我们也可以遵循一定的网路拓扑结构，设计一个“内网”，实现容器之间的通信。

本文主要讲述容器跨主机网络通信。

## 容器跨主机的网络通信涉及的一些问题

1. IPAM：IP地址管理，传统的网络比如说DHCP其实就是一种IPAM。到了容器时代我们谈IPAM，主流的两种方法：基于CIDR的IP地址段分配（每个主机提前设定好一个ip域，其上的所有容器从这个ip域上申请ip）或者精确为每一个容器分配IP。docker最初就是基于单机的IPAM，那么在跨主机时代，就需要有一个集中的数据库，记住ip的分配情况。

2. 容器的跨主机通信。容器发出去的包如何转换成主机实际发出去的包

容器跨主机的网络通信，主要实现思路有两种：二层VLAN网络和Overlay网络。也有种说法是路由技术和隧道技术。

二层VLAN网络的思路是把原先的网络架构改造为互通的大二层网络，**通过特定网络设备直接路由**，实现容器点到点的之间通信。

Overlay网络是指在不改变现有网络基础设施的前提下，**通过约定通信协议**，把二层报文封装在IP报文之上的新的数据格式（**它有个专有名词叫隧道技术**）。

简略的说，前者靠硬件支持，容器通信的数据包跟正常主机通信的数据包变化不大。后者数据包变化较大，通常要封包和解包（这就需要运行一个daemon来干这个活儿），网络模型核心的实现点在主机内部。

## libnetwork

Libnetwork是Docker团队将Docker的网络功能从Docker核心代码中分离出去，形成一个单独的库。 Libnetwork通过**插件的形式**为Docker提供网络功能。 使得用户可以根据自己的需求实现自己的Driver来提供不同的网络功能。 

官方目前计划实现以下Driver：

1. Bridge ： 这个Driver就是Docker现有网络Bridge模式的实现。 （基本完成，主要从之前的Docker网络代码中迁移过来）
2. Null ： Driver的空实现，类似于Docker 容器的None模式。
3. Overlay ： 隧道模式实现多主机通信的方案。 

“Libnetwork所要实现的网络模型（网络拓扑结构）基本是这样的： 

1. 用户可以创建一个或多个网络（一个网络就是一个网桥或者一个VLAN ）
2. 一个容器可以加入一个或多个网络。 
3. 同一个网络中容器可以通信，不同网络中的容器隔离。

”**我觉得这才是将网络从docker分离出去的真正含义，即在创建容器之前，我们可以先创建网络（即创建容器与创建网络是分开的），然后决定让容器加入哪个网络。**

## Libnetwork网络使用方式

### 直接使用

1. 假设存在主机`192.168.56.101`,`192.168.56.102`
2. 修改每个主机的docker启动参数`DOCKER_OPTS=--insecure-registry 192.168.3.56:5000 -H 0.0.0.0:2375 --cluster-store=etcd://192.168.56.101:2379/ --cluster-advertise=192.168.56.101:2375`，重启docker。
3. docker创建overlay网络net1和net2
    
    - `192.168.56.101`或`192.168.56.102`执行`docker network create -d overlay net1``docker network create -d overlay net2`
    - `192.168.56.101`运行容器net1c1,net2c1`docker run -itd --name net1c1 --net net1 ubuntu:14.04`
    - `192.168.56.102`运行容器net1c2,net2c2。

### 通过docker swarm使用

1. 启动etcd集群，存储docker swarm节点信息

    `192.168.56.101`上etcd配置

        ETCD_OPTS=--data-dir=/var/lib/etcd/ \
                  --name wily1 \
                  --initial-advertise-peer-urls http://192.168.56.101:2380 \
                  --listen-peer-urls http://192.168.56.101:2380 \
                  --listen-client-urls http://192.168.56.101:2379,http://127.0.0.1:2379 \
                  --advertise-client-urls http://192.168.56.101:2379 \
                  --initial-cluster-token etcd-cluster-1 \
                  --initial-cluster-state new \
                  --initial-cluster wily1=http://192.168.56.101:2380,wily2=http://192.168.56.102:2380
          
    `192.168.56.102`上etcd配置

        ETCD_OPTS=--data-dir=/var/lib/etcd/ \
                  --name wily1 \
                  --initial-advertise-peer-urls http://192.168.56.102:2380 \
                  --listen-peer-urls http://192.168.56.102:2380 \
                  --listen-client-urls http://192.168.56.102:2379,http://127.0.0.1:2379 \
                  --advertise-client-urls http://192.168.56.102:2379 \
                  --initial-cluster-token etcd-cluster-1 \
                  --initial-cluster-state new \
                  --initial-cluster wily1=http://192.168.56.101:2380,wily2=http://192.168.56.102:2380


2. 启动 docker swarm

    - `192.168.56.101`执行`docker run --name swarm-agent -d swarm join --addr=192.168.56.101:2375 etcd://192.168.56.101:2379/swarm`
    - `192.168.56.102`执行`docker run --name swarm-agent -d swarm join --addr=192.168.56.102:2375 etcd://192.168.56.102:2379/swarm`
    - `192.168.56.101`上启动swarm-manager`docker run --name swarm-manager -d -p 3375:2375 swarm manage etcd://192.168.56.101:2379/swarm`

3. `192.168.56.101`上创建网络net3并启动容器

    `docker -H tcp://localhost:3375 network create -d overlay net3`

    `docker -H tcp://localhost:3375 run -it --net net3 ubuntu bash`

4. `192.168.56.101`上创建网络net4并启动容器，并且指定容器的ip，**这个效果在实际场景中很有用**

    `docker -H tcp://localhost:3375 network create -d overlay net4 --subnet 172.19.0.0/16`

    `docker -H tcp://localhost:3375 run -it --net net4 --ip=172.19.0.6  ubuntu bash`

    **使用`--ip`参数时，必须指定特定的子网**，参见`https://github.com/docker/docker/issues/20547`

5. 用`docker-compose`试试。docker-compose文件

        version: '2'
        services:
          web:
            image: reinblau/php-apache2
            networks: 
              - net5
        
        networks: 
          net5:
            driver: overlay


## 使用systemd配置docker swarm

1. `192.168.56.101`和`192.168.56.102`配置swarm-agent.service

        [Unit]
        Description=Docker Swarm Agent Container
        Documentation=https://docs.docker.com
        Requires=etcd.service
        Requires=docker.service
        After=etcd.service
        After=docker.service
        
        [Service]
        Type=simple
        EnvironmentFile=/etc/default/swarm-agent
        ExecStartPre=-/usr/bin/docker stop ${CONTAINER_NAME}
        ExecStartPre=-/bin/echo "/usr/bin/docker stop ${CONTAINER_NAME}"
        ExecStartPre=-/usr/bin/docker rm ${CONTAINER_NAME}
        ExecStartPre=/usr/bin/docker pull swarm
        ExecStart=/usr/bin/docker run --rm --name ${CONTAINER_NAME} swarm join --addr=${DOCKER_ADDRESS} ${ETCD_SWARM_URL}
        ExecStartPre=-/bin/echo "/usr/bin/docker run --rm --name ${CONTAINER_NAME} swarm join --addr=${DOCKER_ADDRESS} ${ETCD_SWARM_URL}"
        
        [Install]
        WantedBy=multi-user.target
        
2. `192.168.56.101`配置swarm-manager.service

        [Unit]
        Description=Docker Swarm Manager Container
        Documentation=https://docs.docker.com
        Requires=etcd.service
        Requires=docker.service
        After=etcd.service
        After=docker.service
        
        [Service]
        Type=simple
        EnvironmentFile=/etc/default/swarm-manager
        ExecStartPre=-/usr/bin/docker stop ${CONTAINER_NAME}
        ExecStartPre=-/bin/echo "/usr/bin/docker stop ${CONTAINER_NAME}"
        ExecStartPre=-/usr/bin/docker rm ${CONTAINER_NAME}
        ExecStartPre=/usr/bin/docker pull swarm
        ExecStart=/usr/bin/docker run --name ${CONTAINER_NAME} -p ${SWARM_PORT}:2375 swarm manage ${ETCD_SWARM_URL}
        ExecStartPre=-/bin/echo "/usr/bin/docker run --rm --name ${CONTAINER_NAME} -p ${SWARM_PORT}:2375 swarm manage ${ETCD_SWARM_URL}"
        
        [Install]
        WantedBy=multi-user.target

这样就可以使用systemctl来控制docker swarm的启动和关闭了。

## 一些坑

如果你使用virtual box虚拟了两个主机`192.168.56.101`和`192.168.56.102`，并且`192.168.56.102`是由`192.168.56.101`克隆而来，则你需要清除`xx/docker/key.json`（不同系统位置不同），并重启docker。否则两个主机启动的容器可能具有同一个id，进而导致使用docker swarm时出现问题。参见`https://github.com/docker/swarm/issues/380`

## 使用pipework设置容器的ip

在overlay（较高的docker版本）网络下，可以通过参数设置容器的ip。较低的docker版本，可以通过pipework设置容器ip。

1. 安装pipework

        $ git clone https://github.com/jpetazzo/pipework.git
        $ sudo cp -rp pipework/pipework /usr/local/bin/
    
2. 升级iproute（对于centos）

        $ yum install -y http://rdo.fedorapeople.org/rdo-release.rpm
        $ vim /etc/yum.repo.d/rdo-release.repo
        # 将baseurl更新为https://repos.fedorapeople.org/repos/openstack/EOL/openstack-icehouse/epel-6/
        # 更改gpgcheck为0
        $ yum update iproute
        
3. 设置容器的ip

        pipework bridge_name containerid ip/24

## 小结

docker 真是做的越来越全面了，如果仅仅是用用，一切都是参数配置。

    
## 参考文献

[聊聊Docker 1.9的新网络特性][]

[理解Docker跨多主机容器网络][]

[聊聊Docker 1.9的新网络特性]: http://mt.sohu.com/20160118/n434895088.shtml
[理解Docker跨多主机容器网络]: http://www.07net01.com/2016/02/1275302.html