---

layout: post
title: 基于docker搭建测试环境
category: 技术
tags: Container
keywords: Docker shipyard jenkins

---

## 简介（已过时）

当web项目开发完毕后，一般会在测试环境上运行一下，供开发部门调错和测试部门测试。对于具有一定业务规模的公司，几十个上百个web服务，每个服务分别占用一个tomcat目录，配置过程繁琐，且无法集中管理。此外，对于公司的新手来讲，需要一定的背景知识才可以上手。

本文主要讲述基于docker搭建测试环境，或许可以解决部分上述问题。

### 总体思路

1. 业务代码（包括Dockerfile文件）通过git提交，使用Jenkins或hudson触发maven编译项目代码、并制作成docker镜像，push到docker镜像服务器。
2. 登录shipyard，deploy docker容器

### 业务流程

1. 在web项目目录中添加一个Dockerfile文件

        FROM tomcat
        ADD *.war $TOMCAT_HOME/webapps
        # 启动tomcat并监听tomcat日志
        CMD bash start.sh
    
2. 创建一个新的hudson job，并build。
3. 进入`http://shipyard:8080/`,通过web ui决定在哪台主机上运行项目实例，并配置映射端口。

##  基于docker测试环境的安装

jenkin与docker的整合参见:[使用Jenkins来构建Docker容器](http://www.cnblogs.com/Leo_wl/p/4314792.html "")，在此就不班门弄斧了。

该测试环境使用shipyard管理docker镜像和容器（运行web实例）。shipyard, Built on Docker Swarm, Shipyard gives you the ability to manage Docker resources including containers, images, private registries and more.

示例环境描述：在`192.168.56.154`,`192.168.56.155`上搭建docker swarm集群，并在`192.168.56.154`上运行shipyard controller。

## 安装docker registry

    docker run -d -p 5000:5000 -v /root/registry:/tmp/registry registry

### 安装docker swarm

1. 为`192.168.56.154`,`192.168.56.155`安装docker，并配置其`DOCKER_OPTS="--insecure-registry 私服ip:5000 -H 0.0.0.0:2375 -H unix:///var/run/docker.sock"`
2. 为`192.168.56.154`,`192.168.56.155`搭建zookeeper集群（也可以使用现成的zookeeper集群，其它配置工具etcd等也可）
3. 为`192.168.56.154`,`192.168.56.155`搭建docker swarm（zookeeper只是其中一种服务发现的方式）

    - root@192.168.56.155 # `docker run -ti -d --restart=always --name shipyard-swarm-agent swarm join zk://192.168.56.154,192.168.56.155/swarm --addr=192.168.56.155:2375`
        这容器工作就是：不停的向zookeeper注册该节点的信息，进入zookeeper命令行可以看到
        
        [zk: 192.168.56.154:2181(CONNECTED) 5] ls /swarm/docker/swarm/nodes
		[192.168.56.155:2375]
    - root@192.168.56.154 # `docker run -ti -d --restart=always --name shipyard-swarm-agent swarm join zk://192.168.56.154,192.168.56.155/swarm --addr=192.168.56.154:2375`
    - root@192.168.56.154 # `docker run -ti -d --restart=always --name shipyard-swarm-manager -p 2376:2376 swarm manage zk://192.168.56.154,192.168.56.155/swarm --host tcp://0.0.0.0:2376`

        ` --host tcp://0.0.0.0:2376`是设置容器中swarm的http server监听2376端口，`-p 2376:2376`是将容器的2376端口映射出来，**注意2376端口是随意弄的，但该端口不能命名为2375**。至此，docker swarm将以`192.168.56.154:2376`对外提供web服务
        
### shipyard 手动安装步骤

shipyard最新的是3.0.0版，基于docker swarm，其所有组件以docker容器方式运行，有两种部署方式

1. 自动部署，命令：`curl -sSL https://shipyard-project.com/deploy | bash -s`
2. 手动部署,手动依次启动必须的容器组件。

安装过程
    
1. 通过`/root/shipyard/data`持久化数据库中的数据 

        root@192.168.56.154 # `docker run -ti -d --restart=always --name shipyard-rethinkdb -v /root/shipyard/data:/data rethinkdb`
    
2. 安装shipyard-controller

        root@192.168.56.154 # docker run -ti -d --restart=always --name shipyard-controller --link shipyard-rethinkdb:rethinkdb --link shipyard-swarm-manager:swarm -p 8080:8080 shipyard/shipyard:latest server -d tcp://swarm:2376
   
## 需要注意的问题

### docker容器一定可以访问宿主机么

理论上是可以访问的，但如果你的宿主机打开了防火墙，对于`192.168.56.154`执行`docker run -ti -d --restart=always --name shipyard-swarm-agent swarm join zk://192.168.56.154,192.168.56.155/swarm --addr=192.168.56.154:2375`时，可能会失败，因为swarm容器无法访问`192.168.56.154`的2376端口

### 清掉过时的镜像和容器

对于测试环境，业务代码经常更新，因此会产生非常多的docker镜像和容器，需要在合适的实际将其干掉。这涉及到

1. 镜像的命名策略
2. 干掉old镜像以及对应container的时机

我采用以下策略：镜像名与jenkins的JOB_NAME相同，在使用jenkins build镜像时，便通过swarm/docker remote RESTFUL API干掉原有的镜像和容器。


## 优势

1. docker镜像集中管理（通过web ui进行管理）
2. docker容器（类似于一个项目实例）集中管理，并可以监控所有实例的运行状态，还可以创建、删除“运行实例”
3. 减少操作步骤，只需要极少的背景知识（为调试项目，需要懂一点docker命令）。

## 不足

1. 运行的web项目只可以调用其它服务（包括redis、rabbitmq等），不能对外提供服务（不是不可以做，而是复杂的端口映射不好管理（可以使用nginx可以解决））。
