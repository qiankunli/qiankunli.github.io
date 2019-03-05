---

layout: post
title: mesos 集成 calico
category: 技术
tags: Mesos
keywords: Docker, calico

---


## 简介

本文讲述基于物理机安装 mesos 及 calico，流水账，原理思路部分的内容参见 [calico](http://qiankunli.github.io/2018/02/04/calico.html)


环境描述，三台virtualbox ubuntu 16.04 虚拟机

|hostname|ip|作用|
|---|---|---|
|ubuntu-1|192.168.56.101|etcd,docker,mesos master,marathon|
|ubuntu-2|192.168.56.102|etcd,docker,mesos slave|
|ubuntu-3|192.168.56.103|etcd,docker,mesos slave|

calico 与 mesos 集成有两种方式

1. mesos 通过 cni 接口操作 calico。从当下的材料[Integration Guide](https://docs.projectcalico.org/v2.6/getting-started/mesos/installation/integration)看，cni方式多针对 Unified Containerizer，当然docker 也算  Unified Containerizer。 
2. docker 与calico 直接整合，`docker network create calico_network`，marathon 创建项目时指定 calico_network

calico 与 mesos 的整合到 2.6 版本为止，恢复时间未知。

使用cni的一大优点：将网络策略控制提升到编排工具层面。所以，是否使用cni方式

1. 是否有需要。事实上，即便是第二种集成方式，也可以通过calicoctl 控制网络策略
2. 到我本文实践的mesos 1.5，其网络策略设置 不如 k8s 

因此，本文采用第二种集成方式

## 基础组件

1. etcd

	* calico 本身，及calico docker 沟通数据需要
	* 内部测试，无需开启tls
2. zk

	* mesos
	* 可以简单安装`apt-get install zookeeperd`
3. docker

	* Configured with Cluster Store `dockerd --cluster-store=etcd://$ETCD_IP:$ETCD_PORT` docker 支持`/etc/docker/daemon.json` 的高版本可以`{"cluster-store": "etcd://100.108.150.45:2379"}`

## 安装calico

[Integration Guide](https://docs.projectcalico.org/v2.6/getting-started/mesos/installation/integration)



### 下载calicoctl 文件

	sudo wget -O /usr/local/bin/calicoctl https://github.com/projectcalico/calicoctl/releases/download/v1.6.3/calicoctl
	sudo chmod +x /usr/local/bin/calicoctl
	
calicoctl, The calicoctl command line interface provides a number of resource management commands to allow you to create, modify, delete, and view the different Calico resources.

从calico 网络的搭建（下载容器并运行），到calico 网络配置，设置为一个net=none 的容器配置 network namespace，都可以通过 calicoctl 来操作

### 配置calicoctl 

By default calicoctl looks for a configuration file at /etc/calico/calicoctl.cfg.

	apiVersion: v1
	kind: calicoApiConfig
	metadata:
	spec:
	  datastoreType: "etcdv2"
	  etcdEndpoints: "http://192.168.56.101:2379,http://192.168.56.102:2379,192.168.56.103:2379"
	  

### Launch calico/node

以容器的方式提供 calico 服务

直接以命令的方式运行容器

	ETCD_ENDPOINTS=http://$ETCD_IP:$ETCD_PORT calicoctl node run --node-image=quay.io/calico/node:v2.6.8
	
	
systemd service 文件的方式

1. environment file

		ETCD_AUTHORITY=localhost:2379
		ETCD_SCHEME=http
		ETCD_CA_FILE=""
		ETCD_CERT_FILE=""
		ETCD_KEY_FILE=""
		CALICO_HOSTNAME=""
		CALICO_NO_DEFAULT_POOLS=""
		CALICO_IP="192.168.56.101"
		CALICO_IP6=""
		CALICO_AS=""
		CALICO_LIBNETWORK_ENABLED=true
		CALICO_NETWORKING_BACKEND=bird
		# IP_AUTODETECTION_METHOD
	
2. systemd service file 

		[Unit]
		Description=calico-node
		After=docker.service
		Requires=docker.service
		
		[Service]
		EnvironmentFile=/etc/calico/calico.env
		ExecStartPre=-/usr/bin/docker rm -f calico-node
		ExecStart=/usr/bin/docker run --net=host --privileged \
		 --name=calico-node \
		 -e HOSTNAME=${HOSTNAME} \
		 -e IP=${CALICO_IP} \
		 -e IP6=${CALICO_IP6} \
		 -e CALICO_NETWORKING_BACKEND=${CALICO_NETWORKING_BACKEND} \
		 -e AS=${CALICO_AS} \
		 -e NO_DEFAULT_POOLS=${CALICO_NO_DEFAULT_POOLS} \
		 -e CALICO_LIBNETWORK_ENABLED=${CALICO_LIBNETWORK_ENABLED} \
		 -e ETCD_AUTHORITY=${ETCD_AUTHORITY} \
		 -e ETCD_SCHEME=${ETCD_SCHEME} \
		 -e ETCD_CA_CERT_FILE=${ETCD_CA_CERT_FILE} \
		 -e ETCD_CERT_FILE=${ETCD_CERT_FILE} \
		 -e ETCD_KEY_FILE=${ETCD_KEY_FILE} \
		 -e IP_AUTODETECTION_METHOD=${IP_AUTODETECTION_METHOD} \
		 -v /var/log/calico:/var/log/calico \
		 -v /run/docker/plugins:/run/docker/plugins \
		 -v /lib/modules:/lib/modules \
		 -v /var/run/calico:/var/run/calico \
		 -v /var/run/docker.sock:/var/run/docker.sock \
		 -v /run:/run \
		 quay.io/calico/node:v2.6.2
		
		ExecStop=-/usr/bin/docker stop calico-node
		
		Restart=on-failure
		StartLimitBurst=3
		StartLimitInterval=60s
		
		[Install]
		WantedBy=multi-user.target
	
3. 启动 quay.io/calico/node, `systemctl staart calico-node`

quay.io/calico/node:v2.6.8 运行时，比较重要的参数/环境变量主要有：

1. IP ,The IPv4 address to assign this host 
2. IP_AUTODETECTION_METHOD, The method to use to autodetect the IPv4 address for this host.
3. IP 和 IP_AUTODETECTION_METHOD 是互斥关系，若IP 不设置，就会按照一定的策略detect ip。detect 时，运行virtural box 环境的 主机网卡名比较不常见，需要专门指定。综合来看，还是设置IP 好一些。

	
calico/node 启动时会创建两个默认的ipPool

	calicoctl get ipPool
	CIDR
	192.168.0.0/16
	fd80:24e2:f998:72d6::/64
	
ip pool 中有一个ipip选项，若网络环境支持bgp，可以关闭。

### docker 创建网络

	docker network create --driver calico --ipam-driver calico-ipam  --subnet=192.168.0.0/16 calico-test

## mesos

### 安装mesos

二进制方式 [CONFIGURING A MESOS/MARATHON CLUSTER ON UBUNTU 16.04](http://www.admintome.com/blog/configuring-a-dcos-cluster-on-ubuntu-16-04/)

apt 方式

1. `apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF`
2. 添加源

		cat <<EOF >> /etc/apt/sources.list.d/mesosphere.list
		deb http://repos.mesosphere.com/ubuntu xenial main
		EOF
3. `apt-get -y install mesos`

### 配置mesos

参见 [Mesos 安装与使用](https://yeasy.gitbooks.io/docker_practice/content/mesos/installation.html#%E8%BD%AF%E4%BB%B6%E6%BA%90%E5%AE%89%E8%A3%85)

1. 配置 zk

		cat <<EOF >> /etc/mesos/zk
		zk://`hostname`:2181/mesos
		EOF

1. 配置mesos-master，建议在 /etc/mesos-master 目录下，配置至少四个参数文件：ip、quorum、work_dir、cluster。

		
		cat <<EOF >> /etc/mesos-master/quorum
		1
		EOF

	
2. 配置mesos-slave

		cat <<EOF >> /etc/mesos-slave/containerizers
		docker,mesos
		EOF
		
		cat <<EOF >> /etc/mesos-slave/ip
		$slave_ip
		EOF
		
## marathon

[Install Marathon](http://192.168.60.8:8080/ui/#/apps/%2Fdeploy-to-docker-openapi-smart-device-auth-test)

### 安装marathon

	apt-get install marathon
	
### 配置marathon

marathon 配置参数的方式与mesos 基本相同

对于一些系统来说，配置目录为 `/etc/marathon/conf`（需要手动创建），此外默认配置文件在 `/etc/default/marathon`。

	mkdir -p /etc/marathon/conf
	cp /etc/mesos/zk /etc/marathon/conf/master
	
创建 `/etc/marathon/conf/zk` 文件，内容为

	zk://192.168.56.101:2181,192.168.56.102:2181,192.168.56.103:2181/marathon

对于marathon 来说，有一些参数是必配的[Marathon Command-Line Flags](https://mesosphere.github.io/marathon/docs/command-line-flags.html)

注意：**marathon 每次升级都会改动一些东西，未经过完整测试，请不要轻易升级**。

## 启动容器

[Ucloud云上环境使用calico+libnetwork连通容器网络实践](https://zhuanlan.zhihu.com/p/24094454)

marathon.json 

	{
	  "id": "my-docker-task",
	  "cpus": 0.1,
	  "mem": 64.0,
	  "container": {
	      "type": "DOCKER",
	      "docker": {
	          "network": "USER",
	          "image": "nginx"
	      }
	  },
	  "ipAddress": {
	      "networkName": "calico-test"
	  }
	}

## 一些问题

默认情况下，使用calico网络创建的容器的mac 地址是`ee:ee:ee:ee:ee:ee`，但部分环境下，该设置失效，mac 地址任意。 容器与外界通信时，所在host接收到数据后， 以ee:ee:ee:ee:ee:ee 向容器发送数据包，造成网络不通的现象。[
Cannot communicate with the gateway #898](https://github.com/projectcalico/calicoctl/issues/898) 

办法之一是，以Privilege模式运行容器，将cali0 mac 设置一下，`ifconfig cali0 hw ether ee:ee:ee:ee:ee:ee`