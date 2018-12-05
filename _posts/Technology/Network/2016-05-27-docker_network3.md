---

layout: post
title: Docker网络三,基于OVS实现Docker跨主机网络
category: 技术
tags: Network
keywords: Docker,OVS

---

## 前言(就目前docker network发展看，已有些过时)

docker libnetwork已经有跨主机容器互通的方案了，那么为什么还要介绍OVS呢？因为大部分公司的线上环境都是CentOS6，其内核版本不能支持较为高级的docker特性，因此还是要基于OVS搞一套跨主机容器互通的方案。

第一次将博客写成了各个博文的摘抄与大杂烩，作为一个java开发工程师，网络的知识实在是有限。

## docker使用OVS网桥

这部分内容可以参见[How to use OpenVswitch with Docker][]，写的言简意赅，强烈建议细读。文中已经指出，pipework简化了“容器网卡的创建”以及“容器网卡与host网桥的关联“操作（这两个操作一步完成）。

pipework不仅可以操作linux网桥，还可以操作ovs网桥（我看更像个虚拟交换机）在单机环境和多机环境下为docker网络划分vlan。参见[Docker网络详解及pipework源码解读与实践][]，多机环境下，物理主机eth0网卡需要设置为混杂模式（是指一台机器的网卡能够接收所有经过它的数据流，而不论其目的地址是否是它），连接主机的交换机端口应设置为trunk模式，即允许不同VLAN的包通过。当然，如果物理主机的连通不想做复杂配置，可以使用下文所述的方案。

类似的文档还可以参见[Using OVS bridge for docker networking][]，文中提到了docker使用ovs网桥的两种模式：

1. NAT,可以完全替换默认docker0网桥的“容器互通”和“容器与外网之间NAT”的功能
2. Bridge，可以将容器加入到物理网段中

## 跨主机docker容器互通

从目前看，基于ovs的跨主机容器通信有两种模式：

1. GRE，参见[Linux下Bridge和ovs Bridge、gre以及docker的混合应用][]
2. vxlan，参见[Docker+OpenvSwitch搭建VxLAN实验环境][]或[docker高级应用之多台主机网络互联][]

基本思路是：

1. 针对每个主机，使用ovs创建一个ovs-br（ovs网桥），将docker容器的网卡桥接在这个ovs-br上。此时，相当于ovs-br替代了原先docker0网桥的作用，实现主机内容器的互通以及对外网的访问。
2. 针对每个主机，为ovs-br创建一个ovs port和ovs interface（type为gre或vxlan），并设置gre和vxlan的remote_ip为其它物理主机。实现ovs-br的跨主机连通，其相关的容器自然也连通了。

        // GRE类型
        ovs-vsctl add-port ovs-br0 gre0 -- set interface gre0 type=gre options:remote_ip=$REMOTE_IP  
        // vxlan类型
        ovs-vsctl add-port ovs-br0 vx0 -- set interface vx0 type=vxlan options:remote_ip=$REMOTE_IP  


 ![Alt text](/public/upload/docker/docker_ovs_gre.png)

 ![Alt text](/public/upload/docker/docker_ovs_gre2.jpg)


vxlan方式的一个优势是：如果将gre或vxlan比作“网线”的话，对于两台以上主机，比如hostA、hostB和hostC，host之上的container互通只要两根“网线”就行。假设hostA连着hostB，hostB连着hostC，那么hostC上的container自然可以通过hostB找到hostA上的container。而对于GRE方式，则三台主机必须两两连接，此时为集群中添加一台主机则非常麻烦。

## 容器ip自动分配的问题

从[Docker+OpenvSwitch搭建VxLAN实验环境][]中我们可以知道，我们只需为容器准备“网卡”veth，并将其peer veth挂到ovs-br上即可（笼统的说，就是将容器挂到ovs-br上）。容器的ip可以由容器自己去dhcp服务器上获取（dhcp服务器要自己创建），参见[How to use OpenVswitch with Docker][]

## 在多主机状态下使用ovs网桥

在上述方案中，完全用ovs-br替掉了docker默认的docker0网桥。此时，如果容器还要与host进行端口映射，则要在ovs-br进行iptables配置，参见[Using OVS bridge for docker networking][]。一种取巧的方案是，docker还是使用默认的docker0网桥，将ovs-br挂在docker0上，这样既可实现跨主机容器通信，也可以实现容器与主机之间NAT，这样非容器网络中的主机就可以通过`IP:port`方式访问容器中的服务。参见[Docker系列(五)OVS+Docker网络打通示例][]

无论使用哪种方式，在多主机环境下，一个必须要注意的问题是：要限定每个主机的容器的ip可分配范围，以防止不同host出现同一个ip的container。想必这是在[Docker系列(五)OVS+Docker网络打通示例][]中重建docker0网桥的原因。

## 搭建一个多主机docker网络

通过以上部分，一个简单的多主机docker网络方案就呼之欲出了（先在hostA上搭建，然后hostB加入，网络以这种形式逐步扩张）：

1. hosta上创建一个ovs网桥
2. 用docker swarm创建一个容器，网络模式为none
3. 将该容器挂到到host（可以根据docker swarm获取到容器被部署在了哪台host上）的ovs-br上（参见第一小节）
4. 容器的启动脚本中包含“向dhcp服务器请求ip”的逻辑并设置自己的ip
5. 新加入hostb
6. hostb创建与hosta一致的ovs网桥
7. 如果有新的容器启动，则重复第2步到第4步

如果要在跨主机docker网络中划分多个vlan，则对于每个vlan，需要在所有主机上创建对应的网桥并连通，并将相应的网络地址配置到dhcp服务器中。（待确认）

上述逻辑是否可以在docker plugin中实现？

整个逻辑实现下来还是蛮复杂的，如果无法升级linux内核进而使用docker新版本特性，还是用kubernetes方便很多。

## 待续

还是对linux基本的网络命令不熟悉，要加强学习的有

1. iproute2命令
2. pipework命令（openvswtich 官方工具包中有一个ovs-docker，不知道相似否）
3. ovs命令

    
## 参考文献


[利用OpenVSwitch构建多主机Docker网络][]

[Linux下Bridge和ovs Bridge、gre以及docker的混合应用][]

[docker高级应用之多台主机网络互联][]

[Docker+OpenvSwitch搭建VxLAN实验环境][]

[Docker网络详解及pipework源码解读与实践][]


[利用OpenVSwitch构建多主机Docker网络]: http://dockone.io/article/228
[Linux下Bridge和ovs Bridge、gre以及docker的混合应用]: http://www.rendoumi.com/linuxxia-bridgehe-ovsyi-ji-dockerde-hun-he-ying-yong/
[docker高级应用之多台主机网络互联]: http://dl528888.blog.51cto.com/2382721/1611491
[Docker+OpenvSwitch搭建VxLAN实验环境]: http://www.cnblogs.com/yuuyuu/p/5180827.html#commentform
[How to use OpenVswitch with Docker]: http://cloudgeekz.com/400/how-to-use-openvswitch-with-docker.html
[Docker网络详解及pipework源码解读与实践]: http://www.infoq.com/cn/articles/docker-network-and-pipework-open-source-explanation-practice
[Docker系列(五)OVS+Docker网络打通示例]: http://www.cnblogs.com/jianyuan/p/5007517.html
[Using OVS bridge for docker networking]: https://developer.ibm.com/recipes/tutorials/using-ovs-bridge-for-docker-networking/