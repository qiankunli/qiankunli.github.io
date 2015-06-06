---

layout: post
title: docker中的一些点
category: 技术
tags: Docker
keywords: Docker

---

## 前言

## docker如何实现与外界的相互访问

docker容器和外界的交互，主要包含以下情况：

1. 同一主机的容器之间
2. 主机与宿主机之间
3. 主机与宿主机所在网络的其它主机之前
4. 主机与宿主机所在网络的其它主机的容器之间

下面主要介绍第三种情况的实现原理：NAT（通过添加iptables规则实现网络地址转换）


    root@ubuntu1:~# docker run -d -P 4b0eb499efbd
    
    root@ubuntu1:~# docker ps
    CONTAINER ID        IMAGE                             COMMAND                CREATED             STATUS              PORTS                     NAMES
    0afe24ab86b8        docker-registry.sh/myapp:v0.0.1   "/bin/sh -c 'service   2 seconds ago       Up 1 seconds        0.0.0.0:49153->8080/tcp   stupefied_lovelace
    
    root@ubuntu1:~# docker inspect 0afe24ab86b8 | grep IP
        "IPAddress": "172.17.0.2",

    root@ubuntu1:~# sudo iptables -nvL -t nat
    
    Chain POSTROUTING (policy ACCEPT 3 packets, 432 bytes)
     pkts bytes target     prot opt in     out     source               destination
    0     0 MASQUERADE  all  --  *      !docker0  172.17.0.0/16        0.0.0.0/0

    Chain DOCKER (2 references)
     pkts bytes target     prot opt in     out     source               destination
        2   104 DNAT       tcp  --  !docker0 *       0.0.0.0/0            0.0.0.0/0            tcp dpt:49153 to:172.17.0.2:8080

Chain POSTROUTING规则会将源地址为172.17.0.0/16的包（也就是从Docker容器产生的包），并且不是从docker0网卡发出的，进行源地址转换，转换成主机网卡的地址。
Chain DOCKER规则就是对主机eth0收到的目的端口为80的tcp流量进行DNAT转换，将流量发往172.17.0.5:80，也就是我们上面创建的Docker容器。

a “routing process” must be running in the global network namespace to receive traffic from the physical interface, and route it through the appropriate virtual interfaces to to the correct child network namespaces. 

这样就实现了容器与外界的相互访问。

在docker的网络世界里，与其说docker“容器”是如何与外界（容器，物理机，网络上的其它主机）交流的，不如说linux的 内部的network namespace是如何与外界（其它network namespace，根network namespace）交流的。

## docker volume


    // 创建一个容器，包含两个数据卷
    $ docker run -v /var/volume1 -v /var/volume2 -name Volume_Container ubuntu14.04 linux_command
    // 创建App_Container容器，挂载Volume_Container容器中的数据卷
    $ docker run -t -i -rm -volumes-from Volume_Container -name App_Container ubuntu14.04  linux_command
    // 这样两个容器就可以共用这个数据卷了
    
    // 最后可以专门安排一个容器，在应用结束时，将数据卷中的内容备份到主机上
    docker run -rm --volumes-from DATA -v $(pwd):/backup busybox tar cvf /backup/backup.tar /data
    
在默认方式下，volume就是在`/var/lib/docker/volumes`目录下创建一个文件夹，并将该文件夹挂载到容器的某个目录下（以UFS文件系统的方式挂载）。当然，我们也可以指定将主机的某个特定目录（该目录要显式指定）挂载到容器的目录中。
    
    

## 参考文献

[Docker 网络配置][]


[Docker 网络配置]: http://www.oschina.net/translate/docker-network-configuration