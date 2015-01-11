
---
layout: post
title: 如何通过fleet unit files 来构建灵活的服务
category: 技术
tags: CoreOS
keywords: CoreOS Docker
---
## 简介

原文地址：[How to Create Flexible Services for a CoreOS Cluster with Fleet Unit Files][]，有删减。

CoreOS利用一系列的工具，大大简化了集群和Docker服务的管理。其中，Etcd将独立的节点连接起来，并提供一个存储全局数据的地方。大部分实际的服务管理和管理员任务则有fleet来负责。

在[上一个guide][]中，我们过了一遍fleetctl的基本使用：操纵服务和集群中的节点。在那个guide中，我们简单的介绍了fleet unit文件，fleet使用它来定义服务。还创建一个简单的unit 文件，展示了如何使用fleetctl提供一个工作的service。

在本guide中，我们深度学习fleet unit文件，了解一些使你的service更加健壮的技巧。

## 准备

为了完成本教程，我们假设你已经根据[我们以往的教程][]创建了一个CoreOS集群，假定集群包含以下主机：

- coreos-1
- coreos-2
- coreos-3

尽管本教程的大部分内容是关于unit文件的创建，但在描述unit文件中指令（属性）的作用时会用到这些机器。我们同样假设你阅读了[如何使用fleetctl][]，你已经可以使用fleetctl将unit文件部署到集群中。

当你已经做好这些准备后，请继续阅读本教程。

## unit文件的section和类型

因为fleet的服务管理方面主要依赖于集群节点本地的systemd，所以systemd unit file基本就是 fleet unit file。

fleet unit file 有许多类型，service是最常用的一种。这里列出一些 systemd unit file支持的类型，每个类型（文件）以"文件名.类型名"标识（命名）。比如example.service

- service
- socket
- device
- mount
- automount
- timer
- path

尽管这些类型都是有效的，但service类型用的最多。在本guide中，我们也只讨论service类型的配置。

unit 文件是简单的文本文件，以"文件名.类型名（上面提到的）"命名。在文件内部，它们以section 组织，对于fleet，大部分unit 文件是以下格式：

    [Unit]
    generic_unit_directive_1
    generic_unit_directive_2

    [Service]
    service_specific_directive_1
    service_specific_directive_2
    service_specific_directive_3
    
    [X-Fleet]
    fleet_specific_directive
    

**每个小节的头和其他部分都是大小写敏感的**。unit小节用来定义一个unit文件的通用信息，在unit小节定义的信息对所有类型（比如service）都通用。

Service小节用来设置一些只有Service类型文件才用到的指令，上面提到的每一个类型（但不是全部）（比如service和socket）都有相应的节来定义本类型的特定信息。

fleet根据X-Fleet小节来判定如何调度unit文件。在该小节中，你可以设定一些条件来将unit文件部署到某台主机上。

## Building the Main Service

在本节，我们的unit文件将和 [basic guide on running services on CoreOS][]中提到的有所不同。这个文件叫apache.1.service，具体内容如下：

    [Unit]
    Description=Apache web server service
    
    # Requirements
    Requires=etcd.service
    Requires=docker.service
    Requires=apache-discovery.1.service
    
    # Dependency ordering
    After=etcd.service
    After=docker.service
    Before=apache-discovery.1.service
    
    [Service]
    # Let processes take awhile to start up (for first run Docker containers)
    TimeoutStartSec=0
    
    # Change killmode from "control-group" to "none" to let Docker remove
    # work correctly.
    KillMode=none
    
    # Get CoreOS environmental variables
    EnvironmentFile=/etc/environment
    
    # Pre-start and Start
    ## Directives with "=-" are allowed to fail without consequence
    ExecStartPre=-/usr/bin/docker kill apache
    ExecStartPre=-/usr/bin/docker rm apache
    ExecStartPre=/usr/bin/docker pull username/apache
    ExecStart=/usr/bin/docker run --name apache -p ${COREOS_PUBLIC_IPV4}:80:80 \
    username/apache /usr/sbin/apache2ctl -D FOREGROUND
    
    # Stop
    ExecStop=/usr/bin/docker stop apache
    
    [X-Fleet]
    # Don't schedule on the same machine as other Apache instances
    X-Conflicts=apache.*.service
    
我们先说unit section，在本seciton，描述了unit 的基本信息和依赖情况。服务启动前有一系列的requirements，并且本例中使用的是hard requirements。如果我们想让fleet在启动本服务时启动其它服务，并且如果其他服务启动失败不影响本服务的启动，我们可以使用Wants来代替requirements。

接下来，我们明确的列出了requirements 启动的顺序。在服务运行前，确定其依赖的服务是可用的非常重要。我们也是通过这种方式来启动一个该服务的所有的从服务（Sidekick Service，后文会提到）。

在service 小节中，我们关闭了服务启动间隔时间。因为当服务开始在一个节点上运行时，相应镜像需要先从doker registry中pull下来，pull的时间会被计入startup timeout。这个值默认是90秒，通常是足够的。但对于比较复杂的容器来说，可能需要更长时间。

我们把killmode 设置为none，这是因为正常的kill 模式有时会导致容器移除命令执行失败（尤其是执行`docker --rm containerID/containername`时），这会导致下次启动时出问题。

我们把environment 文件也“拉进水”，这样我们就可以访问COREOS_PUBLIC_IPV4。如果服务创建时私有网络可用，`COREOS_PRIVATE_IPV4`是可配的，我们便可以使用每个主机自己的信息配置其container。

ExecStartPre 用来清除上次运行的残留，确保本次的执行环境是干净的。对于头两个ExecStartPre 使用“=-”，来告诉systemd：即使这两个命令执行失败了，也继续执行接下来的命令。这样，docker 将尝试kill 并且 移除先前的容器，没有发现容器也没有关系。最后一个ExecStartPre用来确保将要运行的container是最新的。

X-fleet section包含一个简单的条件，强制fleet将service调度某个没有运行Apache 服务的机器上。这样可以很容易的提高服务的可靠性，因为它们运行在不同的机器上（一个机器挂了，还有另外的机器在运行该服务）。


## Building the Sidekick Announce Service

现在当我们构建主unit文件时候，已经有了一些不错的想法。接下来，我们看一下传统的从service。这个从 service和主 serivce有一定联系，被用来向etcd注册服务。

这个文件，就像它在主unit文件中被引用的那样，叫做`apache-discovery.1.service`，内容如下：

    [Unit]
    Description=Apache web server etcd registration
    
    # Requirements
    Requires=etcd.service
    Requires=apache.1.service
    
    # Dependency ordering and binding
    After=etcd.service
    After=apache.1.service
    BindsTo=apache.1.service
    
    [Service]
    
    # Get CoreOS environmental variables
    EnvironmentFile=/etc/environment
    
    # Start
    ## Test whether service is accessible and then register useful information
    ExecStart=/bin/bash -c '\
      while true; do \
        curl -f ${COREOS_PUBLIC_IPV4}:80; \
        if [ $? -eq 0 ]; then \
          etcdctl set /services/apache/${COREOS_PUBLIC_IPV4} \'{"host": "%H", "ipv4_addr": ${COREOS_PUBLIC_IPV4}, "port": 80}\' --ttl 30; \
        else \
          etcdctl rm /services/apache/${COREOS_PUBLIC_IPV4}; \
        fi; \
        sleep 20; \
      done'
    
    # Stop
    ExecStop=/usr/bin/etcdctl rm /services/apache/${COREOS_PUBLIC_IPV4}
    
    [X-Fleet]
    # Schedule on the same machine as the associated Apache service
    X-ConditionMachineOf=apache.1.service

和讲述主service一样，先从文件的unit节开始。依赖关系和启动顺序就不谈了。

第一个新的指令（参数）是`BindsTo=`，这个指令意味着
当fleet start、stop和restart `apache.1.service`时，对`apache-discovery.1.service`也做同样操作。这意味着，我们使用fleet 只操作一个主服务，即同时操作了主服务与与之"BindsTo"的服务。并且，这个机制是单向的，fleet操作`apache-discovery.1.service`对 `apache.1.service`没有影响。

对于service节，我们应用了`/etc/environment`文件，因为我们需要它包含的环境变量。`ExecStart=`在此处是一个bash脚本，它企图使用暴漏的ip和端口访问主服务。

如果连接成功，使用etcdctl设置etcd中key为`/services/apache/{COREOS_PUBLIC_IPV4}`的值为一个json串：`{"host": "%H", "ipv4_addr": ${COREOS_PUBLIC_IPV4}, "port": 80}`。这个key值的有效期为30s，以此来确保一旦这个节点down掉，etcd中不会残留这个key值的信息。如果连接失败，则立即从etcd移除key值，因为主服务已经失效了。

循环有20s的休息时间，每隔20秒（在30s内etcd中key失效之前）重新检查一下主服务是否有效并重置key。这将更新ttl时间，确保在下一个30秒内key值是有效的。

本例中，stop指令仅仅是将key从etcd中移除，当stop主服务时，因为`BIndsTo=`，本服务也将被执行，从而从etcd移除注册信息。

对于X-fleet小节，我们需要确保该unit和主unit运行在一台机器上。结合`BindsTo=`指令，这样做可以将主节点信息汇报到远程主机上。




## Fleet-Specific Considerations


X-Fleet小节用来描述如何调度unit文件，具有以下指令（属性）：

- X-ConditionMachineID: 它可以指定一个特定的machine来加载unit。

- X-ConditionMachineOf: 它用来设定将unit部署到运行某个特定unit的machine上。

- X-Conflicts: 这个和上一个参数的作用恰恰相反, 它指定了本unit不想和哪些unit运行在同一台机器上。

- X-ConditionMachineMetadata: 它基于machine的metadata来确定调度策略。

- Global: 是一个布尔值，用来确定你是否想把unit部署到集群的每一台machine上。

这些额外的directives（属性）让管理员灵活的在集群上部署服务，在unit被提交到machinie的systemd实例之前，（directives）指定的策略将会被执行。

还有一个问题，什么时候使用fleet来部署相关的unit。除了X-fleet 小节外，fleetctl工具不检查units之间的依赖是否满足。因此在使用fleet部署相关的unit时，会有一些有趣的问题。

这意味着，尽管fleetctl工具会执行必要的步骤直到unit变成期望的状态：提交，加载，执行unit中的命令。但不会解决unit之间的依赖问题。

所以，如果你同时提交了主从两个unit:A和B，但A和B还没有加载。执行`fleetctl start A.service`将加载并执行`A.service` unit。然而，因为`B.service` unit还没有加载，并且因为fleetctl 不会检查unit之间的依赖，`A.service`将会执行失败。因为systemd会检查依赖是否满足，一旦主机的systemd开始执行`A.service`，却没有找到`B.service`，systemd便会终止`A.service`的执行。

为了避免这种伙伴unit执行失败的情况，你可以手动同时启动两个unit`fleet start A.service B.service`  ，不要依赖`BindsTo=`执令。

另一种方法是当运行主unit的时候，确保从unit至少已经被加载。**被加载意味着一台machine已经被选中，并且unit file已被提交到systemd实例**，此时满足了依赖条件，`BindsTo`参数也能够正确执行。

    fleetctl load A.service B.service
    fleetctl start A.service
    
如果fleetctl执行多个unit时失败，请记起这一点。

## Instances and Templates

unit template是fleet一个非常有用的概念。

unit templates依赖systemd一个叫做"instance"的特性。systemd运行时通过计算template unit文件实例化的unit file。template文件和普通unit文件大致相同，只有一点改进。但如果正确使用，威力无穷。

你可以在文件名中加入"@"来标识一个template 文件，比如一个传统的service文件名：`unit.service`，对应template文件则可以叫做`unit@.service`。
当template文件被实例化的时候，在"@"和".service"之间将有一个用户指定的标识符：`unit@instance_id.service`。在template文件内部，可以用%p来指代文件名（即unit），类似的，%i可以指代标识符（即instance_id）。

## Main Unit File as a Template

我们可以创建一个template文件:apache@.service。

    [Unit]
    Description=Apache web server service on port %i
    
    # Requirements
    Requires=etcd.service
    Requires=docker.service
    Requires=apache-discovery@%i.service
    
    # Dependency ordering
    After=etcd.service
    After=docker.service
    Before=apache-discovery@%i.service
    
    [Service]
    # Let processes take awhile to start up (for first run Docker containers)
    TimeoutStartSec=0
    
    # Change killmode from "control-group" to "none" to let Docker remove
    # work correctly.
    KillMode=none
    
    # Get CoreOS environmental variables
    EnvironmentFile=/etc/environment
    
    # Pre-start and Start
    ## Directives with "=-" are allowed to fail without consequence
    ExecStartPre=-/usr/bin/docker kill apache.%i
    ExecStartPre=-/usr/bin/docker rm apache.%i
    ExecStartPre=/usr/bin/docker pull username/apache
    ExecStart=/usr/bin/docker run --name apache.%i -p ${COREOS_PUBLIC_IPV4}:%i:80 \
    username/apache /usr/sbin/apache2ctl -D FOREGROUND
    
    # Stop
    ExecStop=/usr/bin/docker stop apache.%i
    
    [X-Fleet]
    # Don't schedule on the same machine as other Apache instances
    X-Conflicts=apache@*.service


就像你看到的，我们将`apache-discovery.1.service`改为`apache-discovery@%i.service`。即如果我们有一个unit文件`apache@8888.service`，它将需要一个从服务`apache-discovery@8888.service`。%i 曾被 实例化的标识符（即8888） 替换过。在这个例子中，%i也可以被用来指代服务的一些信息，比如apahce server运行时占用的端口。

为了使其工作，我们改变了`docker run`的参数，将container的80端口，映射给主机的某一个端口。在静态的unit 文件中，我们使用`${COREOS_PUBLIC_IPV4}:80:80`,将container的80端口，映射到主机${COREOS_PUBLIC_IPV4}网卡的80端口。在这个template 文件中，我们使用`${COREOS_PUBLIC_IPV4}:%i:80`，使用`%i`来说明我们使用哪个端口。在template文件中选择合适的instance identifier会带来很大的灵活性。

container的名字被改为了基于instance ID的`apache.%i`，**记住container的名字不能够使用`@`标识符**。现在，我们校正了运行container用到的所有指令（参数）。

在X-Fleet小节，我们同样改变了部署信息来替代原先的配置。

## Sidekick Unit as a Template

将从unit文件模板化也是类似的过程，新的从unit文件叫`apache-discovery@.service`，内容如下：

    [Unit]
    Description=Apache web server on port %i etcd registration
    
    # Requirements
    Requires=etcd.service
    Requires=apache@%i.service
    
    # Dependency ordering and binding
    After=etcd.service
    After=apache@%i.service
    BindsTo=apache@%i.service
    
    [Service]
    
    # Get CoreOS environmental variables
    EnvironmentFile=/etc/environment
    
    # Start
    ## Test whether service is accessible and then register useful information
    ExecStart=/bin/bash -c '\
      while true; do \
        curl -f ${COREOS_PUBLIC_IPV4}:%i; \
        if [ $? -eq 0 ]; then \
          etcdctl set /services/apache/${COREOS_PUBLIC_IPV4} \'{"host": "%H", "ipv4_addr": ${COREOS_PUBLIC_IPV4}, "port": %i}\' --ttl 30; \
        else \
          etcdctl rm /services/apache/${COREOS_PUBLIC_IPV4}; \
        fi; \
        sleep 20; \
      done'
    
    # Stop
    ExecStop=/usr/bin/etcdctl rm /services/apache/${COREOS_PUBLIC_IPV4}
    
    [X-Fleet]
    # Schedule on the same machine as the associated Apache service
    X-ConditionMachineOf=apache@%i.service

当主unit和从unit都是静态文件的时候，我们已经讲述了如何将从unit绑定到主unit，现在我们来讲下如何将实例化的从 unit绑定到同样根据模板实例化的主unit。

我们知道，`curl`命令可以被用来检查服务的有效性，为了让其连接到正确的url，我们用instance ID来代替80端口。这是非常必要的，因为在主unit中，我们改变了container映射的端口。

我们改变了写入到etcd的端口信息，同样是使用instance id来替换80端口。这样，设置到etcd的json信息就可以是动态的。它将采集service服务所在主机的hostname,ip地址和端口信息。

最后，我们也更改了X-Fleet小节，我们需要确定这个进程和其对应的主unit运行在一台主机上。

## Instantiating Units from Templates

实例化template文件有多种方法：

fleet和systemd都可以处理链接，我们可以创建模板文件的链接文件：

    ln -s apache@.service apache@8888.service
    ln -s apache-discovery@.service apache-discovery@8888.service
   
这将创建两个链接，叫做`apache@8888.service` 和 `apache-discovery@8888.service`，每个文件都包含了fleet和systemd运行它们所需要的所有信息，我们可以使用`fleetctl`提交、加载和启动这些服务`fleetctl start apache@8888.service apache-discovery@8888.service`。

如果你不想使用链接文件的方式，可以直接使用`fleetctl`来提交模板文件：`fleetctl submit apache@.service apache-discovery@.service`，你可以在运行时为其赋予一个instance identifier，比如：`fleetctl start apache@8888.service apache-discovery@8888.service`

这种方式不需要链接文件，然而一些系统管理员偏好于使用链接文件的方式。因为链接文件一旦创建完毕便可以在任意时刻使用，并且，如果你将所有的链接文件放在一个地方，便可以在同一时刻启动所有实例。

假设我们将静态unit文件全放在static目录下，模板文件放在templates目录下，根据模板文件创建的链接文件全放在instances目录下，你就可以用`fleetctl start instances/*`一次启动所有的实例。如果你的服务很多的话，这将非常方便。

## 小结

通过本章，相信你对如何创建unit文件以及fleet的使用有了深入的了解。利用unit文件中灵活的指令（参数），你可以将你的服务分布化，将相互依赖的服务部署在一起，并将它们的信息注册到etcd中。

本文已授权在[dockerpool][]上发表，欢迎大家参与讨论！


[我们以往的教程]: https://www.digitalocean.com/community/tutorials/how-to-set-up-a-coreos-cluster-on-digitalocean
[如何使用fleetctl]: https://www.digitalocean.com/community/tutorials/how-to-use-fleet-and-fleetctl-to-manage-your-coreos-cluster
[上一个guide]: https://www.digitalocean.com/community/tutorials/how-to-use-fleet-and-fleetctl-to-manage-your-coreos-cluster
[basic guide on running services on CoreOS]: https://www.digitalocean.com/community/tutorials/how-to-create-and-run-a-service-on-a-coreos-cluster
[How to Create Flexible Services for a CoreOS Cluster with Fleet Unit Files]: https://www.digitalocean.com/community/tutorials/how-to-create-flexible-services-for-a-coreos-cluster-with-fleet-unit-files
[http://qiankunli.github.io/ ]: http://qiankunli.github.io/ 
[qiankun.li@qq.com]: qiankun.li@qq.com