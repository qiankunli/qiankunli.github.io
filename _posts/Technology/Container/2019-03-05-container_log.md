---

layout: post
title: 容器狂打日志怎么办？
category: 技术
tags: Container
keywords: container log

---

## 简介

* TOC
{:toc}

除了docker image 时间长了会占用大量磁盘空间外（参见[关于docker image的那点事儿](http://qiankunli.github.io/2015/09/22/docker_image.html)），容器在运行时大量写日志也是个很头疼的问题。

最近碰到一个问题，部分项目在容器疯狂打日志，把磁盘都弄满了，弄满的原因有以下两个

1. 日志写到了stdout下，详情见下文 
2. 日志写到某个文件下，日志数量本身就很大

针对日志文件过大的问题，有几种方法

1. 堵，限定一个容器最多使用多少磁盘空间
2. 疏，以特定用户运行项目，该容器内用户只可以访问特定的文件夹，比如/logs，然后将容器/logs 映射到物理机上，定时清理
3. 监控，随时监控磁盘，异常时报警
4. 日志由日志采集工具收集，不在磁盘上停留

## stdout log

[JSON File logging driver](https://docs.docker.com/config/containers/logging/json-file/) By default, Docker captures the standard output (and standard error) of all your containers, and writes them in files using the JSON format. 对于一个容器来说，当应用把日志输出到 stdout 和 stderr 之后，容器项目在默认情况下就会把这些日志输出到宿主机上的一个 JSON 文件里。

[「Allen 谈 Docker 系列」之 docker logs 实现剖析](http://blog.daocloud.io/allen_docker01/)

对于应用的标准输出(stdout)日志，Docker Daemon 在运行这个容器时就会创建一个协程(goroutine)，负责标准输出日志。由于此 goroutine 绑定了整个容器内所有进程的标准输出文件描述符，因此容器内应用的所有标准输出日志，都会被 goroutine 接收。goroutine 接收到容器的标准输出内容时，立即将这部分内容，写入与此容器—对应的日志文件中，日志文件位于`/var/lib/docker/containers/<container_id>`，文件名为<container_id>-json.log。

![](/public/upload/docker/docker_log.png)

Docker 则通过 docker logs 命令向用户提供日志接口。`docker logs` 实现原理的本质均基于与容器一一对应的 <container-id>-json.log，`kubectl logs`类似

从这可以看到几个问题

1. app 同时输出文件日志和stdout 是一种浪费
2. stdout 日志在 `/var/lib/docker/containers/<container_id>` 下可以被清理， 也可以配置 docker daemon 设置 log-driver 和 log-opts 参数

		 "log-driver":"json-file",
	  	 "log-opts": {"max-size":"500m", "max-file":"3"}
	  	 
3. 这部分文件过大，带来的另一个问题是，删除容器时 json-file 所在的`/var/lib/docker/containers/$ContainerId/xx-json.log` 依然残留在物理机磁盘上，成为耗尽磁盘的定时炸弹
4. 使用定时任务每天执行`docker system prune -af`


## 堵

### 限定容器占用的磁盘空间

[docker的storage-driver是overlay2时，限制单个容器可占用的磁盘空间](https://www.lijiaocn.com/%E9%97%AE%E9%A2%98/2018/12/26/docker-overlay2-size-limit.html)

几个关键字

1. xfs，[linux 文件系统](http://qiankunli.github.io/2018/05/19/linux_file_mount.html) CentOS 7开始，预设的文件系统由原来的EXT4变成了XFS文件系统
2. pquota，也就是 project quotas ，[How to Enable Disk Quotas on an XFS File System](https://www.thegeekdiary.com/how-to-enable-disk-quotas-on-an-xfs-file-system/)XFS supports disk quotas by user, by group, and by project. Project disk quotas allow you to limit the amount of disk space on individual directory hierarchies. 限定一个目录的大小

        # mount 时 指定文件系统类型，使用-o enbale project quotas
        mount –o prjquota /dev/xvdb1 /xfs
        # 限定 project=test 的 /data 目录 soft limit=5M hard limit=6M
        xfs_quota –x –c ‘limit –p bsoft=5m bhard=6m test’ /data

3. `/etc/docker/daemon.json`配置文件如下，这里将每个容器可以使用的磁盘空间设置为1G：

        {
            "data-root": "/data/docker",
            "storage-driver": "overlay2",
            "storage-opts": [
            "overlay2.override_kernel_check=true",
            "overlay2.size=1G"
            ]
        }


### 限定volume 占用的磁盘空间

这是一个传统的文件夹 大小限定问题


## 疏 ==> 非root用户运行进程

[docker和k8s安全机制](http://qiankunli.github.io/2019/03/29/kubernetes_security.html)


## 监控

从物理机维度，当物理机磁盘 剩余到一定占比时 报警

从容器维度 则有两个问题

1. 如果不限制项目的日志文件目录的话，如何自动感知项目的日志文件目录位置？暂时没有找到好办法。
2. 如何清理日志文件？Linux或者Unix系统中，通过`rm -rf`或者文件管理器删除文件，将会从文件系统的目录结构上解除链接（unlink）。如果文件是被打开的（有一个进程正在使用），那么进程将仍然可以读取该文件，磁盘空间也一直被占用。正确姿势是`cat /dev/null > 目标文件`

从物理机角度，有一个方案是执行`docker system df -v` 可以列出每个容器占用的 磁盘空间，当期大小超过一定阈值时，可以根据container id（想办法将container id 与应用信息关联起来） 将其删除。


    CONTAINER ID        IMAGE                                                                                       COMMAND                  LOCAL VOLUMES       SIZE                CREATED ago             STATUS              NAMES
    2ba3bb81f4a6        harbor.test.ximalaya.com/test/wws-library-web:20190305-190207                               "/sbin/my_init"          0                   3.76MB              40 minutes ago ago      Up 40 minutes       mesos-8f4307c7-6a44-467e-9a94-56e09182013d
    98e129663d1c        harbor.test.ximalaya.com/test/anchor-sell-web:20190305-182739                               "/sbin/my_init"          0                   2.47MB              About an hour ago ago   Up About an hour    mesos-60309b8a-27bd-4744-99f9-685f68dca71a
    cd38d9c7fb71        test/docker-count-service-album-test:6                                                      "/usr/local/tomcat/b…"   0                   49.2MB              2 hours ago ago         Up 2 hours          mesos-33f4264e-77fc-4a4f-84c7-aae78519c0ad

现在就是说不清楚，其size  列的大小说的是哪部分？笔者只找到了其中的一半。

## 其它

为提高系统友好性，在删除项目日志后 应向负责人发一个消息提醒。

docker 还可以限制磁盘的读写速度 [限制容器的 Block IO - 每天5分钟玩转 Docker 容器技术（29）](https://www.ibm.com/developerworks/community/blogs/132cfa78-44b0-4376-85d0-d3096cd30d3f/entry/%E9%99%90%E5%88%B6%E5%AE%B9%E5%99%A8%E7%9A%84_Block_IO_%E6%AF%8F%E5%A4%A95%E5%88%86%E9%92%9F%E7%8E%A9%E8%BD%AC_Docker_%E5%AE%B9%E5%99%A8%E6%8A%80%E6%9C%AF_29?lang=en)


