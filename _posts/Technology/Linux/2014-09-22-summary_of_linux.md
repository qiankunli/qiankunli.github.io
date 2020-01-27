---
layout: post
title: Linux中的一些点
category: 技术
tags: Linux
keywords: Linux
---

## 前言 ##

本文记录一些日常使用linux的一些点。

## yum 升级回退

[在RHEL/CentOS系统上使用YUM history命令回滚升级操作](http://os.51cto.com/art/201801/563966.htm)

	yum history undo 13

## 找到第一次报错的位置 并显示前后几行

	grep ERROR  xxx.log -m 1 -A 10

## 删除符合正则规则的文件

假设存在以下文件


	album-listener.log.2018-01-30  
	album-facade.log             
	album-facade.log.2017-12-26  
	album-facade.log.2018-01-23  
	album-facade.log.2018-01-24  
	album-facade.log.2018-05-23  
	album-facade.log.2018-07-01  
	album-facade.log.2018-07-15  
	album-facade.log.2018-07-16  
	album-facade.log.2018-07-31

因为磁盘空间有限，先输出2017年的日志

1. 直观想法是rm，但rm 不支持正则表达式
2. 则可以先找到文件，在通过xargs 串联 rm，`ls | grep ".*2018-03.*" | xargs rm -rf`

后来试验了一下，`rm -rf *2018-03*` 也是可以的。

## 在一个目录下寻找包含某个关键字的文件

`grep -rn "交换机" dir`

## 系统负载评估

[理解Linux系统负荷](http://www.ruanyifeng.com/blog/2011/07/linux_load_average_explained.html)

## 查看 ps -ef

    [root@deployer ~]# ps -ef
    UID        PID  PPID  C STIME TTY          TIME CMD
    root         1     0  0  2018 ?        00:00:29 /usr/lib/systemd/systemd --system --deserialize 21
    root         2     0  0  2018 ?        00:00:00 [kthreadd]
    root         3     2  0  2018 ?        00:00:00 [ksoftirqd/0]
    root         5     2  0  2018 ?        00:00:00 [kworker/0:0H]
    root         9     2  0  2018 ?        00:00:40 [rcu_sched]
    ......
    root       337     2  0  2018 ?        00:00:01 [kworker/3:1H]
    root       380     1  0  2018 ?        00:00:00 /usr/lib/systemd/systemd-udevd
    root       415     1  0  2018 ?        00:00:01 /sbin/auditd
    root       498     1  0  2018 ?        00:00:03 /usr/lib/systemd/systemd-logind
    ......
    root@pts/0
    root     32794 32792  0 Jan10 pts/0    00:00:00 -bash
    root     32901 32794  0 00:01 pts/0    00:00:00 ps -ef

三类进程

1. pid=1 init进程Systemd
2. pid=2 内核线程kthreadd，用户态不带中括号， 内核态带中括号。在 linux 内核中有一些执行定时任务的线程, 比如定时写回脏页的 pdflush, 定期回收内存的 kswapd0, 以及每个 cpu 上都有一个负责负载均衡的 migration 线程等.
3. tty 带问号的，说明不是前台启动的，一般都是后台启动的服务

## Linux中的引号 ##

首先，单引号和双引号，都是为了解决中间有空格的问题。

- 单引号

	当shell碰到第一个单引号时，它忽略掉其后直到右引号的所有特殊字符

- 双引号

	双引号只要求忽略大多数，具体说，括在双引号中的三种特殊字符不被忽略：美元符号，反斜杠和反引号

- 反引号

    命令替换：在执行一条命令时，会先将其中反引号之间的语句，或者是`$()`中的语句当作命令执行一遍，再将结果加入到原命令中重新执行。其中，`$()`格式受到POSIX标准支持，也利于嵌套。比如`docker rm -f $(docker ps -aq)`。常见的用法：将命令的输出结果赋给一个变量
        
        var=`command`
        
## 软件包安装工具

- rpm & yum     后者可以解决依赖关系
- dpkg & apt    后者可以解决依赖关系

## pid文件 ##

- pid文件内容：pid文件为文本文件，内容只有一行, 记录了该进程的ID。 

- pid文件作用：防止进程启动多个副本。只有获得pid文件(固定路径固定文件名)写入权限(F_WRLCK)的进程才能正常启动并把自身的PID写入该文件中。其它同一个程序的多余进程则自动退出。

## /etc/issue和/etc/motd ##

登陆linux的欢迎界面可由/etc/issue和/etc/motd控制。/etc/issue文件的使用方法与/etc/motd文件相差不大,它们的主要区别在于：当一个网络用户或通过串口登录系统上时,/etc/issue的文件内容显示在login提示符之前,而/etc/motd内容显示在用户成功登录系统之后。

## socket文件 ##

socket API原本是为网络通讯设计的，但后来在socket的框架上发展出一种IPC机制，就是UNIXDomain Socket。虽然网络socket也可用于同一台主机的进程间通讯（通过loopback地址127.0.0.1），但是UNIX Domain Socket用于IPC更有效率：不需要经过网络协议栈，不需要打包拆包、计算校验和、维护序号和应答等，只是将应用层数据从一个进程拷贝到另一个进程。这是因为，IPC机制本质上是可靠的通讯，而网络协议是为不可靠的通讯设计的。UNIX Domain Socket也提供面向流和面向数据包两种API接口，类似于TCP和UDP，但是面向消息的UNIX Domain Socket也是可靠的，消息既不会丢失也不会顺序错乱。

UNIX Domain Socket是全双工的，API接口语义丰富，相比其它IPC机制有明显的优越性，目前已成为使用最广泛的IPC机制，比如X Window服务器和GUI程序之间就是通过UNIX Domain Socket通讯的。

来源于：[本地socket unix domain socket][]

socket文件，只有在本级通信（AF_UNIX）时才会用到；对于远程TCP通信（AF_INET）来说，是不存在sock文件一说的。以mysql.sock文件为例，假定AB两台主机，mysql服务运行在B主机上。

- 那么对于A主机来说，访问mysql服务只能通过`ip:port`的方式。
- 对于B主机上程序来说，访问mysql服务除了通过`localhost:port`外，还可以通过socket文件方式。后者的好处是：如果通过mysql.conf更改mysql的服务端口，那么mysql服务重新启动时将生成新的socket文件，对于程序来说，不用做任何改变。而前者则需要改变程序中的端口设置。

另外，指定socket文件url一个例子：`unix:///home/docker/docker.sock`，其中`unix://`类似于`tcp://`，后面的`/home/docker/docker.sock`表明socket文件的路径。

某些linux应用运行前，我们要为其配置监听的ip和端口，而支持sock文件后，我们只需为其配置sock文件即可，比如：

    $ sudo /usr/bin/docker –d –H unix:///home/docker/docker.sock &

而docker client在访问docker daemon时，需要：

    $ docker –H unix:///home/docker/docker.sock ps –a

## rsync

负责进行文件同步的

`rsync -vzrtopgu --delete dir_path1 dir_path2`

注意，dir_path1中的改变都会被同步到dir_path2中，但是，如果dir_path2改变，并且dir_path2在命令中的先后顺序不变的话，其改变不会被同步到dir_path1中。也就是说，这种同步是单向的，以dir_path1为准。

## source和**.** 命令##

有两种方法执行shell scripts:

1. 一种是新产生一个shell，然后执行相应的shell scripts
2. 一种是在当前shell下执行，不再启用其他shell

假设一文件`file`,文件内容为`echo "test!!!!!!!!!!!!!!!!!!!!!"`:

1. 没有可执行权限

  1.1. `./file` 输出`Permission denied`
  
  1.2. `. ./file` 执行成功
  
  1.3. `source file` 执行成功
  
2. 具有可执行权限，file添加一行为`a=test`

  2.1 `./file;echo $a` 输出"test!!!!!!!!!!!!!!!!!!!!!"`,没有输出$a
  
  2.2 `. ./file;echo $a` he `source file;echo $a`则都输出
  
`./file`中的点代表当前路径的意思，会产生一个新的子shell来执行file，也即file中如果设置环境变量，当前shell无法得到。子shell可以继承父shell的环境变量值。与此同时，file必须具有可执行权限。

`. ./file`中第一个点，代表`source`，第二个点代表当前路径。file不需要具备可执行权限，file中的所有指令将在当前shell中执行。也即如果file设置环境变量，当前shell也能感知到。 

## 一次创建多个文件夹

    mkdir -p /etc/confd/{conf.d,templates}
    
## 创建并管理自己的后台程序

/sbin/start-stop-daemon

## inotifywait

监控某个文件或目录的改变

## expect

进行自动化的人机交互，对于设定好的特定匹配形式，以相匹配的动作以应对。

## 测试某个远程主机的特定端口是否开放

1. telnet

    在终端使用者的电脑上使用telnet程序，用它连接到服务器。终端使用者可以在telnet程序中输入命令，这些命令会在服务器上运行，就像直接在服务器的控制台上输入一样。
    
    - 使用telnet操作`192.168.56.153`上的redis服务
    
            user@ubuntu:~$ telnet 192.168.56.153 6379
            Trying 192.168.56.153...
            Connected to 192.168.56.153.
            Escape character is '^]'.
            get name
            $3
            lqk
            
    - 直接在`192.168.56.153`上操作redis

            user@ubuntu:~$ redis-cli get name
            "lqk"
            
    - 退出telnet

        1. ctrl + ]
        2. telnet > quit    // 回车


2. 使用nc命令

        user@ubuntu:~$ nc -z 192.168.56.153 6379
        user@ubuntu:~$ echo $?
        0
        
    返回1 表示连接失败（或者说，端口没有开启）

## 引用

[本地socket unix domain socket](http://blog.csdn.net/bingqingsuimeng/article/details/8470029)

[让进程在后台可靠运行的几种方法
](https://www.ibm.com/developerworks/cn/linux/l-cn-nohup/)