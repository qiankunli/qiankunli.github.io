---

layout: post
title: docker基本原理
category: 技术
tags: Docker
keywords: Docker namespace cgroup

---
## 前言

Docker主要利用Linux内核的namespace机制实现沙盒隔离和cgroup机制实现资源限制，那么接下来我们就这两个方面展开叙述。

linux内核水平有限，这里只是粗略涉及。

未完待续

## linux namespace

命名空间最初是用来解决命名唯一性问题的，即解决不同编码人员编写的代码模块在合并时可能出现的重名问题。

传统上，在Linux以及其他衍生的UNIX变体中，许多资源是全局管理的。这意味着进程之间彼此可能相互影响。偏偏有这样一些场景，比如一场“黑客马拉松”的比赛，组织者需要运行参赛者提供的代码，为了防止一些恶意的程序，必然要提供一套隔离的环境，一些提供在线持续集成服务的网站也有类似的需求。

我们不想让进程之间相互影响，就必须将它们隔离起来，最好都不知道对方的存在。而所谓的隔离，便是隔离他们使用的资源（比如），进而资源的管理也不在是全局的了。

    struct task_struct {
        ……
        /* namespaces */
        struct nsproxy *nsproxy;
        …….
    }
    struct nsproxy {
        atomic_t count;
        struct uts_namespace *uts_ns;
        struct ipc_namespace *ipc_ns;
        struct mnt_namespace *mnt_ns;
        struct pid_namespace *pid_ns;
        struct user_namespace *user_ns;
        struct net             *net_ns;
    }




## linux cgroup

限制资源的使用

通过一定的数据结构及函数。比如统计进程的cpu使用时间（虚拟时间），并在创建和调用该进程时进行判断，如果没有超出相应额度，则调度其继续运行。

### /proc文件系统

`/proc`文件系统是由软件创建，被内核用来向外界报告信息的一个文件系统。/proc 下面的每一个文件都和一个内核函数相关联，当文件的被读取时，与之对应的内核函数用于产生文件的内容。

## 参考文献

[Docker基础技术：Linux CGroup][]

[Linux 命名空间][]


[Docker基础技术：Linux CGroup]: http://coolshell.cn/articles/17049.html#more-17049
[Linux 命名空间]: http://laokaddk.blog.51cto.com/368606/674256