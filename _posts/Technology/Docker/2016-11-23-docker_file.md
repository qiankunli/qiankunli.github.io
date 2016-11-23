---

layout: post
title: docker存储驱动
category: 技术
tags: Docker
keywords: Docker,graphdriver

---

## 简介


docker关于存储方面的总体的理念是`分层（因为不想像virtualbox一样copy），层之间有父子关系`。分层内容的“镜像图（image graph）”代表了各种分层之间的关系，用来处理这些分层的驱动就被叫做“图驱动（graphdriver）”。docker1.2.0 源码中，驱动接口如下


	type Driver interface {
		String() string
		Create(id, parent string) error
		Remove(id string) error
		Get(id, mountLabel string) (dir string, err error)
		Put(id string)
		Exists(id string) bool
		Status() [][2]string
		Cleanup() error
	}
	
换句话说，接口表述了docker上层操作的要求。“用到的数据都在这里了，但实现这个Driver interface，根据文件系统的不同，算法是不同的，这个算法，就是graphdriver”


## 一些基础知识

### 挂载/mount

本质上，Ext3 mount的过程实际上是inode被替代的过程。例如，`/dev/sdb`块设备被mount到`/mnt/alan`目录。命令：`mount -t ext3 /dev/sdb /mnt/alan`。

那么mount这个过程所需要解决的问题就是将/mnt/alan的dentry目录项所指向的inode屏蔽掉，然后重新定位到/dev/sdb所表示的inode索引节点。(实际实现略有不同)[linux文件系统之mount流程分析](http://www.cnblogs.com/cslunatic/p/3683117.html)

但除了将设备mount到rootfs上，mount的花样可多了。

### bind mount

mount --bind olddir newdir

ln分为软链接和硬链接

硬链接只能连文件 你删除它源文件就被删了

软链接相当于windows的快捷方式，文件和目录都可以连，但是你删出它只是删除的这个快捷方式

在操作系统中ln -s和mount -bind也许使用起来没多少差别,但是ftp目录里貌似不能放软连接。把一个文件夹mount到另一个地方，等于在把它挂载到那个地方当成一个磁盘，进到两个文件夹中操作 都是等价的，删除一个文件也会在两边同时删除。

### union mount

参见[Union mount](https://en.wikipedia.org/wiki/Union_mount)，In computer operating systems, union mounting is a way of combining multiple directories into one that appears to contain their combined contents.

union mount的使用场景，As an example application of union mounting, consider the need to update the information contained on a CD-ROM or DVD. While a CD-ROM is not writable, one can overlay the CD's mount point with a writable directory in a union mount. Then, updating files in the union directory will cause them to end up in the writable directory, giving the illusion that the CD-ROM's contents have been updated.

Union FileSystem的核心逻辑是Union Mount，它支持把一个目录A叠加到另一个目录B之上；用户对目录B的读取就是A加上B的内容，而对B目录里文件写入和改写则会保存在目录A上，因为A在上一层。这个类似差分VHD的效果，但是是以文件为单位的。


## 几大文件系统

aufs,vfs,devicemapper,btrfs,它们之间的关系。参见[容器（Docker）概念中存储驱动的深入解析](http://weibo.com/ttarticle/p/show?id=2309404039168383667054)

1. vfs，屏蔽底层的文件系统差异，比如ext2，ext3。在docker 场景下，vfsgraph指的是不提供共享layer的特性。对于vfs来说，Driver.create layer最上层，就是将镜像所有的分层依次拷贝到静态的子文件夹中，然后将这个文件夹挂载到容器的根文件系统。

2. 联合文件系统，aufs,它不是一个真正的文件系统，如ext4或者xfs，它仅仅是站在一个已有的文件系统上提供了这些功能。对于aufs来说，Driver.create layer最上层，就是创建一个新的文件夹（读写layer），将镜像的所有只读层挂进来，然后将这个文件夹挂载到容器的根文件系统。至于说，写时复制，这个就是aufs 文件系统的功能，跟graphDriver就没有关系了。

3. 特定文件系统的实现（devicemapper，btrfs等）。是一个具体的文件系统，有一定的内置特性，比如快照。在这每一个情形中，你都需要新建一个磁盘(准确的说是块设备)并用该文件系统格式化磁盘（或者为了快速测试，用循环挂载的文件作为磁盘），来使用这些选项作为Docker引擎的存储后端。


所以，总结起来，参见[Supported Filesystems](http://www.projectatomic.io/docs/filesystems/)


All backends except the vfs one shares diskspace between base images. However, they work on different levels, so the behaviour is somewhat different. Both devicemapper and btrfs share data on the **block level**, so a single change in a file will cause just the block containing that byte being duplicated. However the aufs backend works on the **file level**, so any change to a file means the entire file will be copied to the top layer and then changed there. The exact behaviour here therefore depends on what kind of write behaviour an application does.

However, any kind of write-heavy load(写负载比较大) inside a container (such as databases or large logs) should generally be done to a volume.（共享文件系统有共享文件系统的问题，所以写负载比较大的操作，还要弄到volume中） A volume is a plain directory from the host mounted into the container, which means it has none of the overhead that the storage backends may have. It also means you can easily access the data from a new container if you update the image, or if you want to access the same data from multiple concurrent containers.（这段解释了我们为什么要用volume）


## 引用

[解析 Linux 中的 VFS 文件系统机制](https://www.ibm.com/developerworks/cn/linux/l-vfs/)
