---

layout: post
title: linux 文件系统
category: 技术
tags: Linux
keywords: network 

---

## 简介

linux系统的进程结构体有以下几个字段

    struct task_struct {
				...
				struct m_inode * pwd;
				struct m_inode * root;
				struct m_inode * executable;				//进程对应可执行文件的i节点
				...
				/* Filesystem information: */
				struct fs_struct                *fs;
				/* Open file information: */
				struct files_struct             *files;
				...
    }

每个进程有一个文件系统的数据结构，还有一个打开文件的数据结构

## vfs

[解析 Linux 中的 VFS 文件系统机制](https://www.ibm.com/developerworks/cn/linux/l-vfs/)

VFS 是文件系统事实上的规范，定义了挂载点、超级块、目录和索引节点等基本数据结构，定义了open/close/write/read 等基本接口

文件系统 是解决 根据 file name 找 file data的问题，从这个角度看，文件系统跟dns 有点异曲同工的意思。

文件系统有内存文件系统，磁盘文件系统，还有基于磁盘文件系统之上的联合文件系统。

## linux 文件系统的加载

rootfs是基于内存的文件系统，所有操作都在内存中完成；也没有实际的存储设备，所以不需要设备驱动程序的参与。基于以上原因，Linux在启动阶段使用rootfs文件系统，当磁盘驱动程序和磁盘文件系统成功加载后，linux系统会将系统根目录从rootfs切换到磁盘文件系统（这句表述不准确）。

参见[linux文件系统初始化过程(2)---挂载rootfs文件系统
](http://blog.csdn.net/luomoweilan/article/details/17894473),linux文件系统中重要的数据结构有：文件、挂载点、超级块、目录项、索引节点等。图中含有两个文件系统（红色和绿色表示的部分），并且绿色文件系统挂载在红色文件系统tmp目录下。

1. 一般来说，每个文件系统在VFS层都是由挂载点、超级块、目录和索引节点组成。PS，这点更新了以前的认知，课本上一般只体现超级块、目录和索引节点。
2. 当挂载一个文件系统时，实际也就是创建这四个数据结构的过程，因此这四个数据结构的地位很重要，关系也很紧密。
3. **由于VFS要求实际的文件系统必须提供以上数据结构，所以不同的文件系统在VFS层可以互相访问。**
4. 如果进程打开了某个文件，还会创建file(文件)数据结构，这样进程就可以通过file来访问VFS的文件系统了。

![](/public/upload/linux/linux_fs.png)

从图中可以看到：

1. 这个图从上往下看，可以知道，各个数据结构通过数组等自己组织在一起，又通过引用上下关联。
2. 超级块、目录和索引节点这些，完成数据的组织和根据文件名寻址。
3. 从上图可以理清挂载点和前三个结构的关系，**挂载点 将 文件系统彼此勾连 起来**。

[Mount Point Definition](http://www.linfo.org/mount_point.html)

1. A filesystem is a hierarchy of directories (also referred to as a directory tree) that is used to organize files on a computer system. On Linux and other Unix-like operating systems, at the very top of this hierarchy is the root directory. 一个文件 系统是一个directory tree，根节点被称为root directory
2. The mount point becomes the root directory of the newly added filesystem, and that filesystem becomes accessible from that directory. vfs 支持同时加载多个文件系统，也就是有多个directory tree 和 root directory。vfs 也是一个directory tree，只有一个root directory，那么必然有很多fs 的root directory 挂载在了vfs directory tree 的非root directory 上。 
3. A mount point is a directory。可以猜测，如果没有挂载点的概念，也就是只有超级块、目录和索引节点，那么类似的效果需要一个 目录 去引用一个超级块。
3. `/etc/fstab` 列出vfs 当前挂载的文件系统以及它们的挂载点

		UUID=2d147dd5-0227-40e5-a143-1923112cb1bd /                       ext4    defaults        1 1
		UUID=1d44ef13-382c-49e5-925a-9eac34bc575d swap                    swap    defaults        0 0
		tmpfs                   /dev/shm                tmpfs   defaults        0 0
		devpts                  /dev/pts                devpts  gid=5,mode=620  0 0
		sysfs                   /sys                    sysfs   defaults        0 0
		proc                    /proc                   proc    defaults        0 0
		
![](/public/upload/linux/linux_fs_1.png)
		
1. 有了挂载点、超级块、目录和索引节点 这几个结构，可以让一个vfs/rootfs 挂载多个 不同的fs
2. 加上linux mount namespace这个结构，可以使得一个vfs 有多个 rootfs
3. 在linux 操作系统中，文件系统和linux 内核是分开存放的，操作系统只有在开机启动时才会加载指定版本的内核镜像。如果一个rootfs 指向一个包含完整操作系统文件的目录，则这个rootfs 便可以作为一个完整的进程文件“环境”。
4. 有了联合文件系统 可读、可写挂载这一套特性，可以让一个rootfs 有layer 感觉

### 几大文件系统

aufs,vfs,devicemapper,btrfs,它们之间的关系。参见[容器（Docker）概念中存储驱动的深入解析](http://weibo.com/ttarticle/p/show?id=2309404039168383667054)

1. vfs，屏蔽底层的文件系统差异，比如ext2，ext3。在docker 场景下，vfsgraph指的是不提供共享layer的特性。对于vfs来说，Driver.create layer最上层，就是将镜像所有的分层依次拷贝到静态的子文件夹中，然后将这个文件夹挂载到容器的根文件系统。

2. 联合文件系统，aufs,它不是一个真正的文件系统，如ext4或者xfs，它仅仅是站在一个已有的文件系统上提供了这些功能。对于aufs来说，Driver.create layer最上层，就是创建一个新的文件夹（读写layer），将镜像的所有只读层挂进来，然后将这个文件夹挂载到容器的根文件系统。至于说，写时复制，这个就是aufs 文件系统的功能，跟graphDriver就没有关系了。

3. 特定文件系统的实现（devicemapper，btrfs等）。是一个具体的文件系统，有一定的内置特性，比如快照。在这每一个情形中，你都需要新建一个磁盘(准确的说是块设备)并用该文件系统格式化磁盘（或者为了快速测试，用循环挂载的文件作为磁盘），来使用这些选项作为Docker引擎的存储后端。CentOS 7开始，预设的文件系统由原来的EXT4变成了XFS文件系统


所以，总结起来，参见[Supported Filesystems](http://www.projectatomic.io/docs/filesystems/)


All backends except the vfs one shares diskspace between base images. However, they work on different levels, so the behaviour is somewhat different. Both devicemapper and btrfs share data on the **block level**, so a single change in a file will cause just the block containing that byte being duplicated. However the aufs backend works on the **file level**, so any change to a file means the entire file will be copied to the top layer and then changed there. The exact behaviour here therefore depends on what kind of write behaviour an application does.

However, any kind of write-heavy load(写负载比较大) inside a container (such as databases or large logs) should generally be done to a volume.（共享文件系统有共享文件系统的问题，所以写负载比较大的操作，还要弄到volume中） A volume is a plain directory from the host mounted into the container, which means it has none of the overhead(天花板) that the storage backends may have. It also means you can easily access the data from a new container if you update the image, or if you want to access the same data from multiple concurrent containers.（这段解释了我们为什么要用volume）

### 传统的linux fs加载过程

参见`https://www.kernel.org/doc/Documentation/filesystems/ramfs-rootfs-initramfs.txt`

What is rootfs?

Rootfs is a special instance of ramfs (or tmpfs, if that's enabled), which is
always present in 2.6 systems.it's smaller and simpler for the kernel
to just make sure certain lists can't become empty.Most systems just mount another filesystem over rootfs and ignore it（一般情况下，通过某种文件系统挂载内容至挂载点的话，挂载点目录中原先的内容将会被隐藏）.


What is initramfs?

All 2.6 Linux kernels contain a gzipped "cpio" format archive, which is
extracted into rootfs when the kernel boots up.  After extracting, the kernel
checks to see if rootfs contains a file "init", and if so it executes it as PID
1.  If found, this init process is responsible for bringing the system the
rest of the way up, including locating and mounting the real root device (if
any).  If rootfs does not contain an init program after the embedded cpio
archive is extracted into it, the kernel will fall through to the older code
to locate and mount a root partition, then exec some variant of /sbin/init
out of that.

所以一个linux的启动过程经历了rootfs ==> 挂载initramfs ==> 挂载磁盘上的真正的fs

为什么要有initrd？

linux系统在启动时，会执行文件系统中的`/sbin/init`进程完成linux系统的初始化，执行`/sbin/init`进程的前提是linux内核已经拿到了存在硬盘上的系统镜像文件（加载设备驱动，挂载文件系统）。linux 发行版必须适应各种不同的硬件架构，将所有的驱动编译进内核是不现实的。Linux发行版在内核中只编译了基本的硬件驱动
，在 linux内核启动前，boot loader会将存储介质中的initrd文件(cpio是其中的一种)加载到内存，内核启动时会在访问真正的根文件系统前先访问该内存中的initrd文件系统，执行initrd文件系统的某个文件（不同的linux版本差异较大），扫描设备，加载驱动。


## 镜像文件

An archive file is a file that is composed of one or more computer files **along with metadata**. Archive files are used to collect multiple data files together into a single file for easier portability and storage, or simply to compress files to use less storage space. Archive files often store directory structures, error detection and correction information, arbitrary comments, and sometimes use built-in encryption.


## 挂载/mount

例如，`/dev/sdb`块设备被mount到`/mnt/alan`目录。命令：`mount -t ext3 /dev/sdb /mnt/alan`。

那么mount这个过程所需要解决的问题就是将`/mnt/alan`的dentry目录项所指向的inode屏蔽掉，重新定位到`/dev/sdb`所表示的inode索引节点。这个描述并不准确，但有利于简化理解。参见[linux文件系统之mount流程分析](http://www.cnblogs.com/cslunatic/p/3683117.html)

**如果将mount的过程理解为：inode被替代的过程。**除了将设备mount到rootfs上，根据被替代方式的不同，mount的花样可多了。

||一般用途|备注|
|---|---|---|
|mount|挂载设备|**需要加载设备的super block**，关联到inode| 
|bind mount|挂载目录|替代inode| 
|union mount|合并目录|有机的整合几个inode为一个新的inode，替代原来的inode| 

### bind mount

mount --bind olddir newdir

ln分为软链接和硬链接

硬链接只能连文件 你删除它源文件就被删了

软链接相当于windows的快捷方式，文件和目录都可以连，但是你删出它只是删除的这个快捷方式

在操作系统中ln -s和mount -bind也许使用起来没多少差别,但是ftp目录里貌似不能放软连接。把一个文件夹mount到另一个地方，等于在把它挂载到那个地方当成一个磁盘，进到两个文件夹中操作 都是等价的，删除一个文件也会在两边同时删除。

### union mount

参见[Union mount](https://en.wikipedia.org/wiki/Union_mount)，In computer operating systems, union mounting is a way of combining multiple directories into one that appears to contain their combined contents.

union mount的使用场景——读写只读文件系统，As an example application of union mounting, consider the need to update the information contained on a CD-ROM or DVD. While a CD-ROM is not writable, one can overlay the CD's mount point with a writable directory in a union mount. Then, updating files in the union directory will cause them to end up in the writable directory, giving the illusion that the CD-ROM's contents have been updated. 

Union FileSystem的核心逻辑是Union Mount，它支持把一个目录A和另一个目录B union，提供一个虚拟的C目录（目录union的语义）。对于特定的权限设置策略，如果设置A目录可写，B目录只读。用户对目录C的读取就是A加上B的内容，而对C目录里文件写入和改写则会保存在目录A上。这样，就有了A在B上层的感觉。
[DOCKER基础技术：AUFS](http://coolshell.cn/articles/17061.html)

[Docker存储驱动简介](https://linux.cn/thread-16017-1-1.html)


## 引用

[存储之道 - 51CTO技术博客 中的《一个IO的传奇一生》](http://alanwu.blog.51cto.com/3652632/d-8)



