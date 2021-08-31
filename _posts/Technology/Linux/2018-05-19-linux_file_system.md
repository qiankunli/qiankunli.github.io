---

layout: post
title: linux vfs轮廓
category: 技术
tags: Linux
keywords: network 

---

## 简介

* TOC
{:toc}

许式伟：怎么让很多的软件进程同时使用这些外置存储设备，而不会乱呢？直接基于物理的存储地址进行读写肯定是行不通的，**过上几个月你自己可能都不记得什么数据写到哪里了**。所以和内存管理不同，外部存储的管理，我们希望这些写到存储中的数据是“**自描述**”的某种数据格式，我们可以随时查看之前写了哪些内容，都什么时候写的。这就是文件系统的来源。

许式伟：存储它不应该只能保存一个文件，而是应该是多个。既然是多个，就需要组织这些文件。那么，怎么组织呢？操作系统的设计者们设计了文件系统这样的东西，来组织这些文件。虽然文件系统的种类有很多（比如：FAT32、NTFS、EXT3、EXT4 等等），但是它们有统一的抽象：文件系统是一颗树；节点要么是目录，要么是文件；文件必然是叶节点；根节点是目录，目录可以有子节点。

## 单个文件系统设计

来自彭东《操作系统实战》

**文件系统只是一个设备**，文件系统一定要有储存设备，HD 机械硬盘、SSD 固态硬盘、U 盘、各种 TF 卡等都属于存储设备，这些设备上的文件储存格式都不相同，甚至同一个硬盘上不同的分区的储存格式也不同。这个储存格式就是相应文件系统在储存设备上组织储存文件的方式。不难发现让文件系统成为 Cosmos 内核中一部分，是个非常愚蠢的想法。因此：文件系统组件是独立的与内核分开的；第二，操作系统需要动态加载和删除不同的文件系统组件。

关于文件系统存放文件数据的格式，类 UNIX 系统和 Windows 系统都采用了相同的方案，那就是**逻辑上认为一个文件就是一个可以动态增加、减少的线性字节数组**。我们如何把这个逻辑上的文件数据字节数组，映射到具体的储存设备上呢？现在的机械硬盘、SSD 固态硬盘、TF 卡，它们都是以储存块为单位储存数据的，一个储存块的大小可以是 512、1024、2048、4096 字节，访问这些储存设备的最小单位也是一个储存块，不像内存设备可以最少访问一个字节。

现在 PC 机上的文件数量都已经上十万的数量级了，如果把十万个文件顺序地排列在一起，要找出其中一个文件，那是非常困难的，所以，需要一个叫文件目录或者叫文件夹的东西，我们习惯称其为目录。这样我们就可以用不同的目录来归纳不同的文件。可以看出，整个文件层次结构就像是一棵倒挂的树。

```c
// 文件系统的超级块或者文件系统描述块
typedef struct s_RFSSUBLK
{
    spinlock_t rsb_lock;    //超级块在内存中使用的自旋锁
    uint_t rsb_mgic;        //文件系统标识
    uint_t rsb_vec;         //文件系统版本
    uint_t rsb_flg;         //标志
    uint_t rsb_stus;        //状态
    size_t rsb_sz;          //该数据结构本身的大小
    size_t rsb_sblksz;      //超级块大小
    size_t rsb_dblksz;      //文件系统逻辑储存块大小，我们这里用的是4KB
    uint_t rsb_bmpbks;      //位图的开始逻辑储存块
    uint_t rsb_bmpbknr;     //位图占用多少个逻辑储存块
    uint_t rsb_fsysallblk;  //文件系统有多少个逻辑储存块
    rfsdir_t rsb_rootdir;   //根目录，后面会看到这个数据结构的
}rfssublk_t;
// 目录
typedef struct s_RFSDIR
{
    uint_t rdr_stus;            //目录状态
    uint_t rdr_type;            //目录类型，可以是空类型、目录类型、文件类型、已删除的类型
    uint_t rdr_blknr;           //指向文件数据管理头的块号，不像内存可以用指针，只能按块访问
    char_t rdr_name[DR_NM_MAX]; //名称数组，大小为DR_NM_MAX
}rfsdir_t;
// 文件，包含文件名、状态、类型、创建时间、访问时间、大小，更为重要的是要知道该文件使用了哪些逻辑储存块。
typedef struct s_fimgrhd
{
    uint_t fmd_stus;//文件状态
    uint_t fmd_type;//文件类型：可以是目录文件、普通文件、空文件、已删除的文件
    uint_t fmd_flg;//文件标志
    uint_t fmd_sfblk;//文件管理头自身所在的逻辑储存块
    uint_t fmd_acss;//文件访问权限
    uint_t fmd_newtime;//文件的创建时间，换算成秒
    uint_t fmd_acstime;//文件的访问时间，换算成秒
    uint_t fmd_fileallbk;//文件一共占用多少个逻辑储存块
    uint_t fmd_filesz;//文件大小
    uint_t fmd_fileifstbkoff;//文件数据在第一块逻辑储存块中的偏移
    uint_t fmd_fileiendbkoff;//文件数据在最后一块逻辑储存块中的偏移
    uint_t fmd_curfwritebk;//文件数据当前将要写入的逻辑储存块
    uint_t fmd_curfinwbkoff;//文件数据当前将要写入的逻辑储存块中的偏移
    filblks_t fmd_fleblk[FBLKS_MAX];//文件占用逻辑储存块的数组，一共32个filblks_t结构
    uint_t fmd_linkpblk;//指向文件的上一个文件管理头的逻辑储存块
    uint_t fmd_linknblk;//指向文件的下一个文件管理头的逻辑储存块
}fimgrhd_t;
// 基于上述结构的驱动程序
drvstus_t rfs_entry(driver_t* drvp,uint_t val,void* p){……}
drvstus_t rfs_exit(driver_t* drvp,uint_t val,void* p){……}
drvstus_t rfs_open(device_t* devp,void* iopack){……}
drvstus_t rfs_close(device_t* devp,void* iopack){……}
drvstus_t rfs_read(device_t* devp,void* iopack){……}
drvstus_t rfs_write(device_t* devp,void* iopack){……}
drvstus_t rfs_lseek(device_t* devp,void* iopack){……}
drvstus_t rfs_ioctrl(device_t* devp,void* iopack){……}
drvstus_t rfs_dev_start(device_t* devp,void* iopack){……}
drvstus_t rfs_dev_stop(device_t* devp,void* iopack){……}
drvstus_t rfs_set_powerstus(device_t* devp,void* iopack){……}
drvstus_t rfs_enum_dev(device_t* devp,void* iopack){……}
drvstus_t rfs_flush(device_t* devp,void* iopack){……}
drvstus_t rfs_shutdown(device_t* devp,void* iopack){……}
```
格式化操作并不是把设备上所有的空间都清零，而是在这个设备上重建了文件系统用于管理文件的那一整套数据结构。

## vfs 数据结构 / 两个关系

[从文件 I/O 看 Linux 的虚拟文件系统](https://www.ibm.com/developerworks/cn/linux/l-cn-vfs/index.html)

![](/public/upload/linux/linux_vfs.png)

### 进程与超级块、文件、索引结点、目录项的关系

VFS 为了屏蔽各个文件系统的差异，就必须要定义一组通用的数据结构，规范各个文件系统的实现，每种结构都对应一套回调函数集合，这是典型的面向对象的设计方法。这些数据结构包含描述文件系统信息的超级块、表示文件名称的目录结构、描述文件自身信息的索引节点结构、表示打开一个文件的实例结构。

![](/public/upload/linux/linux_vfs_2.jpg)

**超级重点**：有了超级块和超级块函数集合结构，VFS 就能让一个文件系统的信息和表示变得规范了。也就是说，文件系统只要实现了 super_block 和super_operations 两个结构，就可以插入到 VFS 中了。Linux 系统中所有文件都是用目录组织的，对应数据结构 dentry 和dentry_operations。VFS 用 inode 结构表示一个文件索引结点，它里面包含文件权限、文件所属用户、文件访问和修改时间、文件数据块号等一个文件的全部信息，一个 inode 结构就对应一个文件，但这个 inode 结构是 VFS 使用的，跟某个具体文件系统上的“inode”结构并不是一一对应关系。inode 结构还有一套函数集合inode_operations，用于具体文件系统根据自己特有的信息，构造出 VFS 使用的 inode 结构。应用程序直接处理的就是文件，而不是超级块、索引节点或目录项，VFS 设计了一个文件对象结构file表示进程已打开的文件，包含了我们非常熟悉的信息，如访问模式、当前读写偏移等。进程每打开一个文件就会建立一个 file 结构实例，并将其地址放入数组中，最后返回对应的数组下标，就是我们调用 open 函数返回的那个整数。对于 file 结构，也有对应的函数集合 file_operations 结构。

**超级块、目录结构、文件索引节点，打开文件的实例，通过四大对象就可以描述抽象出一个文件系统了。而四大对象的对应的操作函数集合，又由具体的文件系统来实现，这两个一结合，一个文件系统的状态和行为都具备了**。

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

    struct files_struct {//打开的文件集
            atomic_t count;              /*结构的使用计数*/
            ……
            int max_fds;                 /*文件对象数的上限*/
            int max_fdset;               /*文件描述符的上限*/
            int next_fd;                 /*下一个文件描述符*/
            struct file ** fd;           /*全部文件对象数组*/
            ……
    };
    
    struct fs_struct {//建立进程与文件系统的关系
            atomic_t count;              /*结构的使用计数*/
            rwlock_t lock;               /*保护该结构体的锁*/
            int umask；                  /*默认的文件访问权限*/
            struct dentry * root;        /*根目录的目录项对象*/
            struct dentry * pwd;         /*当前工作目录的目录项对象*/
            struct dentry * altroot；    /*可供选择的根目录的目录项对象*/
            struct vfsmount * rootmnt;   /*根目录的安装点对象*/
            struct vfsmount * pwdmnt;    /*pwd的安装点对象*/
            struct vfsmount * altrootmnt;/*可供选择的根目录的安装点对象*/
    };

### 超级块、安装点和具体的文件系统的关系

linux 和 windows 一个很大区别就是，window 有c盘、d盘等，而linux 则都是从根目录开始，好像只有一个磁盘一样。为了实现这个效果，linux 就要引入vfsmount 等概念。

![](/public/upload/linux/linux_vfs_1.jpg)

1. 被Linux支持的文件系统，都有且仅有一个file_system_type结构
2. 每安装一个文件系统，就对应有一个超级块和安装点

## 文件访问

查找时，在遍历路径的过程中，会逐层地将各个路径组成部分解析成目录项对象，如果此目录项对象在目录项缓存中，则直接从缓存中获得；如果该目录项在缓存中不存在，则进行一次实际的读盘操作，从磁盘中读取该目录项所对应的索引节点。得到索引节点后，则建立索引节点与该目录项的联系。如此循环，直到最终找到目标文件对应的目录项，也就找到了索引节点，这样就建立了文件对象与实际的物理文件的关联。

![](/public/upload/linux/linux_file_class_diagram.png)

文件对象所对应的文件操作函数 列表是通过索引结点的域i_fop得到的

## mount 

### 数据结构

struct mount代表着一个mount实例，最开始所有字段都在 vfsmount ，后来拆分为vfsmount 和mount 两个。
```c
struct mount {
    struct hlist_node mnt_hash;
    struct mount *mnt_parent;   // 装载点所在的父文件系统
    struct dentry *mnt_mountpoint; // 装载点在父文件系统中的dentry
    struct vfsmount mnt;
    union {
        struct rcu_head mnt_rcu;
        struct llist_node mnt_llist;
    };
    struct list_head mnt_mounts;	/* list of children, anchored here */
    struct list_head mnt_child;	/* and going through their mnt_child */
    struct list_head mnt_instance;	/* mount instance on sb->s_mounts */
    const char *mnt_devname;	/* Name of device e.g. /dev/dsk/hda1 */
    struct list_head mnt_list;
    ......
} __randomize_layout;
struct vfsmount {
    struct dentry *mnt_root; // 当前文件系统根目录的dentry
    struct super_block *mnt_sb;	// 指向超级块的指针
    int mnt_flags;
} __randomize_layout;
```
一个文件系统可以挂装载到不同的挂载点。所以**文件系统树的一个位置要由<mount, dentry>二元组（或者说<vfsmount, dentry>）来确定**，也有说是根据 dentry 的状态位确定被 被mount，进而根据path hash从mount hashtable 里获取的mount struct。
```c
struct path {
	struct vfsmount *mnt;  
	struct dentry *dentry;
};
```

### mount 过程

[linux文件系统之mount流程分析](https://www.cnblogs.com/cslunatic/p/3683117.html)

一个磁盘如何被使用？

1. `insmod xx.ko` 加载块设备驱动
2. `mknod /dev/xx type major minor` 创建设备文件，实质将文件操作与设备驱动程序关联，对于字符设备，操作`/dev/xx`便是读写字符设备了，对于块设备，会复杂一点。
3. 例如，`mount -t ext3 /dev/sdb /mnt/alan`，`/dev/sdb`块设备被mount到`/mnt/alan`目录。

那么mount 如何实现这个神奇的效果呢？mount系统调用 入口

```c
SYSCALL_DEFINE5(mount, char __user *, dev_name, char __user *, dir_name, char __user *, type, unsigned long, flags, void __user *, data){
    ......
    ret = do_mount(kernel_dev, dir_name, kernel_type, flags, options);
    ......
}
```

接下里的调用链：do_mount->do_new_mount

```c
static int do_new_mount(struct path *path, const char *fstype, int flags,
            int mnt_flags, const char *name, void *data){
    ...
    mnt = vfs_kern_mount(type, flags, name, data);
    ...
    err = do_add_mount(real_mount(mnt), path, mnt_flags);
    ...
}
```

do_new_mount()函数主要分成两大部分：

1. 建立vfsmount对象和superblock对象，必要时从设备上获取文件系统元数据；
2. 将vfsmount对象加入到mount树和Hash Table中，并且将原来的dentry对象无效掉。

`/dev/sdb`被mount之后，用户想要访问该设备上的一个文件ab.c，假设该文件的地址为：`/mnt/alan/ab.c`。

1. 在打开该文件的时候，首先需要进行path解析。
2. 在解析到`/mnt/alan`的时候，得到`/mnt/alan`的dentry目录项，并且发现该目录项已经被标识为DCACHE_MOUNTED。
2. 之后，会采用`/mnt/alan`计算HASH值去检索VFSMOUNT Hash Table，得到对应的vfsmount对象。
3. 然后采用vfsmount指向的mnt_root目录项替代`/mnt/alan`原来的dentry，从而实现了dentry和inode的重定向。
4. 在新的dentry的基础上，解析程序继续执行，最终得到表示ab.c文件的inode对象。

总结一下就是：[Mount Point Definition](http://www.linfo.org/mount_point.html)The mount point becomes the root directory of the newly added filesystem, and that filesystem becomes accessible from that directory. 

### vfs_kern_mount

    struct vfsmount *
    vfs_kern_mount(struct file_system_type *type, int flags, const char *name, void *data){
        ......
        mnt = alloc_vfsmnt(name);
        ......
        // 从文件系统中读取超级块
        root = mount_fs(type, flags, name, data);
        ......
        mnt->mnt.mnt_root = root;
        mnt->mnt.mnt_sb = root->d_sb;
        mnt->mnt_mountpoint = mnt->mnt.mnt_root;
        mnt->mnt_parent = mnt;
        list_add_tail(&mnt->mnt_instance, &root->d_sb->s_mounts);
        return &mnt->mnt;
    }

    struct dentry * mount_fs(struct file_system_type *type, int flags, const char *name, void *data)
    {
        struct dentry *root;
        struct super_block *sb;
        char *secdata = NULL;
        int error = -ENOMEM;
        ...
        root = type->mount(type, flags, name, data);
        ...
        sb = root->d_sb;
        ...	
    }


1. alloc_vfsmnt，vfs_kern_mount 先是创建 struct mount 结构，内部包含一个vfsmount 结构
2. mount_fs，mount_fs()函数中会调用特定文件系统的mount方法，对于 `/dev/sdb`设备上的ext3文件系统，`ext3_mount--> mount_bdev`，Mount_bdev()函数主要完成superblock对象的内存初始化，并且加入到全局superblock链表中。
3. Vfsmount中的mnt_root指向superblock对象的s_root根目录项。

### do_add_mount

`do_add_mount--> graft_tree--> attach_recursive_mnt`将创建的vfsmount对象加入到mount树和VFSMOUNT Hash Table中，并且将老的dentry目录项标识成DCACHE_MOUNTED，一旦dentry被标识成DCACHE_MOUNTED，也就意味着在访问路径上对其进行了屏蔽。

## 挂载方式

**如果将mount的过程理解为：inode被替代的过程。**除了将设备mount到rootfs上，根据被替代方式的不同，mount的花样可多了。

### bind mount

`mount --bind olddir newdir`

ln分为软链接和硬链接

硬链接只能连文件 你删除它源文件就被删了

软链接相当于windows的快捷方式，文件和目录都可以连，但是你删出它只是删除的这个快捷方式

在操作系统中ln -s和mount -bind也许使用起来没多少差别,但是ftp目录里貌似不能放软连接。把一个文件夹mount到另一个地方，等于在把它挂载到那个地方当成一个磁盘，进到两个文件夹中操作 都是等价的，删除一个文件也会在两边同时删除。

### union mount

联合文件系统是一种 堆叠文件系统，通过不停地叠加文件实现对文件的修改。其中，增加操作通过在读写层增加新文件实现，删除操作一般通过添加额外的删除属性文件实现，比如删除a.file时读写层增加一个a.file.delete文件。修改只读层文件时，需要先复制一份儿文件到读写层，然后修改复制的文件。

参见[Union mount](https://en.wikipedia.org/wiki/Union_mount)，In computer operating systems, union mounting is a way of combining multiple directories into one that appears to contain their combined contents.

union mount的使用场景——读写只读文件系统，As an example application of union mounting, consider the need to update the information contained on a CD-ROM or DVD. While a CD-ROM is not writable, one can overlay the CD's mount point with a writable directory in a union mount. Then, updating files in the union directory will cause them to end up in the writable directory, giving the illusion that the CD-ROM's contents have been updated. 

Union FileSystem的核心逻辑是Union Mount，它支持把一个目录A和另一个目录B union，提供一个虚拟的C目录（目录union的语义）。对于特定的权限设置策略，如果设置A目录可写，B目录只读。用户对目录C的读取就是A加上B的内容，而对C目录里文件写入和改写则会保存在目录A上。这样，就有了A在B上层的感觉。
[DOCKER基础技术：AUFS](http://coolshell.cn/articles/17061.html)

[Docker存储驱动简介](https://linux.cn/thread-16017-1-1.html)

## rootfs

rootfs是基于内存的文件系统，所有操作都在内存中完成；也没有实际的存储设备，所以不需要设备驱动程序的参与。基于以上原因，Linux在启动阶段使用rootfs文件系统。

参见`https://www.kernel.org/doc/Documentation/filesystems/ramfs-rootfs-initramfs.txt`

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

[存储之道 - 51CTO技术博客 中的《一个IO的传奇一生》](http://alanwu.blog.51cto.com/3652632/d-8)



