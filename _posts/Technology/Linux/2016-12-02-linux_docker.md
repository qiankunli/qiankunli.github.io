---

layout: post
title: docker中涉及到的一些linux知识
category: 技术
tags: Linux
keywords: network 

---

## 简介

## 镜像文件

An archive file is a file that is composed of one or more computer files along with metadata. Archive files are used to collect multiple data files together into a single file for easier portability and storage, or simply to compress files to use less storage space. Archive files often store directory structures, error detection and correction information, arbitrary comments, and sometimes use built-in encryption.

rootfs是基于内存的文件系统，所有操作都在内存中完成；也没有实际的存储设备，所以不需要设备驱动程序的参与。基于以上原因，Linux在启动阶段使用rootfs文件系统，当磁盘驱动程序和磁盘文件系统成功加载后，linux系统会将系统根目录从rootfs切换到磁盘文件系统。

所以呢，文件系统有内存文件系统，磁盘文件系统，还有基于磁盘文件系统之上的联合文件系统。

linux文件系统中重要的数据结构有：文件、挂载点、超级块、目录项、索引节点等。图中含有两个文件系统（红色和绿色表示的部分），并且绿色文件系统挂载在红色文件系统tmp目录下。一般来说，每个文件系统在VFS层都是由挂载点、超级块、目录和索引节点组成；当挂载一个文件系统时，实际也就是创建这四个数据结构的过程，因此这四个数据结构的地位很重要，关系也很紧密。由于VFS要求实际的文件系统必须提供以上数据结构，所以不同的文件系统在VFS层可以互相访问。
    如果进程打开了某个文件，还会创建file(文件)数据结构，这样进程就可以通过file来访问VFS的文件系统了。

![](/public/upload/linux/linux_fs.png)

这个图从上往下看，可以知道

1. 挂载点和超级块本身